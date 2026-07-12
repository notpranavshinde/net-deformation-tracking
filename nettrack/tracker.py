"""Topology-constrained stereo tracking for visually identical net markers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import lil_matrix

from .bootstrap import bootstrap_mesh
from .detect import Detection, detect_in_window, detect_markers
from .geometry import StereoCalibration, project_points, triangulate_point
from .track_quality import node_edge_deviation, suspect_node_reports
from .topology import NetTopology


@dataclass
class MeshTrackerConfig:
    """Numerical and confidence controls for :class:`MeshTracker`."""

    gate_radius_factor: float = 0.48
    edge_weight: float = 0.1
    temporal_weight: float = 0.01
    robust_loss_scale: float = 2.5
    max_inferred_streak: int = 20
    velocity_damping: float = 0.35
    min_gate_px: float = 8.0
    max_gate_px: float = 90.0
    audit_sigma: float = 4.0
    audit_min_px: float = 4.0
    rest_refine_frames: int = 10
    setup_check_frames: int = 7
    setup_check_min_hits: int = 2
    setup_cluster_radius_factor: float = 0.28
    reacquire_after: int = 2
    excursion_threshold: float = 0.08
    excursion_min_len: int = 5
    solver_ftol: float = 1e-4
    solver_xtol: float = 1e-4
    solver_gtol: float = 1e-4
    solver_max_nfev: int = 5


@dataclass
class MeshFrameResult:
    """Solved state, observations, statuses, and metrics for one frame pair."""

    pair_index: int
    left_frame: int
    right_frame: int
    positions: np.ndarray
    statuses: tuple[str, ...]
    left_assignments: tuple[Detection | None, ...]
    right_assignments: tuple[Detection | None, ...]
    left_output: np.ndarray
    right_output: np.ndarray
    valid: np.ndarray
    inferred_quality: np.ndarray
    metrics: dict


@dataclass
class MeshTrackResult:
    """Complete clip result plus the JSON-ready paper metrics report."""

    topology: NetTopology
    frames: list[MeshFrameResult]
    rest_lengths: np.ndarray
    report: dict = field(default_factory=dict)

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([frame.positions for frame in self.frames], dtype=np.float64)


class MeshTracker:
    """Track a single stereo 3-D mesh with fixed identity and connectivity.

    Reprojection residuals are evaluated in distorted pixel coordinates.  Edge
    and temporal residuals are converted to pixel-like units using the median
    projected edge spacing, allowing one robust-loss scale to cover all terms.
    """

    def __init__(
        self,
        calibration: StereoCalibration,
        topology: NetTopology,
        config: MeshTrackerConfig | None = None,
        detection_function: Callable[..., list[Detection]] = detect_markers,
    ) -> None:
        self.calibration = calibration
        self.topology = topology
        self.config = config or MeshTrackerConfig()
        self._detect = detection_function
        self._edge_indices = topology.edge_indices
        self._expected_detection_count = len(topology)

    @staticmethod
    def _points(points: Sequence) -> np.ndarray:
        parsed = []
        for point in points:
            if isinstance(point, dict):
                parsed.append((point.get("x", point.get("u")), point.get("y", point.get("v"))))
            else:
                parsed.append((point[0], point[1]))
        return np.asarray(parsed, dtype=np.float64).reshape(-1, 2)

    def _projected_spacing(self, xyz: np.ndarray) -> float:
        if len(self._edge_indices) == 0:
            return 30.0
        values = []
        for camera in (self.calibration.left, self.calibration.right):
            uv = project_points(xyz, camera)
            delta = uv[self._edge_indices[:, 0]] - uv[self._edge_indices[:, 1]]
            values.extend(np.linalg.norm(delta, axis=1).tolist())
        return max(float(np.median(values)), 2.0)

    def _assign(
        self, predicted_uv: np.ndarray, detections: Sequence[Detection], gate_px: float
    ) -> list[Detection | None]:
        assigned: list[Detection | None] = [None] * len(predicted_uv)
        if not detections or len(predicted_uv) == 0:
            return assigned
        observed = np.asarray([(item.u, item.v) for item in detections], dtype=np.float64)
        distance = np.linalg.norm(predicted_uv[:, None, :] - observed[None, :, :], axis=2)
        quality = np.asarray([item.quality for item in detections], dtype=np.float64)
        cost = distance / (0.5 + 0.5 * quality[None, :])
        cost[distance > gate_px] = 1e9
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if distance[row, col] <= gate_px:
                assigned[int(row)] = detections[int(col)]
        return assigned

    def _sparsity(
        self, left: Sequence[Detection | None], right: Sequence[Detection | None]
    ):
        n, edge_count = len(self.topology), len(self._edge_indices)
        observation_count = sum(item is not None for item in left) + sum(
            item is not None for item in right
        )
        rows = 2 * observation_count + edge_count + 3 * n
        pattern = lil_matrix((rows, 3 * n), dtype=np.int8)
        row = 0
        for assignments in (left, right):
            for index, detection in enumerate(assignments):
                if detection is not None:
                    pattern[row:row + 2, 3 * index:3 * index + 3] = 1
                    row += 2
        for a, b in self._edge_indices:
            pattern[row, 3 * a:3 * a + 3] = 1
            pattern[row, 3 * b:3 * b + 3] = 1
            row += 1
        for index in range(n):
            pattern[row:row + 3, 3 * index:3 * index + 3] = 1
            row += 3
        return pattern.tocsr()

    def _solve(
        self,
        initial: np.ndarray,
        predicted: np.ndarray,
        left: Sequence[Detection | None],
        right: Sequence[Detection | None],
        rest_lengths: np.ndarray,
        spacing_px: float,
    ):
        config = self.config
        edge_scale = config.edge_weight * spacing_px
        temporal_scale = config.temporal_weight * spacing_px / max(
            float(np.median(rest_lengths)) if len(rest_lengths) else 1.0, 1e-9
        )

        def residual(flat):
            xyz = flat.reshape(-1, 3)
            values = []
            for camera, assignments in (
                (self.calibration.left, left),
                (self.calibration.right, right),
            ):
                uv = project_points(xyz, camera)
                for index, detection in enumerate(assignments):
                    if detection is not None:
                        weight = np.sqrt(max(0.15, detection.quality))
                        values.extend(weight * (uv[index] - (detection.u, detection.v)))
            if len(self._edge_indices):
                delta = xyz[self._edge_indices[:, 0]] - xyz[self._edge_indices[:, 1]]
                lengths = np.linalg.norm(delta, axis=1)
                values.extend(edge_scale * (lengths - rest_lengths) / np.maximum(rest_lengths, 1e-9))
            values.extend((temporal_scale * (xyz - predicted)).reshape(-1))
            return np.asarray(values, dtype=np.float64)

        return least_squares(
            residual,
            initial.reshape(-1),
            method="trf",
            loss="soft_l1",
            f_scale=config.robust_loss_scale,
            jac_sparsity=self._sparsity(left, right),
            ftol=config.solver_ftol,
            xtol=config.solver_xtol,
            gtol=config.solver_gtol,
            max_nfev=config.solver_max_nfev,
        )

    def _audit(
        self,
        xyz: np.ndarray,
        assignments: list[Detection | None],
        camera,
    ) -> int:
        uv = project_points(xyz, camera)
        present = [(i, item) for i, item in enumerate(assignments) if item is not None]
        if not present:
            return 0
        errors = np.asarray(
            [np.linalg.norm(uv[i] - (item.u, item.v)) for i, item in present], dtype=float
        )
        median = float(np.median(errors))
        mad = 1.4826 * float(np.median(np.abs(errors - median)))
        threshold = max(self.config.audit_min_px, median + self.config.audit_sigma * max(mad, 0.25))
        dropped = 0
        for (index, _item), error in zip(present, errors):
            if error > threshold:
                assignments[index] = None
                dropped += 1
        return dropped

    def track(
        self,
        left_frames: Iterable[np.ndarray],
        right_frames: Iterable[np.ndarray],
        setup_left: Sequence,
        setup_right: Sequence,
        *,
        frame_indices: Iterable[tuple[int, int]] | None = None,
        left_roi=None,
        right_roi=None,
        progress: Callable[[int, dict], None] | None = None,
        bootstrap_callback: Callable[[dict], None] | None = None,
    ) -> MeshTrackResult:
        """Track paired frames; setup arrays are aligned with topology node order."""
        left_iter, right_iter = iter(left_frames), iter(right_frames)
        indices = iter(frame_indices) if frame_indices is not None else None
        setup_pairs, setup_indices = [], []
        for pair_index in range(max(1, self.config.setup_check_frames)):
            try:
                pair = next(left_iter), next(right_iter)
            except StopIteration:
                break
            setup_pairs.append(pair)
            setup_indices.append(next(indices) if indices is not None else (pair_index, pair_index))
        if not setup_pairs:
            raise ValueError("At least one stereo frame pair is required")
        setup_l, setup_r = self._points(setup_left), self._points(setup_right)
        if len(setup_l) != len(self.topology) or len(setup_r) != len(self.topology):
            raise ValueError("Setup point counts must equal topology node count")

        init_started = time.perf_counter()
        setup_check = bootstrap_mesh(
            [pair[0] for pair in setup_pairs], [pair[1] for pair in setup_pairs],
            setup_l, setup_r, self.topology, self.calibration,
            left_roi=left_roi, right_roi=right_roi, detection_function=self._detect,
            setup_check_frames=self.config.setup_check_frames,
            setup_check_min_hits=self.config.setup_check_min_hits,
            cluster_radius_factor=self.config.setup_cluster_radius_factor,
        )
        positions = setup_check.positions
        model_l, model_r = setup_check.left_model, setup_check.right_model
        self.topology = setup_check.topology
        self._edge_indices = self.topology.edge_indices
        if not len(self.topology):
            raise ValueError("Setup check could not establish any stereo marker identities")
        if bootstrap_callback is not None:
            bootstrap_callback(setup_check.report)
        initialization_timing = {
            "setup_check": time.perf_counter() - init_started,
            "total": time.perf_counter() - init_started,
        }
        if not np.all(np.isfinite(positions)):
            raise ValueError("Initial setup triangulated to non-finite coordinates")
        rest = np.linalg.norm(
            positions[self._edge_indices[:, 0]] - positions[self._edge_indices[:, 1]], axis=1
        ) if len(self._edge_indices) else np.empty(0)
        velocity = np.zeros_like(positions)
        inferred_streak = np.zeros(len(self.topology), dtype=np.int32)
        rest_history = [rest.copy()]
        frames: list[MeshFrameResult] = []
        timings_total = {"detect_s": 0.0, "assign_s": 0.0, "solve_s": 0.0, "total_s": 0.0}

        remaining_pairs = zip(left_iter, right_iter)
        all_pairs = chain(setup_pairs, remaining_pairs)
        all_indices = chain(setup_indices, indices) if indices is not None else None
        for pair_index, (left_image, right_image) in enumerate(all_pairs):
            started = time.perf_counter()
            left_frame, right_frame = next(all_indices) if all_indices is not None else (pair_index, pair_index)
            predicted = positions + self.config.velocity_damping * velocity
            spacing = self._projected_spacing(predicted)
            gate = float(np.clip(self.config.gate_radius_factor * spacing, self.config.min_gate_px, self.config.max_gate_px))
            tick = time.perf_counter()
            projected_predictions = (
                project_points(predicted, self.calibration.left),
                project_points(predicted, self.calibration.right),
            )
            det_l = self._detect(left_image, model_l, roi=left_roi, expected_count=self._expected_detection_count)
            det_r = self._detect(right_image, model_r, roi=right_roi, expected_count=self._expected_detection_count)
            reacquired = [0, 0]
            for side, (image, model, found, projected) in enumerate(zip(
                (left_image, right_image), (model_l, model_r), (det_l, det_r), projected_predictions,
            )):
                for index in np.flatnonzero(inferred_streak >= self.config.reacquire_after):
                    item = detect_in_window(image, model, projected[index], gate, True)
                    if item is None:
                        continue
                    if any(np.hypot(item.u - old.u, item.v - old.v) < 2.0 for old in found):
                        continue
                    found.append(item)
                    reacquired[side] += 1
            detect_s = time.perf_counter() - tick
            tick = time.perf_counter()
            left_assignment = self._assign(projected_predictions[0], det_l, gate)
            right_assignment = self._assign(projected_predictions[1], det_r, gate)
            assign_s = time.perf_counter() - tick
            tick = time.perf_counter()
            solve_initial = predicted.copy()
            for index, (left_item, right_item) in enumerate(zip(left_assignment, right_assignment)):
                if left_item is not None and right_item is not None:
                    solve_initial[index] = triangulate_point(
                        (left_item.u, left_item.v), (right_item.u, right_item.v), self.calibration
                    )
            solved = self._solve(solve_initial, predicted, left_assignment, right_assignment, rest, spacing)
            new_positions = solved.x.reshape(-1, 3)
            audit_drops = self._audit(new_positions, left_assignment, self.calibration.left)
            audit_drops += self._audit(new_positions, right_assignment, self.calibration.right)
            if audit_drops:
                solved = self._solve(new_positions, predicted, left_assignment, right_assignment, rest, spacing)
                new_positions = solved.x.reshape(-1, 3)
            solve_s = time.perf_counter() - tick

            measured_l = np.asarray([item is not None for item in left_assignment])
            measured_r = np.asarray([item is not None for item in right_assignment])
            measured = measured_l | measured_r
            inferred_streak[measured] = 0
            inferred_streak[~measured] += 1
            valid = inferred_streak <= self.config.max_inferred_streak
            statuses = []
            for l_ok, r_ok, is_valid in zip(measured_l, measured_r, valid):
                if l_ok and r_ok:
                    statuses.append("measured-both")
                elif l_ok:
                    statuses.append("measured-left")
                elif r_ok:
                    statuses.append("measured-right")
                elif is_valid:
                    statuses.append("inferred")
                else:
                    statuses.append("lost")
            projected_l = project_points(new_positions, self.calibration.left)
            projected_r = project_points(new_positions, self.calibration.right)
            output_l, output_r = projected_l.copy(), projected_r.copy()
            for index, item in enumerate(left_assignment):
                if item is not None:
                    output_l[index] = item.u, item.v
            for index, item in enumerate(right_assignment):
                if item is not None:
                    output_r[index] = item.u, item.v
            confidence = np.power(0.78, inferred_streak.astype(float))
            one_view_quality = np.asarray([
                max(l.quality if l else 0.0, r.quality if r else 0.0)
                for l, r in zip(left_assignment, right_assignment)
            ])
            inferred_quality = np.where(measured, np.maximum(0.05, one_view_quality) * 0.8, 0.65 * confidence)
            inferred_quality[~valid] = 0.0

            velocity = new_positions - positions
            positions = new_positions
            edge_lengths = np.linalg.norm(
                positions[self._edge_indices[:, 0]] - positions[self._edge_indices[:, 1]], axis=1
            ) if len(self._edge_indices) else np.empty(0)
            if pair_index + 1 < self.config.rest_refine_frames and len(rest):
                rest_history.append(edge_lengths)
                history = np.asarray(rest_history)
                center = np.median(history, axis=0)
                deviation = np.abs(history - center)
                cutoff = 3.0 * np.maximum(np.median(deviation, axis=0), 1e-9)
                rest = np.asarray([
                    np.mean(history[:, i][deviation[:, i] <= cutoff[i]]) for i in range(len(rest))
                ])
            per_node_deviation = node_edge_deviation(
                self._edge_indices, edge_lengths, rest, len(self.topology),
            )
            total_s = time.perf_counter() - started
            metrics = {
                "pair_index": pair_index, "left_frame": int(left_frame), "right_frame": int(right_frame),
                "detections_left": len(det_l), "detections_right": len(det_r),
                "assignments_left": int(measured_l.sum()), "assignments_right": int(measured_r.sum()),
                "inferred": int(np.count_nonzero(np.asarray(statuses) == "inferred")),
                "lost": int(np.count_nonzero(np.asarray(statuses) == "lost")),
                "audit_drops": int(audit_drops), "solver_iterations": int(solved.nfev),
                "reacquired_left": reacquired[0], "reacquired_right": reacquired[1],
                "node_edge_relative_deviation": per_node_deviation.tolist(),
                "edge_length_mean": float(np.mean(edge_lengths)) if len(edge_lengths) else 0.0,
                "edge_length_std": float(np.std(edge_lengths)) if len(edge_lengths) else 0.0,
                "rest_length_mean": float(np.mean(rest)) if len(rest) else 0.0,
                "rest_length_std": float(np.std(rest)) if len(rest) else 0.0,
                "edge_length_mean_minus_rest": float(np.mean(edge_lengths - rest)) if len(rest) else 0.0,
                "edge_relative_rmse_vs_rest": float(np.sqrt(np.mean(((edge_lengths - rest) / rest) ** 2))) if len(rest) else 0.0,
                "timing_s": {"detect": detect_s, "assign": assign_s, "solve": solve_s, "total": total_s},
            }
            for key, value in (("detect_s", detect_s), ("assign_s", assign_s), ("solve_s", solve_s), ("total_s", total_s)):
                timings_total[key] += value
            frames.append(MeshFrameResult(
                pair_index, int(left_frame), int(right_frame), positions.copy(), tuple(statuses),
                tuple(left_assignment), tuple(right_assignment), output_l, output_r, valid.copy(),
                inferred_quality, metrics,
            ))
            if progress is not None:
                progress(pair_index + 1, metrics)

        if not frames:
            raise ValueError("The stereo iterables did not contain a complete frame pair")
        node_reports, suspect_count = suspect_node_reports(
            frames, self.topology, self.config.excursion_threshold, self.config.excursion_min_len,
        )
        one_view_count = sum(len(node["one_view_intervals"]) for node in node_reports)
        totals = {
            "frames": len(frames),
            "inferred_vertex_frames": sum(frame.metrics["inferred"] for frame in frames),
            "lost_vertex_frames": sum(frame.metrics["lost"] for frame in frames),
            "audit_drops": sum(frame.metrics["audit_drops"] for frame in frames),
            "detections_left": sum(frame.metrics["detections_left"] for frame in frames),
            "detections_right": sum(frame.metrics["detections_right"] for frame in frames),
            "assignments_left": sum(frame.metrics["assignments_left"] for frame in frames),
            "assignments_right": sum(frame.metrics["assignments_right"] for frame in frames),
            "solver_iterations": sum(frame.metrics["solver_iterations"] for frame in frames),
            "reacquired_left": sum(frame.metrics["reacquired_left"] for frame in frames),
            "reacquired_right": sum(frame.metrics["reacquired_right"] for frame in frames),
            "suspect_intervals": suspect_count,
            "one_view_intervals": one_view_count,
            "timing_s": timings_total,
            "mean_seconds_per_pair": timings_total["total_s"] / len(frames),
        }
        report = {
            "schema_version": 2,
            "coordinate_residual_space": "distorted_pixels",
            "node_ids": list(self.topology.node_ids),
            "edges": [list(edge) for edge in self.topology.edges],
            "config": asdict(self.config),
            "rest_lengths": rest.tolist(),
            "initialization_timing_s": initialization_timing,
            "setup_check": setup_check.report,
            "nodes": node_reports,
            "frames": [frame.metrics for frame in frames],
            "totals": totals,
        }
        return MeshTrackResult(self.topology, frames, rest.copy(), report)
