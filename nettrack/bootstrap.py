"""Structure-first identity bootstrap for a calibrated stereo marker net."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .colormodel import MarkerColorModel
from .detect import Detection, detect_markers
from .geometry import StereoCalibration, project_points, triangulate_point
from .setup_check import (
    boundary_edge_suspects, cluster_detections, extrapolate_missing_boundaries, repair_boundary_shifts,
    rescue_across_frames,
)
from .topology import NetTopology


@dataclass
class BootstrapResult:
    """Established mesh state plus models, assignments, and JSON-ready audit."""

    topology: NetTopology
    positions: np.ndarray
    left_points: np.ndarray
    right_points: np.ndarray
    left_assignments: tuple[Detection, ...]
    right_assignments: tuple[Detection, ...]
    left_model: MarkerColorModel
    right_model: MarkerColorModel
    report: dict


def _grid_xy(topology: NetTopology) -> np.ndarray:
    cols = topology.grid_cols
    if not cols:
        vertical = sorted({abs(a - b) for a, b in topology.edges if abs(a - b) > 1})
        cols = vertical[0] if vertical else len(topology)
    return np.asarray([(node_id % cols, node_id // cols) for node_id in topology.node_ids], float)


def _spacing(points: np.ndarray, topology: NetTopology) -> float:
    edges = topology.edge_indices
    if not len(edges):
        return 30.0
    values = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    values = values[np.isfinite(values) & (values > 1.0)]
    return float(np.median(values)) if len(values) else 30.0


def _smooth_fit(grid: np.ndarray, hints: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Robust projective seed with a smooth non-projective residual surface."""
    finite = np.all(np.isfinite(hints), axis=1)
    seed = np.full_like(hints, np.nan)
    inliers = finite.copy()
    if np.count_nonzero(finite) >= 4:
        cv2.setRNGSeed(0)
        matrix, mask = cv2.findHomography(
            grid[finite], hints[finite], cv2.RANSAC, max(3.0, 0.34 * spacing),
            maxIters=3000, confidence=0.999,
        )
        if matrix is not None:
            seed = cv2.perspectiveTransform(grid[:, None, :].astype(np.float64), matrix)[:, 0]
            inliers[finite] = mask.reshape(-1).astype(bool)
    if not np.all(np.isfinite(seed)):
        design = np.column_stack((np.ones(len(grid)), grid, grid[:, 0] * grid[:, 1]))
        coef, *_ = np.linalg.lstsq(design[finite], hints[finite], rcond=None)
        seed = design @ coef
        residual = np.linalg.norm(seed - hints, axis=1)
        center = float(np.median(residual[finite]))
        mad = 1.4826 * float(np.median(np.abs(residual[finite] - center)))
        inliers = finite & (residual <= max(3.0, center + 3.5 * max(mad, 0.5)))
    centered_grid = grid[inliers] - np.mean(grid[inliers], axis=0) if np.any(inliers) else np.empty((0, 2))
    if np.count_nonzero(inliers) >= 3 and np.linalg.matrix_rank(centered_grid) == 2:
        source_grid, source_uv = grid[inliers], hints[inliers]
        refined = np.empty_like(seed)
        count = min(20, len(source_grid))
        for index, target in enumerate(grid):
            distance = np.linalg.norm(source_grid - target, axis=1)
            nearest = np.argsort(distance, kind="stable")[:count]
            local_grid, local_uv = source_grid[nearest], source_uv[nearest]
            design = np.column_stack((np.ones(count), local_grid))
            weight = 1.0 / np.maximum(0.35, distance[nearest])
            coefficient, *_ = np.linalg.lstsq(design * weight[:, None], local_uv * weight[:, None], rcond=None)
            refined[index] = np.array([1.0, *target]) @ coefficient
        seed = refined
    return seed, inliers


def _local_spacing(predicted: np.ndarray, topology: NetTopology) -> np.ndarray:
    values = np.full(len(predicted), np.nan)
    for index, node_id in enumerate(topology.node_ids):
        neighbors = [topology.index(value) for value in topology.neighbors(node_id)]
        if neighbors:
            values[index] = np.median(np.linalg.norm(predicted[neighbors] - predicted[index], axis=1))
    fallback = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 30.0
    return np.where(np.isfinite(values), values, fallback)


def _assign(predicted: np.ndarray, detections: Sequence[Detection], gates: np.ndarray):
    assigned: list[Detection | None] = [None] * len(predicted)
    if not detections:
        return assigned
    observed = np.asarray([(item.u, item.v) for item in detections], float)
    distance = np.linalg.norm(predicted[:, None] - observed[None], axis=2)
    quality = np.asarray([item.quality for item in detections], float)
    cost = distance / (0.65 + 0.35 * quality[None])
    cost[distance > gates[:, None]] = 1e8
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols):
        if distance[row, col] <= gates[row]:
            assigned[int(row)] = detections[int(col)]
    return assigned


def _anchored_assign(predicted, hints, hint_inliers, detections, gates, spacing):
    """Lock robust close hints, then globally solve only the uncertain remainder."""
    assigned: list[Detection | None] = [None] * len(predicted)
    anchor_nodes = np.flatnonzero(hint_inliers)
    used: set[int] = set()
    if len(anchor_nodes) and detections:
        observed = np.asarray([(item.u, item.v) for item in detections], float)
        distance = np.linalg.norm(hints[anchor_nodes, None] - observed[None], axis=2)
        cost = distance.copy()
        anchor_gates = np.maximum(6.0, 0.30 * spacing[anchor_nodes])
        cost[distance > anchor_gates[:, None]] = 1e8
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if distance[row, col] <= anchor_gates[row]:
                assigned[int(anchor_nodes[row])] = detections[int(col)]
                used.add(int(col))
    remaining_nodes = [index for index, item in enumerate(assigned) if item is None]
    remaining_detections = [item for index, item in enumerate(detections) if index not in used]
    uncertain = _assign(predicted[remaining_nodes], remaining_detections, gates[remaining_nodes])
    for index, item in zip(remaining_nodes, uncertain):
        assigned[index] = item
    return assigned


def _view_bootstrap(hints, topology, detections):
    grid = _grid_xy(topology)
    base_spacing = _spacing(hints, topology)
    predicted, hint_inliers = _smooth_fit(grid, hints, base_spacing)
    if detections:
        observed = np.asarray([(item.u, item.v) for item in detections], float)
        marker_distance = np.min(np.linalg.norm(hints[:, None] - observed[None], axis=2), axis=1)
        close = marker_distance <= max(5.0, 0.14 * base_spacing)
        hint_inliers = close | (hint_inliers & (marker_distance <= max(6.0, 0.30 * base_spacing)))
        cleaned_hints = hints.copy()
        cleaned_hints[~hint_inliers] = np.nan
        predicted, hint_inliers = _smooth_fit(grid, cleaned_hints, base_spacing)
    for _ in range(2):
        local = _local_spacing(predicted, topology)
        assigned = _anchored_assign(
            predicted, hints, hint_inliers, detections,
            np.clip(0.62 * local, 8.0, 160.0), local,
        )
        good = np.asarray([item is not None for item in assigned])
        if np.count_nonzero(good) < 3:
            break
        observed = predicted.copy()
        observed[good] = [(assigned[i].u, assigned[i].v) for i in np.flatnonzero(good)]
        predicted, _ = _smooth_fit(grid, observed, float(np.median(local)))
    local = _local_spacing(predicted, topology)
    assigned = _anchored_assign(
        predicted, hints, hint_inliers, detections,
        np.clip(0.62 * local, 8.0, 160.0), local,
    )
    missing = [index for index, item in enumerate(assigned) if item is None]
    if missing:
        used = {id(item) for item in assigned if item is not None}
        remaining = [item for item in detections if id(item) not in used]
        observed_grid = grid[[index for index, item in enumerate(assigned) if item is not None]]
        observed_uv = np.asarray(
            [(item.u, item.v) for item in assigned if item is not None], float
        )
        local_predictions = []
        by_cell = {(int(x), int(y)): index for index, (x, y) in enumerate(grid)}
        for index in missing:
            col, row = (int(value) for value in grid[index])
            neighbor_estimates = []
            for first, second in (((col - 1, row), (col + 1, row)), ((col, row - 1), (col, row + 1))):
                if first in by_cell and second in by_cell:
                    a, b = assigned[by_cell[first]], assigned[by_cell[second]]
                    if a is not None and b is not None:
                        neighbor_estimates.append(0.5 * np.array([a.u + b.u, a.v + b.v]))
            distance = np.linalg.norm(observed_grid - grid[index], axis=1)
            nearest = np.argsort(distance, kind="stable")[:min(12, len(distance))]
            design = np.column_stack((np.ones(len(nearest)), observed_grid[nearest]))
            weight = 1.0 / np.maximum(0.35, distance[nearest])
            coefficient, *_ = np.linalg.lstsq(
                design * weight[:, None], observed_uv[nearest] * weight[:, None], rcond=None
            )
            affine = np.array([1.0, *grid[index]]) @ coefficient
            local_predictions.append(np.mean(neighbor_estimates, axis=0) if neighbor_estimates else affine)
        recovered = _assign(
            np.asarray(local_predictions), remaining,
            np.clip(0.78 * local[missing], 10.0, 210.0),
        )
        for index, item in zip(missing, recovered):
            assigned[index] = item
    return predicted, local, assigned, hint_inliers


def _stereo_error(left: Detection, right: Detection, calibration: StereoCalibration):
    xyz = triangulate_point((left.u, left.v), (right.u, right.v), calibration)
    errors = [
        np.linalg.norm(project_points(xyz[None], camera)[0] - (item.u, item.v))
        for camera, item in ((calibration.left, left), (calibration.right, right))
    ]
    return float(max(errors)), xyz


def _monotonic_violations(predicted: np.ndarray, topology: NetTopology) -> set[int]:
    grid = _grid_xy(topology)
    violations: set[int] = set()
    for axis in (0, 1):
        segments = []
        owners = []
        for a, b in topology.edge_indices:
            delta_grid = grid[b] - grid[a]
            if abs(delta_grid[axis]) == 1 and delta_grid[1 - axis] == 0:
                sign = np.sign(delta_grid[axis])
                segments.append(sign * (predicted[b] - predicted[a]))
                owners.append((a, b))
        if segments:
            direction = np.median(np.asarray(segments), axis=0)
            for segment, (a, b) in zip(segments, owners):
                if np.dot(segment, direction) <= 0:
                    violations.update((int(a), int(b)))
    return violations


def bootstrap_mesh(
    left_image: np.ndarray | Sequence[np.ndarray],
    right_image: np.ndarray | Sequence[np.ndarray],
    setup_left: np.ndarray,
    setup_right: np.ndarray,
    topology: NetTopology,
    calibration: StereoCalibration,
    *,
    left_roi=None,
    right_roi=None,
    detection_function: Callable[..., list[Detection]] = detect_markers,
    setup_check_frames: int = 7,
    setup_check_min_hits: int = 2,
    cluster_radius_factor: float = 0.28,
) -> BootstrapResult:
    """Establish identities from a deterministic multi-frame setup check."""
    def image_list(value):
        if isinstance(value, np.ndarray):
            return [value]
        return list(value)

    images = [image_list(left_image)[:setup_check_frames], image_list(right_image)[:setup_check_frames]]
    if not images[0] or not images[1] or len(images[0]) != len(images[1]):
        raise ValueError("Setup check requires an equal nonzero number of stereo frames")
    frame_count = len(images[0])
    min_hits = min(max(1, int(setup_check_min_hits)), frame_count)
    hints = [np.asarray(setup_left, float).reshape(-1, 2), np.asarray(setup_right, float).reshape(-1, 2)]
    if any(len(value) != len(topology) for value in hints):
        raise ValueError("Setup point counts must equal topology node count")
    models = [MarkerColorModel().fit(side_images[0], points) for side_images, points in zip(images, hints)]
    detections, candidate_hits = [], []
    for side_images, model, roi, points in zip(images, models, (left_roi, right_roi), hints):
        by_frame = [
            detection_function(image, model, roi=roi, expected_count=len(topology))
            for image in side_images
        ]
        radius = max(3.0, cluster_radius_factor * _spacing(points, topology))
        candidates, hits = cluster_detections(by_frame, radius, min_hits)
        detections.append(candidates)
        candidate_hits.append(hits)
    views = [
        _view_bootstrap(points, topology, found)
        for points, topology, found in zip(hints, (topology, topology), detections)
    ]
    predicted = [view[0] for view in views]
    spacing = [view[1] for view in views]
    assigned = [view[2] for view in views]
    audit = []
    for side, name in enumerate(("left", "right")):
        audit.extend(repair_boundary_shifts(
            assigned[side], predicted[side], spacing[side], topology, name,
        ))
        extrapolate_missing_boundaries(assigned[side], predicted[side], topology)
        assigned[side], new_hits = rescue_across_frames(
            images[side], models[side], predicted[side], spacing[side], assigned[side], min_hits,
        )
        candidate_hits[side].update(new_hits)

    initial_errors = [
        _stereo_error(left, right, calibration)[0]
        for left, right in zip(assigned[0], assigned[1])
        if left is not None and right is not None
    ]
    if initial_errors:
        center = float(np.median(initial_errors))
        mad = 1.4826 * float(np.median(np.abs(np.asarray(initial_errors) - center)))
        stereo_gate = max(3.0, center + 15.0 * max(mad, 0.10))
    else:
        stereo_gate = 3.0

    # Repair a stereo mismatch using an unused, structurally plausible detection.
    for index in range(len(topology)):
        left, right = assigned[0][index], assigned[1][index]
        if left is not None and right is not None and _stereo_error(left, right, calibration)[0] <= stereo_gate:
            continue
        best = None
        for side in (0, 1):
            fixed = assigned[1 - side][index]
            if fixed is None:
                continue
            used = [value for value in assigned[side] if value is not None]
            for candidate in detections[side]:
                if any(
                    id(candidate) == id(value) or np.hypot(candidate.u - value.u, candidate.v - value.v) < 0.12 * spacing[side][index]
                    for value in used
                ):
                    continue
                distance = np.linalg.norm(np.array([candidate.u, candidate.v]) - predicted[side][index])
                if assigned[side][index] is not None and distance > 0.72 * spacing[side][index]:
                    continue
                pair = (candidate, fixed) if side == 0 else (fixed, candidate)
                error, _ = _stereo_error(*pair, calibration)
                score = error + distance / max(spacing[side][index], 1.0)
                if error <= stereo_gate and (best is None or score < best[0]):
                    best = (score, side, candidate)
        if best is not None:
            assigned[best[1]][index] = best[2]

    violations = _monotonic_violations(predicted[0], topology) | _monotonic_violations(predicted[1], topology)
    nodes, positions, active = [], [], []
    xyz_full = np.full((len(topology), 3), np.nan)
    grid_cells = _grid_xy(topology)
    for index, node_id in enumerate(topology.node_ids):
        left, right = assigned[0][index], assigned[1][index]
        reason = None
        xyz = None
        if left is None or right is None:
            reason = "no_detection"
        else:
            error, xyz = _stereo_error(left, right, calibration)
            col, row = grid_cells[index]
            boundary = col in (0, topology.grid_cols - 1) or row in (0, topology.grid_rows - 1)
            node_stereo_gate = max(stereo_gate, 12.0) if boundary else stereo_gate
            if error > node_stereo_gate:
                reason = "stereo_inconsistent"
        distances = {
            side: (None if assigned[k][index] is None else float(np.linalg.norm(
                np.array([assigned[k][index].u, assigned[k][index].v]) - hints[k][index]
            ))) for k, side in enumerate(("left", "right"))
        }
        repaired_views = [side for k, side in enumerate(("left", "right")) if distances[side] is not None and distances[side] > max(6.0, 0.34 * spacing[k][index])]
        status = "absent" if reason else "repaired" if repaired_views else "confirmed"
        nodes.append({
            "obj_id": int(node_id), "row": int(_grid_xy(topology)[index, 1]),
            "col": int(_grid_xy(topology)[index, 0]), "status": status, "reason": reason,
            "repair_distance_px": distances, "repaired_views": repaired_views,
            "predicted_left": [float(value) for value in predicted[0][index]],
            "predicted_right": [float(value) for value in predicted[1][index]],
            "left": None if left is None else [float(left.u), float(left.v)],
            "right": None if right is None else [float(right.u), float(right.v)],
            "monotonic_violation": index in violations,
        })
        if reason is None:
            active.append(index)
            positions.append(xyz)
            xyz_full[index] = xyz
        hits = {
            side: (0 if assigned[k][index] is None else int(candidate_hits[k].get(id(assigned[k][index]), 1)))
            for k, side in enumerate(("left", "right"))
        }
        nodes[-1]["seen_frames"] = hits
        nodes[-1]["setup_check_frames"] = frame_count
    boundary_flags = boundary_edge_suspects(topology, xyz_full)
    audit.extend(boundary_flags)
    active_node_ids = [topology.node_ids[index] for index in active]
    active_ids = set(active_node_ids)
    active_topology = NetTopology(active_node_ids, [edge for edge in topology.edges if edge[0] in active_ids and edge[1] in active_ids])
    active_topology.grid_cols, active_topology.grid_rows = topology.grid_cols, topology.grid_rows
    counts = {name: sum(node["status"] == name for node in nodes) for name in ("confirmed", "repaired", "absent")}
    report = {
        "stage": "setup check",
        "summary": {
            **counts, "setup_check_frames": frame_count, "min_hits": min_hits,
            "detections_left": len(detections[0]), "detections_right": len(detections[1]),
            "monotonic_violations": len(violations), "stereo_gate_px": stereo_gate,
            "boundary_shift_repaired": sum(item["type"] == "boundary_shift_repaired" for item in audit),
            "boundary_suspect": sum(item["type"] == "boundary_suspect" for item in audit),
        },
        "nodes": nodes,
        "audit": audit,
    }
    return BootstrapResult(
        active_topology, np.asarray(positions, float).reshape(-1, 3),
        np.asarray([[assigned[0][i].u, assigned[0][i].v] for i in active], float),
        np.asarray([[assigned[1][i].u, assigned[1][i].v] for i in active], float),
        tuple(assigned[0][i] for i in active), tuple(assigned[1][i] for i in active),
        models[0], models[1], report,
    )
