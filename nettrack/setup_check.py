"""Multi-frame setup-check candidate and lattice-consistency helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .detect import Detection, detect_in_window
from .topology import NetTopology


@dataclass
class _Cluster:
    detections: list[Detection] = field(default_factory=list)
    frames: set[int] = field(default_factory=set)

    @property
    def center(self) -> np.ndarray:
        return np.median([(item.u, item.v) for item in self.detections], axis=0)


def cluster_detections(
    detections_by_frame: Sequence[Sequence[Detection]], radius: float, min_hits: int,
) -> tuple[list[Detection], dict[int, int]]:
    """Cluster detections across frames and return robust candidates and hit counts."""
    clusters: list[_Cluster] = []
    for frame_index, detections in enumerate(detections_by_frame):
        for detection in sorted(detections, key=lambda item: (item.v, item.u)):
            eligible = [
                (float(np.linalg.norm(cluster.center - (detection.u, detection.v))), index)
                for index, cluster in enumerate(clusters) if frame_index not in cluster.frames
            ]
            distance, match = min(eligible, default=(float("inf"), -1))
            if distance > radius:
                clusters.append(_Cluster([detection], {frame_index}))
            else:
                clusters[match].detections.append(detection)
                clusters[match].frames.add(frame_index)
    candidates, hits = [], {}
    for cluster in clusters:
        if len(cluster.frames) < min_hits:
            continue
        center = cluster.center
        candidate = Detection(
            float(center[0]), float(center[1]),
            int(round(np.median([item.area_px for item in cluster.detections]))),
            float(np.median([item.solidity for item in cluster.detections])),
            float(np.median([item.quality for item in cluster.detections])),
        )
        candidates.append(candidate)
        hits[id(candidate)] = len(cluster.frames)
    candidates.sort(key=lambda item: (item.v, item.u))
    return candidates, hits


def rescue_across_frames(
    images: Sequence[np.ndarray], model, predicted: np.ndarray, local_spacing: np.ndarray,
    assigned: list[Detection | None], min_hits: int,
) -> tuple[list[Detection | None], dict[int, int]]:
    """Use relaxed local searches across setup-check frames for missing nodes."""
    hit_counts: dict[int, int] = {}
    for index, current in enumerate(assigned):
        if current is not None:
            continue
        found = []
        radius = float(np.clip(0.55 * local_spacing[index], 8, 120))
        for image in images:
            item = detect_in_window(image, model, predicted[index], radius, True)
            if item is not None and np.linalg.norm(np.array([item.u, item.v]) - predicted[index]) <= 0.62 * local_spacing[index]:
                found.append(item)
        # Each local search concerns the same node; keep the dominant local cluster.
        if found:
            center = np.median([(item.u, item.v) for item in found], axis=0)
            close = [item for item in found if np.linalg.norm(np.array([item.u, item.v]) - center) <= radius * 0.35]
        else:
            close = []
        if len(close) < min_hits:
            continue
        item = Detection(
            float(np.median([value.u for value in close])), float(np.median([value.v for value in close])),
            int(round(np.median([value.area_px for value in close]))),
            float(np.median([value.solidity for value in close])),
            float(np.median([value.quality for value in close])),
        )
        separation = max(3.0, 0.12 * local_spacing[index])
        if not any(np.hypot(item.u - old.u, item.v - old.v) < separation for old in assigned if old is not None):
            assigned[index] = item
            hit_counts[id(item)] = len(close)
    return assigned, hit_counts


def _grid_lines(topology: NetTopology) -> list[list[int]]:
    cols = topology.grid_cols or len(topology)
    cells = {divmod(int(node_id), cols): index for index, node_id in enumerate(topology.node_ids)}
    rows = [[index for (row, _col), index in sorted(cells.items()) if row == wanted]
            for wanted in sorted({row for row, _col in cells})]
    columns = [[index for (_row, col), index in sorted(cells.items()) if col == wanted]
               for wanted in sorted({col for _row, col in cells})]
    return [line for line in rows + columns if len(line) >= 2]


def _node_cost(index: int, item: Detection, predicted: np.ndarray, spacing: np.ndarray) -> float:
    return float(np.linalg.norm(np.array([item.u, item.v]) - predicted[index]) / max(spacing[index], 1.0)) ** 2


def repair_boundary_shifts(
    assigned: list[Detection | None], predicted: np.ndarray, spacing: np.ndarray,
    topology: NetTopology, view: str,
) -> list[dict]:
    """Repair a one-cell boundary label shift when the alternative is clearly better."""
    events = []
    for original in _grid_lines(topology):
        for line in (original, list(reversed(original))):
            boundary, inner = line[:2]
            source, target = None, None
            if assigned[boundary] is not None and assigned[inner] is None:
                source, target = boundary, inner
            elif assigned[boundary] is None and assigned[inner] is not None:
                source, target = inner, boundary
            if source is None:
                continue
            item = assigned[source]
            old_cost = _node_cost(source, item, predicted, spacing)
            new_cost = _node_cost(target, item, predicted, spacing)
            if len(line) >= 3 and assigned[line[2]] is not None:
                neighbor = assigned[line[2]]
                observed_step = np.linalg.norm(np.array([item.u - neighbor.u, item.v - neighbor.v]))
                base_step = float(np.median(spacing[line[:3]]))
                if source == boundary:
                    old_expected, new_expected = 2.0 * base_step, base_step
                else:
                    old_expected, new_expected = base_step, 2.0 * base_step
                old_cost += 2.0 * ((observed_step - old_expected) / max(spacing[source], 1.0)) ** 2
                new_cost += 2.0 * ((observed_step - new_expected) / max(spacing[target], 1.0)) ** 2
            distance = np.linalg.norm(np.array([item.u, item.v]) - predicted[target])
            if new_cost + 0.12 >= old_cost or distance > 0.75 * spacing[target]:
                continue
            assigned[target], assigned[source] = item, None
            if source == boundary and len(line) >= 3 and assigned[line[2]] is not None:
                neighbor = assigned[line[2]]
                predicted[source] = 2.0 * np.array([item.u, item.v]) - np.array([neighbor.u, neighbor.v])
            events.append({
                "type": "boundary_shift_repaired", "view": view,
                "affected_ids": [int(topology.node_ids[source]), int(topology.node_ids[target])],
                "from_obj_id": int(topology.node_ids[source]), "to_obj_id": int(topology.node_ids[target]),
                "cost_before": old_cost, "cost_after": new_cost,
            })
    return events


def extrapolate_missing_boundaries(
    assigned: Sequence[Detection | None], predicted: np.ndarray, topology: NetTopology,
) -> None:
    """Place an unassigned boundary prediction one measured lattice step outward."""
    for original in _grid_lines(topology):
        for line in (original, list(reversed(original))):
            boundary, inner = line[:2]
            if len(line) < 3 or assigned[boundary] is not None:
                continue
            first, second = assigned[inner], assigned[line[2]]
            if first is not None and second is not None:
                predicted[boundary] = 2.0 * np.array([first.u, first.v]) - np.array([second.u, second.v])


def boundary_edge_suspects(topology: NetTopology, xyz_full: np.ndarray) -> list[dict]:
    """Flag implausible triangulated edges touching the outer grid boundary."""
    if not topology.grid_cols or not topology.grid_rows:
        return []
    cols, rows = topology.grid_cols, topology.grid_rows
    records = []
    lengths = []
    for a, b in topology.edges:
        ia, ib = topology.index(a), topology.index(b)
        if np.all(np.isfinite(xyz_full[[ia, ib]])):
            lengths.append((a, b, float(np.linalg.norm(xyz_full[ia] - xyz_full[ib]))))
    median = float(np.median([value for _a, _b, value in lengths])) if lengths else float("nan")
    if not np.isfinite(median) or median <= 0:
        return records
    for a, b, length in lengths:
        cells = [divmod(int(value), cols) for value in (a, b)]
        touches = any(row in (0, rows - 1) or col in (0, cols - 1) for row, col in cells)
        ratio = length / median
        if touches and (ratio < 0.5 or ratio > 1.5):
            records.append({"type": "boundary_suspect", "affected_ids": [int(a), int(b)], "edge_length_ratio": ratio})
    return records
