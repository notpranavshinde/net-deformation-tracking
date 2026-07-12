"""Per-node mesh-deviation metrics and suspect-interval extraction."""

from __future__ import annotations

import numpy as np

from .topology import NetTopology


def node_edge_deviation(
    edge_indices: np.ndarray, edge_lengths: np.ndarray, rest_lengths: np.ndarray, node_count: int,
) -> np.ndarray:
    """Return each node's mean incident-edge relative deviation from rest."""
    values = np.zeros(node_count, dtype=float)
    counts = np.zeros(node_count, dtype=np.int32)
    if not len(edge_indices):
        return values
    relative = np.abs(edge_lengths - rest_lengths) / np.maximum(rest_lengths, 1e-9)
    for edge_index, (first, second) in enumerate(edge_indices):
        for node in (int(first), int(second)):
            values[node] += relative[edge_index]
            counts[node] += 1
    return np.divide(values, counts, out=np.zeros_like(values), where=counts > 0)


def suspect_node_reports(frames, topology: NetTopology, threshold: float, min_len: int) -> tuple[list[dict], int]:
    """Collapse consecutive above-threshold samples into maximal node intervals."""
    reports, total = [], 0
    cols = topology.grid_cols or len(topology)
    for index, obj_id in enumerate(topology.node_ids):
        values = np.asarray([frame.metrics["node_edge_relative_deviation"][index] for frame in frames], float)
        edge_suspect = values > threshold
        one_view_only = np.asarray([
            frame.statuses[index] in ("measured-left", "measured-right") for frame in frames
        ], dtype=bool)
        def collapse(flags, include_deviation=False):
            intervals = []
            start = None
            for frame_index, flagged in enumerate(np.r_[flags, False]):
                if flagged and start is None:
                    start = frame_index
                elif not flagged and start is not None:
                    end = frame_index - 1
                    if end - start + 1 >= min_len:
                        first, last = frames[start], frames[end]
                        interval = {
                            "start_frame": int(first.left_frame), "end_frame": int(last.left_frame),
                            "left_start_frame": int(first.left_frame), "left_end_frame": int(last.left_frame),
                            "right_start_frame": int(first.right_frame), "right_end_frame": int(last.right_frame),
                            "length": int(end - start + 1),
                        }
                        if include_deviation:
                            interval["max_mean_relative_deviation"] = float(np.max(values[start:end + 1]))
                            interval["reasons"] = ["edge_deviation"]
                        intervals.append(interval)
                    start = None
            return intervals

        intervals = collapse(edge_suspect, include_deviation=True)
        one_view_intervals = collapse(one_view_only)
        total += len(intervals)
        row, col = divmod(int(obj_id), cols)
        reports.append({
            "obj_id": int(obj_id), "row": int(row), "col": int(col),
            "suspect_intervals": intervals,
            "one_view_intervals": one_view_intervals,
        })
    return reports, total
