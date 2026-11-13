#!/usr/bin/env python3
"""Plain-assert checks for labelling a marker grid by growing it from the corners."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nettrack.grid_grow import flat_grid, grow_grid


def bulged_net(cols=12, rows=12):
    """A net like the real views: spacing shrinks down the frame (perspective), the middle bulges
    sideways by ~1.5 knots, and the top edge is wavy."""
    pts = []
    y = 300.0
    for r in range(rows):
        step = 390.0 - 21.0 * r  # 390 px at the top, ~160 px at the bottom
        for c in range(cols):
            bulge = 1.5 * 260.0 * np.sin(np.pi * r / (rows - 1)) * np.sin(np.pi * c / (cols - 1))
            wave = 60.0 * (c % 2) if r == 0 else 0.0
            pts.append((1500.0 + c * 270.0 + bulge, y + wave))
        y += step
    return np.array(pts)


def test_bulged_net_labels():
    truth = bulged_net()
    corners = truth[[0, 11, 132, 143]]
    assert np.max(np.linalg.norm(flat_grid(corners, 12, 12) - truth, axis=1)) > 260.0  # stretching fails here
    points, gap = grow_grid(truth.copy(), corners, 12, 12)
    assert not gap.any()
    assert np.allclose(points, truth)


def test_missing_and_duplicate_markers():
    truth = bulged_net()
    corners = truth[[0, 11, 132, 143]]
    missing = [5, 66]  # one on the top edge, one inside
    det = np.delete(truth, missing, axis=0)
    det = np.vstack([det, truth[3] + (20.0, 0.0)])  # a marker detected twice
    points, gap = grow_grid(det, corners, 12, 12)
    assert set(np.flatnonzero(gap)) == set(missing), np.flatnonzero(gap)
    ok = ~gap
    assert np.allclose(points[ok], truth[ok])  # no knot labelled as its neighbour
    assert np.all(np.linalg.norm(points[missing] - truth[missing], axis=1) < 0.45 * 270.0)


def test_reflected_top_row_and_missed_edge_marker():
    """As in a real right camera view: the top row reflected in the water surface just above the net,
    the automatic top corners on that reflection, and a marker on the right edge not detected."""
    truth = bulged_net()
    top = truth[:12]
    reflection = top + (top - truth[12:24]) * 0.55  # mirrored above the top row, compressed
    det = np.vstack([np.delete(truth, [119], axis=0), reflection])
    corners = np.array([reflection[0], reflection[11], truth[132], truth[143]])
    info = {}
    points, gap = grow_grid(det, corners, 12, 12, info)
    assert np.flatnonzero(gap).tolist() == [119], np.flatnonzero(gap)
    ok = ~gap
    assert np.allclose(points[ok], truth[ok])
    assert np.allclose(info["corners"][:2], truth[[0, 11]])
    assert any("reflection" in note for note in info["notes"]), info["notes"]


def main():
    test_bulged_net_labels()
    test_missing_and_duplicate_markers()
    test_reflected_top_row_and_missed_edge_marker()
    print("nettrack grid-grow tests passed")


if __name__ == "__main__":
    main()
