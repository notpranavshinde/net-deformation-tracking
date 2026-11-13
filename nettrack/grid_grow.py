"""Label a full marker grid in one camera frame by growing it from the corners.

Stretching a flat grid between 4 corners fails when the net bulges by more than about half a knot
spacing: guesses snap onto the neighbouring knot. Growing labels one knot at a time from labelled
neighbours, so the net only has to be smooth locally:
  edges     walk marker to marker from one reviewed corner to the next (edges are wavy, and
            perspective shrinks the spacing, so a predicted step length would overshoot or fall
            short), then reconcile the count with the grid size:
              one marker short   the longest step for its neighbourhood skipped a missing
                                 marker: a gap goes there
              one marker extra   a first (or last) step much shorter than the next one means the
                                 corner sits on a row that is not part of the net, e.g. the top
                                 row reflected in the water surface: that corner moves one in
  interior  predict (r, c) from (r-1, c), (r, c-1), (r-1, c-1) and take the nearest marker
A knot with no marker near its prediction keeps the prediction and is flagged as a gap.

Corners are ordered top-left, top-right, bottom-left, bottom-right, as reviewed by the user: the
left, right and top edges are walked between them.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colormodel import MarkerColorModel
from .detect import detect_markers


def flat_grid(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Bilinear grid between the corners, row-major."""
    tl, tr, bl, br = np.asarray(corners, float)
    out = []
    for r in range(rows):
        b = r / (rows - 1)
        for c in range(cols):
            a = c / (cols - 1)
            out.append((1 - a) * (1 - b) * tl + a * (1 - b) * tr + (1 - a) * b * bl + a * b * br)
    return np.array(out)


def orange_seeds(frame_bgr: np.ndarray, rough_points: np.ndarray, spacing: float) -> np.ndarray:
    """Pixels on markers without clicks: the most orange spot near each rough point, kept when it is
    about as orange as the typical spot. Used to learn the marker colour."""
    s = 0.25
    small = cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32), (5, 5), 0)
    r = max(2, int(0.4 * spacing * s))
    found = []
    for x, y in np.asarray(rough_points, float):
        cx, cy = int(x * s), int(y * s)
        x0, y0 = max(0, cx - r), max(0, cy - r)
        win = a[y0:cy + r + 1, x0:cx + r + 1]
        if win.size:
            iy, ix = np.unravel_index(np.argmax(win), win.shape)
            found.append(((x0 + ix + 0.5) / s, (y0 + iy + 0.5) / s, win[iy, ix]))
    found = np.array(found)
    keep = found[:, 2] >= np.median(found[:, 2]) - 6
    return found[keep, :2]


def detect(frame_bgr: np.ndarray, corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """All marker centres in the frame (sub-pixel), with the colour learned from the frame itself."""
    rough = flat_grid(corners, cols, rows)
    spacing = float(np.median(np.linalg.norm(np.diff(rough.reshape(rows, cols, 2), axis=1), axis=2)))
    model = MarkerColorModel().fit(frame_bgr, orange_seeds(frame_bgr, rough, spacing))
    return np.array([(d.u, d.v) for d in detect_markers(frame_bgr, model, expected_count=cols * rows)], float).reshape(-1, 2)


def grow_grid(detections: np.ndarray, corners: np.ndarray, cols: int, rows: int,
              info: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Row-major points (cols * rows, 2) and a gap flag per knot (True: no marker found there).

    ``info``, when given, receives ``corners`` (the corners used, after any move) and ``notes``
    (one line per corner moved, edge cut short, or marker assumed missing on an edge).
    """
    det = np.asarray(detections, float).reshape(-1, 2)
    if len(det) < 4:
        raise ValueError(f"Only {len(det)} markers detected")
    used = np.zeros(len(det), bool)
    excluded = np.zeros(len(det), bool)  # markers on a row or column found not to be part of the net
    nn = np.sort(np.linalg.norm(det[:, None] - det[None], axis=2), axis=1)[:, 1]
    step0 = float(np.median(nn))
    notes: list[str] = []

    def mark_used(p):
        used[np.linalg.norm(det - p, axis=1) < 0.3 * step0] = True  # the marker and its duplicates

    def take(pred, step):
        d = np.linalg.norm(det - pred, axis=1)
        d[used | excluded] = np.inf
        i = int(np.argmin(d))
        if d[i] <= 0.45 * step:
            mark_used(det[i])
            return det[i], False
        return np.asarray(pred, float), True

    def walk_edge(start, end, names):
        """Markers from the start corner to the end corner, each the nearest unused one ahead within
        35 degrees of the direction to the end, with a gap wherever a step is far longer than the
        steps around it (a marker was not detected there)."""
        taken = excluded.copy()

        def claim(i):
            taken[np.linalg.norm(det - det[i], axis=1) < 0.3 * step0] = True

        def nearest(p):
            d = np.linalg.norm(det - p, axis=1)
            i = int(np.argmin(d))
            return i if d[i] <= 0.45 * nn[i] else None

        i0, i_end = nearest(start), nearest(end)
        pts, gaps = [det[i0] if i0 is not None else np.asarray(start, float)], [i0 is None]
        if i0 is not None:
            claim(i0)
        while True:
            cur = pts[-1]
            remaining = np.linalg.norm(end - cur)
            u = (end - cur) / max(remaining, 1e-9)
            v = det - cur
            dist = np.linalg.norm(v, axis=1)
            ok = ~taken & (dist > 0.5 * step0) & (v @ u >= np.cos(np.radians(35)) * dist)
            if not ok.any():
                break
            j = int(np.flatnonzero(ok)[np.argmin(dist[ok])])
            if i_end is None and v[j] @ u > remaining + 0.45 * nn[j]:  # past an undetected end corner
                break
            pts.append(det[j])
            gaps.append(False)
            claim(j)
            if j == i_end:
                break
        if i_end is None:
            pts.append(np.asarray(end, float))  # the reviewed corner has no detected marker
            gaps.append(True)
        k = 0
        while k < len(pts) - 1:
            steps = np.array([np.linalg.norm(q - p) for p, q in zip(pts, pts[1:])])
            local = np.median(np.r_[steps[max(0, k - 2):k], steps[k + 1:k + 3]]) if len(steps) > 1 else steps[k]
            if steps[k] > 1.6 * local:
                pts.insert(k + 1, (pts[k] + pts[k + 1]) / 2)
                gaps.insert(k + 1, True)
                notes.append(f"{names[0]} to {names[1]} edge: no marker detected between knots {k} and {k + 1}")
                continue
            k += 1
        return pts, gaps

    def end_ratios(pts):
        """First and last step relative to the steps next to them; well below 1 means that end
        corner sits on an extra row or column (e.g. a reflection in the water surface)."""
        steps = np.array([np.linalg.norm(q - p) for p, q in zip(pts, pts[1:])])
        return steps[0] / np.median(steps[1:4]), steps[-1] / np.median(steps[-4:-1])

    def reconcile(edges, n):
        """Make every edge n points long: split the longest steps when short, drop one end when one long.
        Returns the dropped point of each edge that lost one."""
        dropped = {}
        for name, (pts, gaps) in edges.items():
            while len(pts) < n:
                steps = np.array([np.linalg.norm(q - p) for p, q in zip(pts, pts[1:])])
                k = int(np.argmax(steps / np.median(steps)))
                pts.insert(k + 1, (pts[k] + pts[k + 1]) / 2)
                gaps.insert(k + 1, True)
                notes.append(f"{name} edge: one knot short, assumed a missing marker between knots {k} and {k + 1}")
        long = {name: e for name, e in edges.items() if len(e[0]) == n + 1}
        if long:
            # Every edge with an extra point drops the same end: the one with the strongest evidence.
            evidence = [(r, end) for pts, _ in long.values() for r, end in zip(end_ratios(pts), ("start", "end"))]
            ratio, end = min(evidence)
            if ratio > 0.8:
                long = {}  # no end looks extra: keep the start, as for any other overshoot below
            for name, (pts, gaps) in long.items():
                k = 0 if end == "start" else -1
                notes.append(f"{name} edge: one knot too many; dropped the {'first' if k == 0 else 'last'} point "
                             f"({ratio:.2f} x the next step, e.g. a reflection in the water surface), "
                             f"so that corner moved one knot in")
                dropped[name] = pts[k]
                del pts[k], gaps[k]
        for name, (pts, gaps) in edges.items():
            if len(pts) > n:  # walked past the end corner (e.g. a click off its marker): trust the start
                notes.append(f"{name} edge: {len(pts)} markers between its corners, kept the first {n} from the start corner; "
                             "check the end corner")
                del pts[n:], gaps[n:]
        return dropped

    tl, tr, bl, br = np.asarray(corners, float)
    vertical = {"left": walk_edge(tl, bl, ("top-left", "bottom-left")),
                "right": walk_edge(tr, br, ("top-right", "bottom-right"))}
    extra = reconcile(vertical, rows)
    if len(extra) == 2:  # both vertical edges started (or ended) on the same extra row: drop all of it
        extra_row, _ = walk_edge(extra["left"], extra["right"], ("extra row", "extra row"))
        notes[:] = [nt for nt in notes if not nt.startswith("extra row")]
        for q in extra_row:
            excluded[np.linalg.norm(det - q, axis=1) < 0.3 * step0] = True
    else:
        for q in extra.values():
            excluded[np.linalg.norm(det - q, axis=1) < 0.3 * step0] = True
    left, left_gap = vertical["left"]
    right = vertical["right"][0]
    horizontal = {"top": walk_edge(left[0], right[0], ("top-left", "top-right"))}
    reconcile(horizontal, cols)
    top, top_gap = horizontal["top"]
    if info is not None:
        info["corners"] = np.array([left[0], right[0], left[-1], right[-1]])
        info["notes"] = notes

    P = np.full((rows, cols, 2), np.nan)
    gap = np.zeros((rows, cols), bool)
    P[:, 0], gap[:, 0] = left, left_gap
    P[0, :], gap[0, :] = top, top_gap
    for p in [*left, *top]:
        mark_used(p)
    for r in range(1, rows):
        for c in range(1, cols):
            pred = P[r - 1, c] + P[r, c - 1] - P[r - 1, c - 1]
            step = min(np.linalg.norm(P[r, c - 1] - P[r - 1, c - 1]), np.linalg.norm(P[r - 1, c] - P[r - 1, c - 1]))
            P[r, c], gap[r, c] = take(pred, step)
    return P.reshape(-1, 2), gap.reshape(-1)
