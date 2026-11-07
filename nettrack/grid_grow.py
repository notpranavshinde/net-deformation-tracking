"""Label a full marker grid in one camera frame by growing it from the corners.

Stretching a flat grid between 4 corners fails when the net bulges by more than about half a knot
spacing: guesses snap onto the neighbouring knot. Growing labels one knot at a time from labelled
neighbours, so the net only has to be smooth locally:
  edges     walk marker to marker toward the far corner (edges are wavy, and perspective shrinks
            the spacing, so a predicted step length would overshoot or fall short)
  interior  predict (r, c) from (r-1, c), (r, c-1), (r-1, c-1) and take the nearest marker
A knot with no marker near its prediction keeps the prediction and is flagged as a gap; a step
twice as long as the last one means a marker was missed, and also leaves a gap.

Corners are ordered top-left, top-right, bottom-left, bottom-right. Only the top-left must be on
its marker; the other three only give directions.
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


def grow_grid(detections: np.ndarray, corners: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major points (cols * rows, 2) and a gap flag per knot (True: no marker found there)."""
    det = np.asarray(detections, float).reshape(-1, 2)
    if len(det) < 4:
        raise ValueError(f"Only {len(det)} markers detected")
    used = np.zeros(len(det), bool)
    nn = np.sort(np.linalg.norm(det[:, None] - det[None], axis=2), axis=1)[:, 1]
    step0 = float(np.median(nn))
    P = np.full((rows, cols, 2), np.nan)
    gap = np.zeros((rows, cols), bool)

    def take(pred, step):
        d = np.linalg.norm(det - pred, axis=1)
        d[used] = np.inf
        i = int(np.argmin(d))
        if d[i] <= 0.45 * step:
            used[np.linalg.norm(det - det[i], axis=1) < 0.3 * step0] = True  # also its duplicates
            return det[i], False
        return np.asarray(pred, float), True

    def walk(prev, target):
        """Nearest unused marker ahead of prev, within 35 degrees of the direction to target."""
        u = (target - prev) / np.linalg.norm(target - prev)
        v = det - prev
        dist = np.linalg.norm(v, axis=1)
        ok = ~used & (dist > 0.5 * step0) & (v @ u >= np.cos(np.radians(35)) * dist)
        return det[np.flatnonzero(ok)[np.argmin(dist[ok])]] if ok.any() else None

    tl, tr, bl, _ = np.asarray(corners, float)
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                pred, step = tl, step0
            elif r == 0 or c == 0:
                prev, target, left = (P[0, c - 1], tr, cols - c) if r == 0 else (P[r - 1, 0], bl, rows - r)
                nxt = walk(prev, target)
                pred = nxt if nxt is not None else prev + (target - prev) / left
                before = (P[0, c - 1] - P[0, c - 2]) if r == 0 and c >= 2 else (P[r - 1, 0] - P[r - 2, 0]) if c == 0 and r >= 2 else None
                if before is not None and np.linalg.norm(pred - prev) > 1.6 * np.linalg.norm(before):
                    pred = prev + (pred - prev) / 2  # skipped a missing marker: leave a gap here
                step = np.linalg.norm(pred - prev)
            else:
                pred = P[r - 1, c] + P[r, c - 1] - P[r - 1, c - 1]
                step = min(np.linalg.norm(P[r, c - 1] - P[r - 1, c - 1]), np.linalg.norm(P[r - 1, c] - P[r - 1, c - 1]))
            P[r, c], gap[r, c] = take(pred, step)
    return P.reshape(-1, 2), gap.reshape(-1)
