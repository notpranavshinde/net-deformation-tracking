"""Where grid growing and the saved setup disagree, which one is right?

Objective test: triangulate each knot from its left/right labels with the calibration. A correct
labelling reprojects within a few px, and its 3D edges to grid neighbours have the usual length
(the netting does not stretch); a knot labelled as its neighbour fails both.
Also writes a contact sheet of the disagreeing knots for checking by eye.

    python experiments/setup_auto/adjudicate.py <run index in eval_setup.runs()>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_setup as e  # noqa: E402
from nettrack.geometry import load_stereo_calibration, project_points, triangulate_point  # noqa: E402

G = e.GRID


def labelling_3d(left: np.ndarray, right: np.ndarray, calib):
    xyz = np.array([triangulate_point(tuple(l), tuple(r), calib) for l, r in zip(left, right)])
    err = np.maximum(np.linalg.norm(project_points(xyz, calib.left) - left, axis=1),
                     np.linalg.norm(project_points(xyz, calib.right) - right, axis=1))
    g = xyz.reshape(G, G, 3)
    edge = np.full((G, G), np.nan)  # worst relative deviation of a knot's edges from the median edge
    lengths = np.concatenate([np.linalg.norm(np.diff(g, axis=0), axis=2).ravel(), np.linalg.norm(np.diff(g, axis=1), axis=2).ravel()])
    med = np.median(lengths)
    for r in range(G):
        for c in range(G):
            nb = [g[r + dr, c + dc] for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)) if 0 <= r + dr < G and 0 <= c + dc < G]
            edge[r, c] = max(abs(np.linalg.norm(g[r, c] - q) - med) / med for q in nb)
    return err, edge.ravel(), med


def main() -> int:
    job = e.runs()[int(sys.argv[1])]
    run = job["run"]
    ref_l, ref_r, start = e.final_points(run)
    auto = e.semi_auto(run)
    calib = load_stereo_calibration(job["stereo"])
    frames, grown = {}, {}
    for k, s in enumerate(("left", "right")):
        frames[s] = e.read_frames(e.VIDEOS / job["videos"][k], start, 1)[0]
        grown[s] = e.grow(frames[s], auto[s + "_corners"])[0]
    ref = {"left": ref_l, "right": ref_r}
    res = {name: labelling_3d(p["left"], p["right"], calib) for name, p in (("saved setup", ref), ("grid growing", grown))}
    differ = np.zeros(G * G, bool)
    for s in ref:
        differ |= np.linalg.norm(grown[s] - ref[s], axis=1) > e.WRONG * e.local_spacing(ref[s])
    print(f"{run.name}: {differ.sum()} knots labelled differently")
    for name, (err, edge, med) in res.items():
        print(f"  {name:13s} median edge {med * 1000:.1f} mm | on those knots: reprojection median {np.median(err[differ]):.1f} px, "
              f"worst-edge deviation median {np.median(edge[differ]) * 100:.0f}% | all knots: reprojection > 10 px: {(err > 10).sum()}, "
              f"edge off > 15%: {(edge > 0.15).sum()}")

    # Contact sheet, left camera: saved setup in magenta, grid growing in cyan, 360 px crops.
    tiles = []
    for i in np.flatnonzero(differ)[:40]:
        x, y = ref_l[i] if np.linalg.norm(grown["left"][i] - ref_l[i]) < 400 else grown["left"][i]
        x0, y0 = int(max(0, x - 180)), int(max(0, y - 180))
        crop = frames["left"][y0:y0 + 360, x0:x0 + 360].copy()
        for p, col in ((ref_l[i], (255, 0, 255)), (grown["left"][i], (255, 255, 0))):
            c = (int(p[0] - x0), int(p[1] - y0))
            cv2.circle(crop, c, 22, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.circle(crop, c, 22, col, 3, cv2.LINE_AA)
        crop = cv2.copyMakeBorder(crop, 0, 360 - crop.shape[0], 0, 360 - crop.shape[1], cv2.BORDER_CONSTANT)
        cv2.putText(crop, f"knot {i} (r{i // G} c{i % G})", (8, 26), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(crop)
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)])
        out = e.OUT / f"disagreements_{run.name}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  sheet: {out}  (magenta = saved setup, cyan = grid growing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
