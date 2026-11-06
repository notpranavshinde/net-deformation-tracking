"""Render the tracking video for the first wave test (A = 0.05 m, F = 0.25, trial 2) for the deck.

Left camera, CoTracker3 + snap from the queue run, cropped to the grid and encoded at 15 fps
(half speed), same look as cotracker_tracking.mp4.
Output: work/cotracker3/deck/media/wave_tracking.mp4, .png and _cover.png

    python experiments/cotracker3/make_wave_media.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_deck_media as dm  # noqa: E402

ROOT = dm.ROOT
QUEUE = ROOT / "work" / "pipeline_queue" / "20261106_052812_545745"
RUN = QUEUE / "run_001_A_0.1_F_0.25_trial2"  # queue label predates the rename to A = 0.05 m
NAME = "wave_tracking"


def main() -> int:
    job = json.loads((QUEUE / "queue_manifest.json").read_text())["jobs"][0]
    start, end = int(job["start_frame"]), int(job["end_frame"])
    tracks = pd.read_csv(RUN / "cotracker3" / "left" / "tracks_2d.csv")
    n = end - start
    pts = np.full((n, dm.N, 2), np.nan)
    pts[tracks["frame"], tracks["obj_id"]] = tracks[["u", "v"]].to_numpy()
    hidden = np.zeros((n, dm.N), bool)
    hidden[tracks["frame"], tracks["obj_id"]] = tracks["visible"].to_numpy() == 0

    pad = 120
    x0, y0 = int(np.nanmin(pts[..., 0])) - pad, max(0, int(np.nanmin(pts[..., 1])) - pad)
    x1, y1 = int(np.nanmax(pts[..., 0])) + pad, int(np.nanmax(pts[..., 1])) + pad
    width = 960
    s = width / (x1 - x0)

    cap = cv2.VideoCapture(job["left_video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for t in range(n):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"video ended at frame {start + t}")
        img = cv2.resize(frame[y0:y1, x0:x1], (width, int((y1 - y0) * s)), interpolation=cv2.INTER_AREA)
        p = pts[t] - [x0, y0]
        dm.draw_grid(img, p, s, (70, 120, 60), 1)
        for i, q in enumerate(p):
            cv2.circle(img, tuple((q * s).astype(int)), 4, dm.INFER if hidden[t, i] else dm.COT, -1, cv2.LINE_AA)
        dm.stamp(img, f"A = 0.05 m wave  |  {t / 29.97:4.1f} s  |  half speed")
        out.append(dm.header(img, "CoTracker3: wave test, left camera", dm.COT))
    cap.release()
    dm.encode(out, NAME, 150)
    cv2.imwrite(str(dm.OUT / f"{NAME}_cover.png"), out[150])
    h, w = out[0].shape[:2]
    (dm.OUT / f"{NAME}.json").write_text(json.dumps({"w": w, "h": h, "crop": [x0, y0, x1, y1]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
