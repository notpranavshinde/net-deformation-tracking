#!/usr/bin/env python3
"""Manual trial15 frame-1771 detector check; skips when local assets are absent."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from nettrack.colormodel import MarkerColorModel
from nettrack.detect import detect_markers

FRAME_INDEX = 1771
EXPECTED_COUNT = 144
SETUP_PATH = REPO / "work/trial15_baseline/run_001_v_0.05mps_a/setup/prompts/points_left_right.json"
VIDEO_DIR = REPO / "raw videos"
OUTPUT_DIR = REPO / "work/nettrack_eval"


def main() -> int:
    videos = {side: VIDEO_DIR / f"trial15{side}.MP4" for side in ("left", "right")}
    missing = [path for path in (SETUP_PATH, *videos.values()) if not path.exists()]
    if missing:
        print("SKIP: local trial15 assets are absent:")
        for path in missing:
            print(f"  {path}")
        return 0

    setup = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for side, video_path in videos.items():
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            print(f"SKIP: could not read frame {FRAME_INDEX} from {video_path}")
            continue

        points = np.asarray([[item["x"], item["y"]] for item in setup[side]], dtype=np.float64)
        model = MarkerColorModel().fit(frame, points)
        started = time.perf_counter()
        detections = detect_markers(frame, model, expected_count=EXPECTED_COUNT)
        elapsed = time.perf_counter() - started
        observed = np.asarray([[item.u, item.v] for item in detections], dtype=np.float64)
        if len(observed):
            nearest = np.min(np.linalg.norm(points[:, None, :] - observed[None, :, :], axis=2), axis=1)
        else:
            nearest = np.full(len(points), np.inf)
        missed = np.flatnonzero(nearest > 60.0).tolist()
        print(
            f"{side}: detections={len(detections)}, time={elapsed:.3f}s, "
            f"setup_indices_without_detection_within_60px={missed}"
        )

        overlay = frame.copy()
        for detection in detections:
            center = (int(round(detection.u)), int(round(detection.v)))
            cv2.circle(overlay, center, 14, (40, 255, 40), 3, cv2.LINE_AA)
        output_path = OUTPUT_DIR / f"detection_check_{side}.png"
        if not cv2.imwrite(str(output_path), overlay):
            raise RuntimeError(f"Could not write {output_path}")
        print(f"{side}: overlay={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
