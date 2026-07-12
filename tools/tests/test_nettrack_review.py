#!/usr/bin/env python3
"""Plain-assert checks for the self-contained marker review artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nettrack.run_tracker import main as tracker_main
from nettrack.review import generate_review
from nettrack.synth import SyntheticNetScene


def test_synthetic_review():
    fps = 20.0
    scene = SyntheticNetScene(
        3, 3, image_size=(480, 360), seed=113,
        dropout_probability=1e-12,
        hidden_intervals={4: (4 / fps, 8 / fps)},
    )
    root = Path(__file__).with_name("_tmp") / "nettrack_review_fixed"
    root.mkdir(parents=True, exist_ok=True)
    left_video, right_video, rig_json = scene.write_videos(
        root, 12, fps, video_suffix=".avi", codec="MJPG"
    )
    points = []
    for camera in scene.cameras:
        points.append([[item["u"], item["v"]] for item in scene.ground_truth(camera, 0.0)])
    setup = root / "setup.json"
    setup.write_text(json.dumps({
        "left": points[0], "right": points[1],
        "crops": {"left": {"crop_applied": False}, "right": {"crop_applied": False}},
    }), encoding="utf-8")
    out = root / "out"
    assert tracker_main([
        "--left-input", str(left_video), "--right-input", str(right_video),
        "--start-frame", "0", "--end-frame", "12", "--setup-json", str(setup),
        "--stereo", str(rig_json), "--out", str(out), "--grid-cols", "3",
        "--grid-rows", "3", "--review-html",
    ]) == 0
    report_path = out / "mesh_track_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    node = next(item for item in report["nodes"] if item["obj_id"] == 4)
    node["suspect_intervals"] = [{
        "start_frame": 4, "end_frame": 8,
        "left_start_frame": 4, "left_end_frame": 8,
        "right_start_frame": 4, "right_end_frame": 8,
        "source_start_frame": 4, "source_end_frame": 8,
        "length": 5, "max_mean_relative_deviation": 0.2,
    }]
    node["one_view_intervals"] = [{
        "start_frame": 2, "end_frame": 7,
        "left_start_frame": 2, "left_end_frame": 7,
        "right_start_frame": 2, "right_end_frame": 7,
        "source_start_frame": 2, "source_end_frame": 7,
        "length": 6,
    }]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    generate_review(out, left_video, right_video, 0, frames=9)
    review = out / "marker_review.html"
    assert review.is_file()
    html = review.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert html.count("data:image/jpeg;base64,") == 18
    assert html.count('"id":') == 9 * 9
    assert '"status":"measured"' in html
    assert '"status":"inferred"' in html
    assert '"suspectNow":true' in html
    assert '"source_start_frame":4' in html
    assert '"oneViewIntervals"' in html
    assert "one-camera only, frames" in html
    assert "check frames" in html and "#8f5cff" in html
    for status in ("measured", "inferred", "repaired", "absent"):
        assert status in html


def main():
    test_synthetic_review()
    print("nettrack review tests passed")


if __name__ == "__main__":
    main()
