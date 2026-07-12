#!/usr/bin/env python3
"""Plain-assert synthetic checks for the Phase 1a nettrack detector."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nettrack.colormodel import MarkerColorModel
from nettrack.detect import detect_markers
from nettrack.geometry import load_stereo_calibration, project_point, triangulate_point
from nettrack.synth import SyntheticNetScene


def match_errors(detections, truth, max_match_distance=2.0):
    visible = [item for item in truth if item["visible"]]
    expected = np.array([[item["u"], item["v"]] for item in visible], dtype=float)
    actual = np.array([[item.u, item.v] for item in detections], dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return np.empty(0), 0
    distances = np.linalg.norm(expected[:, None, :] - actual[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    errors = distances[rows, cols]
    accepted = errors <= max_match_distance
    return errors[accepted], int(np.count_nonzero(accepted))


def assert_detection_frame(scene, camera, t, model, frame=None):
    frame = scene.render_frame(camera, t) if frame is None else frame
    truth = scene.ground_truth(camera, t)
    visible_count = sum(item["visible"] for item in truth)
    detections = detect_markers(frame, model, expected_count=visible_count)
    errors, matched = match_errors(detections, truth)
    recall = matched / max(visible_count, 1)
    assert recall >= 0.99, (camera.name, t, visible_count, len(detections), recall)
    assert len(detections) == visible_count, (camera.name, t, visible_count, len(detections))
    assert float(np.mean(errors)) < 0.5, (camera.name, t, np.mean(errors))
    assert float(np.max(errors)) < 1.5, (camera.name, t, np.max(errors))


def test_detection_and_white_balance():
    scene = SyntheticNetScene(12, 12, seed=73)
    models = {}
    for camera in scene.cameras:
        frame0 = scene.render_frame(camera, 0.0)
        points = np.array(
            [[item["u"], item["v"]] for item in scene.ground_truth(camera, 0.0) if item["visible"]]
        )
        model = MarkerColorModel().fit(frame0, points)
        restored = MarkerColorModel.from_json(model.to_json())
        assert np.allclose(restored.ab_mean, model.ab_mean)
        assert np.allclose(restored.ab_cov, model.ab_cov)
        assert restored.area_median == model.area_median
        models[camera.name] = restored

    for frame_index in range(10):
        t = frame_index / 20.0
        for camera in scene.cameras:
            frame = scene.render_frame(camera, t)
            assert_detection_frame(scene, camera, t, models[camera.name], frame)
            gains = np.array([1.08, 0.96, 1.10], dtype=np.float32)
            shifted = np.clip(frame.astype(np.float32) * gains, 0.0, 255.0).astype(np.uint8)
            assert_detection_frame(scene, camera, t, models[camera.name], shifted)


def test_dropout_is_not_hallucinated():
    clean = SyntheticNetScene(12, 12, seed=91)
    dropout = SyntheticNetScene(12, 12, seed=91, dropout_probability=0.12)
    for clean_camera, dropout_camera in zip(clean.cameras, dropout.cameras):
        frame0 = clean.render_frame(clean_camera, 0.0)
        setup = np.array([[item["u"], item["v"]] for item in clean.ground_truth(clean_camera, 0.0)])
        model = MarkerColorModel().fit(frame0, setup)
        t = 0.35
        truth = dropout.ground_truth(dropout_camera, t)
        invisible = [item for item in truth if not item["visible"]]
        assert invisible
        detections = detect_markers(
            dropout.render_frame(dropout_camera, t),
            model,
            expected_count=sum(item["visible"] for item in truth),
        )
        errors, matched = match_errors(detections, truth)
        visible_count = sum(item["visible"] for item in truth)
        assert matched / visible_count >= 0.99
        assert len(detections) == visible_count
        assert np.mean(errors) < 0.5
        for missing in invisible:
            nearest = min(
                (np.hypot(d.u - missing["u"], d.v - missing["v"]) for d in detections),
                default=np.inf,
            )
            assert nearest > 3.0, (clean_camera.name, missing["marker_id"], nearest)


def test_geometry_round_trip():
    scene = SyntheticNetScene(12, 12, seed=17)
    rig_path = Path(__file__).with_name("_nettrack_synthetic_rig.json")
    try:
        rig_path.write_text(json.dumps({"rig": scene.rig.to_dict()}), encoding="utf-8")
        rig = load_stereo_calibration(rig_path)
    finally:
        rig_path.unlink(missing_ok=True)
    rng = np.random.default_rng(5)
    errors = []
    for point in scene.points_3d(0.4):
        left = project_point(point, rig.left) + rng.normal(0.0, 0.2, 2)
        right = project_point(point, rig.right) + rng.normal(0.0, 0.2, 2)
        recovered = triangulate_point(left, right, rig)
        errors.append(np.linalg.norm(recovered - point))
    assert float(np.mean(errors)) < 0.015, np.mean(errors)
    assert float(np.max(errors)) < 0.05, np.max(errors)


def main():
    test_detection_and_white_balance()
    test_dropout_is_not_hallucinated()
    test_geometry_round_trip()
    print("nettrack detection tests passed")


if __name__ == "__main__":
    main()
