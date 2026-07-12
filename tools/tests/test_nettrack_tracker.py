#!/usr/bin/env python3
"""Plain-assert synthetic checks for the Phase 1b mesh tracker."""

from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nettrack.geometry import triangulate_point
from nettrack.run_tracker import TRACK_FIELDNAMES, main as tracker_main
from nettrack.synth import SyntheticNetScene
from nettrack.topology import NetTopology
from nettrack.tracker import MeshTracker


FPS = 20.0
IMAGE_SIZE = (800, 600)


def setup_points(scene, t=0.0):
    return tuple(
        np.asarray([[item["u"], item["v"]] for item in scene.ground_truth(camera, t)])
        for camera in scene.cameras
    )


def run_scene(scene, frame_count, topology=None):
    times = np.arange(frame_count, dtype=float) / FPS
    left, right = setup_points(scene)
    topology = topology or NetTopology.from_grid(scene.grid_cols, scene.grid_rows)
    result = MeshTracker(scene.rig, topology).track(
        (scene.render_frame("left", t) for t in times),
        (scene.render_frame("right", t) for t in times),
        left,
        right,
    )
    truth = np.asarray([scene.points_3d(t) for t in times])
    return result, truth


def rig_noise_bound(scene, sigma_px=0.5):
    """Compute a rig-specific RMS depth bound from the requested pixel noise."""
    rng = np.random.default_rng(1234)
    errors = []
    for t in (0.0, 1.5, 2.95):
        xyz = scene.points_3d(t)
        left = np.asarray([[item["u"], item["v"]] for item in scene.ground_truth(scene.rig.left, t)])
        right = np.asarray([[item["u"], item["v"]] for item in scene.ground_truth(scene.rig.right, t)])
        for index in rng.choice(len(xyz), size=min(30, len(xyz)), replace=False):
            for _ in range(4):
                recovered = triangulate_point(
                    left[index] + rng.normal(0.0, sigma_px, 2),
                    right[index] + rng.normal(0.0, sigma_px, 2),
                    scene.rig,
                )
                errors.append(np.linalg.norm(recovered - xyz[index]))
    return float(np.sqrt(np.mean(np.square(errors))))


def assert_no_identity_errors(result, truth, radius):
    errors = np.linalg.norm(result.positions - truth, axis=2)
    assert float(np.max(errors)) < radius, (float(np.max(errors)), radius)


def test_nominal():
    scene = SyntheticNetScene(12, 12, image_size=IMAGE_SIZE, seed=73)
    result, truth = run_scene(scene, 60)
    noise_bound = rig_noise_bound(scene)
    topology = result.topology
    edge = topology.edge_indices
    spacing_3d = float(np.median(np.linalg.norm(truth[0, edge[:, 0]] - truth[0, edge[:, 1]], axis=1)))
    assert_no_identity_errors(result, truth, max(4.0 * noise_bound, 0.30 * spacing_3d))
    rmse = float(np.sqrt(np.mean(np.square(result.positions - truth))))
    assert rmse < 1.5 * noise_bound, (rmse, noise_bound)
    edge_std = np.asarray([frame.metrics["edge_length_std"] for frame in result.frames])
    assert float(np.ptp(edge_std)) < 0.08 * spacing_3d, np.ptp(edge_std)
    assert result.report["coordinate_residual_space"] == "distorted_pixels"


def test_dropouts_and_recovery():
    hidden_id = 65
    scene = SyntheticNetScene(
        12, 12, image_size=IMAGE_SIZE, seed=19, dropout_probability=0.10,
        hidden_intervals={hidden_id: (15 / FPS, 30 / FPS)},
    )
    result, truth = run_scene(scene, 40)
    hidden_status = [result.frames[index].statuses[hidden_id] for index in range(15, 30)]
    assert hidden_status == ["inferred"] * 15, hidden_status
    assert result.frames[30].statuses[hidden_id].startswith("measured")
    hidden_error = np.linalg.norm(result.positions[15:30, hidden_id] - truth[15:30, hidden_id], axis=1)
    edge = result.topology.edge_indices
    spacing = float(np.median(np.linalg.norm(truth[0, edge[:, 0]] - truth[0, edge[:, 1]], axis=1)))
    assert float(np.max(hidden_error)) < 0.35 * spacing, np.max(hidden_error)
    assert result.report["totals"]["lost_vertex_frames"] == 0
    assert_no_identity_errors(result, truth, 0.48 * spacing)


def test_near_crossing_one_view():
    pair = (65, 66)
    scene = SyntheticNetScene(
        12, 12, image_size=IMAGE_SIZE, seed=29, crossing_pair=pair,
        crossing_time_s=1.5, crossing_width_s=0.50, crossing_separation_px=20.0,
    )
    result, truth = run_scene(scene, 60)
    left_separations = []
    for index in range(60):
        uv = np.asarray([
            [item["u"], item["v"]]
            for item in scene.ground_truth(scene.rig.left, index / FPS)
        ])
        left_separations.append(np.linalg.norm(uv[pair[0]] - uv[pair[1]]))
    assert min(left_separations) < 2.0 * 4.0 * scene.marker_sigma_px
    for frame_index, frame in enumerate(result.frames):
        first, second = pair
        for camera, assignments in (
            (scene.rig.left, frame.left_assignments), (scene.rig.right, frame.right_assignments)
        ):
            uv = np.asarray([
                [item["u"], item["v"]]
                for item in scene.ground_truth(camera, frame_index / FPS)
            ])
            for own, other in ((first, second), (second, first)):
                detection = assignments[own]
                if detection is not None:
                    observed = np.asarray([detection.u, detection.v])
                    assert np.linalg.norm(observed - uv[own]) < np.linalg.norm(observed - uv[other])


def test_holes():
    present = np.ones((12, 12), dtype=bool)
    present[3:9, 3:9] = False
    topology = NetTopology.from_grid(12, 12, present=present)
    assert len(topology) == 108
    scene = SyntheticNetScene(12, 12, image_size=IMAGE_SIZE, seed=41, present=present)
    result, truth = run_scene(scene, 24, topology)
    assert set(result.report["node_ids"]) == set(topology.node_ids)
    assert all(a in topology.node_ids and b in topology.node_ids for a, b in topology.edges)
    edge = topology.edge_indices
    spacing = float(np.median(np.linalg.norm(truth[0, edge[:, 0]] - truth[0, edge[:, 1]], axis=1)))
    assert_no_identity_errors(result, truth, 0.40 * spacing)


def test_csv_contract_cli():
    scene = SyntheticNetScene(4, 3, image_size=(480, 360), seed=53)
    root = Path(__file__).with_name("_tmp") / "nettrack_cli_fixed"
    root.mkdir(parents=True, exist_ok=True)
    with nullcontext(root):
        left_video, right_video, rig_json = scene.write_videos(
            root, 8, FPS, video_suffix=".avi", codec="MJPG"
        )
        setup_time = 2 / FPS
        left, right = setup_points(scene, setup_time)
        setup_path = root / "setup.json"
        setup_path.write_text(json.dumps({
            "left": left.tolist(), "right": right.tolist(),
            "crops": {
                "left": {"crop_applied": False}, "right": {"crop_applied": False},
            },
            "frame_range": {"start_frame": 2, "end_frame": 7},
        }), encoding="utf-8")
        corrections = root / "corrections.json"
        corrections.write_text(json.dumps({
            "version": 1, "coordinate_space": "cropped_original_scale",
            "corrections": {"left": [], "right": []},
        }), encoding="utf-8")
        sync = root / "sync.json"
        sync.write_text(json.dumps({"trim_left": 0, "trim_right": 0}), encoding="utf-8")
        out = root / "out"
        status = tracker_main([
            "--left-input", str(left_video), "--right-input", str(right_video),
            "--start-frame", "2", "--end-frame", "7", "--setup-json", str(setup_path),
            "--corrections-json", str(corrections), "--stereo", str(rig_json),
            "--sync-json", str(sync), "--out", str(out), "--grid-cols", "4", "--grid-rows", "3",
        ])
        assert status == 0
        report = json.loads((out / "mesh_track_report.json").read_text())
        for side in ("left", "right"):
            with (out / side / "tracks_2d.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                assert reader.fieldnames == TRACK_FIELDNAMES
                rows = list(reader)
            assert sorted({int(row["frame"]) for row in rows}) == list(range(5))
            assert sorted({int(row["obj_id"]) for row in rows}) == list(range(12))
            assert len(rows) == 5 * 12
            invalid = sum(int(row["valid"]) == 0 for row in rows)
            assert invalid == report["totals"]["lost_vertex_frames"]


def main():
    test_nominal()
    test_dropouts_and_recovery()
    test_near_crossing_one_view()
    test_holes()
    test_csv_contract_cli()
    print("nettrack tracker tests passed")


if __name__ == "__main__":
    main()
