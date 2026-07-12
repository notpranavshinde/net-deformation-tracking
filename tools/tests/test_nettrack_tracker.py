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
from nettrack.bootstrap import bootstrap_mesh
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


def run_scene(scene, frame_count, topology=None, setups=None):
    times = np.arange(frame_count, dtype=float) / FPS
    left, right = setups if setups is not None else setup_points(scene)
    topology = topology or NetTopology.from_grid(scene.grid_cols, scene.grid_rows)
    result = MeshTracker(scene.rig, topology).track(
        (scene.render_frame("left", t) for t in times),
        (scene.render_frame("right", t) for t in times),
        left,
        right,
    )
    truth = np.asarray([scene.points_3d(t) for t in times])
    return result, truth


def corrupted_setups(scene):
    """Exercise water offsets, bilateral corruption, swaps, and neighbor hints."""
    truth = list(setup_points(scene))
    hints = [value.copy() for value in truth]
    corrupt_ids = {13, 18, 27, 33, 34, 40, 49, 58, 67, 75, 91, 96, 106, 119, 124}
    for index in (13, 27, 40, 58, 75, 91, 106, 124):
        hints[0][index] += (55.0, -42.0)
    for index in (18, 67, 119):
        hints[0][index] += (-60.0, 45.0)
        hints[1][index] += (65.0, -48.0)
    hints[0][33], hints[0][34] = hints[0][34].copy(), hints[0][33].copy()
    for index in (49, 96):
        hints[0][index] = truth[0][index + 1]
    return tuple(hints), corrupt_ids


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
    hidden_index = result.topology.index(hidden_id)
    hidden_status = [result.frames[index].statuses[hidden_index] for index in range(15, 30)]
    assert hidden_status == ["inferred"] * 15, hidden_status
    assert result.frames[30].statuses[hidden_index].startswith("measured")
    hidden_error = np.linalg.norm(result.positions[15:30, hidden_index] - truth[15:30, hidden_id], axis=1)
    active_truth = truth[:, np.asarray(result.topology.node_ids, dtype=int)]
    full_edge = NetTopology.from_grid(scene.grid_cols, scene.grid_rows).edge_indices
    spacing = float(np.median(np.linalg.norm(truth[0, full_edge[:, 0]] - truth[0, full_edge[:, 1]], axis=1)))
    assert float(np.max(hidden_error)) < 0.35 * spacing, np.max(hidden_error)
    assert result.report["totals"]["lost_vertex_frames"] == 0
    assert_no_identity_errors(result, active_truth, 0.48 * spacing)


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


def test_bootstrap_repairs_corrupted_hints_and_full_track():
    scene = SyntheticNetScene(12, 12, image_size=IMAGE_SIZE, seed=83)
    setups, corrupt_ids = corrupted_setups(scene)
    result, truth = run_scene(scene, 24, setups=setups)
    audit = result.report["bootstrap"]
    assert audit["summary"]["absent"] == 0, audit["summary"]
    repaired = {node["obj_id"] for node in audit["nodes"] if node["status"] == "repaired"}
    assert repaired == corrupt_ids, (repaired, corrupt_ids)
    assert audit["summary"]["monotonic_violations"] == 0
    edge = result.topology.edge_indices
    spacing = float(np.median(np.linalg.norm(truth[0, edge[:, 0]] - truth[0, edge[:, 1]], axis=1)))
    assert_no_identity_errors(result, truth, 0.30 * spacing)
    noise_bound = rig_noise_bound(scene)
    rmse = float(np.sqrt(np.mean(np.square(result.positions - truth))))
    assert rmse < 1.5 * noise_bound, (rmse, noise_bound)


def test_bootstrap_absent_frame_zero_dropout():
    hidden_id = 65
    scene = SyntheticNetScene(
        12, 12, image_size=IMAGE_SIZE, seed=84, dropout_probability=1e-12,
        hidden_intervals={hidden_id: (0.0, 0.1)},
    )
    result, truth = run_scene(scene, 8)
    node = result.report["bootstrap"]["nodes"][hidden_id]
    assert node["status"] == "absent" and node["reason"] == "no_detection", node
    assert hidden_id not in result.topology.node_ids
    kept = np.asarray(result.topology.node_ids, dtype=int)
    spacing = float(np.median(np.linalg.norm(truth[0, 1:] - truth[0, :-1], axis=1)))
    assert float(np.max(np.linalg.norm(result.positions - truth[:, kept], axis=2))) < 0.40 * spacing


def test_bootstrap_holed_grid_with_corruption():
    present = np.ones((12, 12), dtype=bool)
    present[3:9, 3:9] = False
    topology = NetTopology.from_grid(12, 12, present=present)
    scene = SyntheticNetScene(12, 12, image_size=IMAGE_SIZE, seed=85, present=present)
    truth = list(setup_points(scene))
    hints = [value.copy() for value in truth]
    packed = (5, 17, 42, 71, 96)
    for index in packed:
        hints[0][index] += (54.0, -40.0)
    audit = bootstrap_mesh(
        scene.render_frame("left", 0.0), scene.render_frame("right", 0.0),
        hints[0], hints[1], topology, scene.rig,
    )
    repaired = {node["obj_id"] for node in audit.report["nodes"] if node["status"] == "repaired"}
    assert repaired == {topology.node_ids[index] for index in packed}, repaired
    assert audit.report["summary"]["absent"] == 0
    assert audit.topology.node_ids == topology.node_ids
    assert float(np.max(np.linalg.norm(audit.left_points - truth[0], axis=1))) < 1.0
    assert float(np.max(np.linalg.norm(audit.right_points - truth[1], axis=1))) < 1.0


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
        audit_out = root / "audit_out"
        status = tracker_main([
            "--audit-only", "--left-input", str(left_video), "--right-input", str(right_video),
            "--start-frame", "2", "--end-frame", "7", "--setup-json", str(setup_path),
            "--stereo", str(rig_json), "--sync-json", str(sync), "--out", str(audit_out),
            "--grid-cols", "4", "--grid-rows", "3", "--review-html",
        ])
        assert status == 0
        audit = json.loads((audit_out / "setup_audit.json").read_text())
        assert audit["summary"]["confirmed"] == 12
        assert (audit_out / "bootstrap_overlay_left.png").is_file()
        audit_html = (audit_out / "marker_review.html").read_text(encoding="utf-8")
        assert audit_html.count("data:image/jpeg;base64,") == 2
        assert not (audit_out / "mesh_track_report.json").exists()


def main():
    test_nominal()
    test_dropouts_and_recovery()
    test_near_crossing_one_view()
    test_holes()
    test_bootstrap_repairs_corrupted_hints_and_full_track()
    test_bootstrap_absent_frame_zero_dropout()
    test_bootstrap_holed_grid_with_corruption()
    test_csv_contract_cli()
    print("nettrack tracker tests passed")


if __name__ == "__main__":
    main()
