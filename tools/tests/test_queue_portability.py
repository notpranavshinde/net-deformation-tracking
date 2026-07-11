#!/usr/bin/env python3
"""Plain assert checks for portable pipeline queue manifests."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_pipeline_queue as queue


class Args:
    sam2_scale = None
    sam2_model_id = None
    check_only = False


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def make_fake_video(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 12),
        )
        if not writer.isOpened():
            raise RuntimeError("VideoWriter unavailable")
        color = int(sum(payload) % 255)
        for index in range(4):
            frame = np.full((12, 16, 3), (color + index) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
    except Exception:
        path.write_bytes(payload)


def build_manifest(repo_root, queue_dir, video_root):
    left_video = video_root / "left_cam.mp4"
    right_video = video_root / "right_cam.mp4"
    make_fake_video(left_video, b"left-video-bytes")
    make_fake_video(right_video, b"right-video-bytes")

    split_dir = queue_dir / "split"
    calibration_dir = queue_dir / "calibration"
    run_dir = queue_dir / "run_001_v1"
    setup_dir = run_dir / "setup"
    result_dir = repo_root / "triangulation" / "results" / "v1"

    points_json = setup_dir / "prompts" / "points_left_right.json"
    corrections_json = setup_dir / "prompts" / "corrections.json"
    write_json(
        points_json,
        {
            "left": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "right": [[0, 0], [1, 0], [0, 1], [1, 1]],
        },
    )
    write_json(
        setup_dir / "setup_manifest.json",
        {
            "left_input": str(left_video.resolve()),
            "right_input": str(right_video.resolve()),
            "points_json": str(points_json.resolve()),
            "left_count": 4,
            "right_count": 4,
            "frame_range": {"start_frame": 10, "end_frame": 20},
        },
    )
    write_json(corrections_json, {"version": 1, "corrections": {"left": [], "right": []}})

    split_payload = {
        "version": 1,
        "mode": "virtual",
        "status": "complete",
        "left_source": queue.video_signature(left_video),
        "right_source": queue.video_signature(right_video),
        "reference_side": "right",
        "output_root": str((split_dir / "clips").resolve()),
        "fps": 30.0,
        "regions": [
            {"start_frame": 0, "end_frame": 10},
            {"start_frame": 10, "end_frame": 20},
        ],
        "clips": [
            {
                "index": 1,
                "start_frame": 0,
                "end_frame": 10,
                "left_video": str(left_video.resolve()),
                "right_video": str(right_video.resolve()),
            },
            {
                "index": 2,
                "start_frame": 10,
                "end_frame": 20,
                "left_video": str(left_video.resolve()),
                "right_video": str(right_video.resolve()),
            },
        ],
    }
    split_manifest = split_dir / "split_manifest.json"
    write_json(split_manifest, split_payload)

    for path in (
        calibration_dir / "sync.json",
        calibration_dir / "stats" / "stats_left.json",
        calibration_dir / "stats" / "stats_right.json",
        calibration_dir / "mono.npz",
        calibration_dir / "mono_report.json",
        calibration_dir / "stereo.npz",
        calibration_dir / "stereo_report.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    manifest = {
        "version": 2,
        "queue_id": "queue_a",
        "created_at": queue.utc_now(),
        "updated_at": queue.utc_now(),
        "setup_mode": "grid",
        "grid_cols": 2,
        "grid_rows": 2,
        "section_layout": None,
        "preprocessing": {
            "mode": "split_raw_pair",
            "raw_left": str(left_video.resolve()),
            "raw_right": str(right_video.resolve()),
            "splitter": {
                "output_root": str((split_dir / "clips").resolve()),
                "manifest": str(split_manifest.resolve()),
                "stage": {"status": "complete"},
            },
            "calibration": {
                "mode": "generated",
                "work_dir": str(calibration_dir.resolve()),
                "left_video": str(left_video.resolve()),
                "right_video": str(right_video.resolve()),
                "split_clip_index": 1,
                "start_frame": 0,
                "end_frame": 10,
                "sync_json": str((calibration_dir / "sync.json").resolve()),
                "stats_dir": str((calibration_dir / "stats").resolve()),
                "mono_npz": str((calibration_dir / "mono.npz").resolve()),
                "stereo_npz": str((calibration_dir / "stereo.npz").resolve()),
                "settings": {
                    "cols": 9,
                    "rows": 7,
                    "square_mm": 40.0,
                    "scale": 1.0,
                    "workers": 32,
                    "sync_mode": "audio",
                    "stats_step": 1,
                    "stats_max_scan": 100000,
                    "mono_max_scan": 100,
                    "stereo_max_pairs": 50,
                },
                "stages": {
                    "sync": {"status": "complete", "fingerprint": "old"},
                    "stats": {"status": "complete", "fingerprint": "old"},
                    "mono": {"status": "complete", "fingerprint": "old"},
                    "stereo": {"status": "complete", "fingerprint": "old"},
                },
            },
        },
        "settings": {
            "scale": 0.2,
            "gpu_mode": "dual",
            "batch_size": 36,
            "preview": False,
            "visualize": True,
            "model_id": queue.SAM2_MODEL_ID,
            "frame_step": 1,
        },
        "jobs": [
            {
                "index": 1,
                "split_clip_index": 2,
                "velocity": "v1",
                "velocity_slug": "v1",
                "left_video": str(left_video.resolve()),
                "right_video": str(right_video.resolve()),
                "start_frame": 10,
                "end_frame": 20,
                "run_dir": str(run_dir.resolve()),
                "setup_dir": str(setup_dir.resolve()),
                "setup_json": str(points_json.resolve()),
                "corrections_json": str(corrections_json.resolve()),
                "sam2_out": str((run_dir / "sam2").resolve()),
                "logs_dir": str((run_dir / "logs").resolve()),
                "result_dir": str(result_dir.resolve()),
                "stages": {
                    "setup": {"status": "complete"},
                    "sam2": {"status": "pending"},
                    "triangulation": {"status": "pending"},
                },
            }
        ],
    }
    manifest_path = queue_dir / "queue_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, left_video, right_video


def main():
    tmp_parent = Path(__file__).resolve().parent / "_tmp"
    tmp_parent.mkdir(exist_ok=True)
    tmp_root = tmp_parent / "run"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir()
    try:
        repo_root = tmp_root / "repo"
        queue.REPO_ROOT = repo_root
        queue.QUEUE_ROOT = repo_root / "work" / "pipeline_queue"
        queue.RESULTS_ROOT = repo_root / "triangulation" / "results"
        queue.MACHINE_PATHS_FILE = repo_root / "machine_paths.json"
        repo_root.mkdir(parents=True)

        fresh_video_root = tmp_root / "fresh_videos"
        fresh_left = fresh_video_root / "fresh_left.mp4"
        fresh_right = fresh_video_root / "fresh_right.mp4"
        make_fake_video(fresh_left, b"fresh-left-video-bytes")
        make_fake_video(fresh_right, b"fresh-right-video-bytes")
        prompt_videos = iter([fresh_left, fresh_right])
        original_prompt_video_path = queue.prompt_video_path
        original_prompt_setup_mode = queue.prompt_setup_mode
        original_prompt_positive_int = queue.prompt_positive_int
        try:
            queue.prompt_video_path = lambda _label: next(prompt_videos)
            queue.prompt_setup_mode = lambda: "grid"
            queue.prompt_positive_int = lambda _label: 2
            fresh_path, fresh_manifest = queue.create_manifest(Args())
        finally:
            queue.prompt_video_path = original_prompt_video_path
            queue.prompt_setup_mode = original_prompt_setup_mode
            queue.prompt_positive_int = original_prompt_positive_int
        assert fresh_manifest["version"] == 3
        assert json.loads(fresh_path.read_text())["version"] == 3

        queue_dir = queue.QUEUE_ROOT / "queue_a"
        video_root = tmp_root / "videos_a"
        manifest_path, left_video, right_video = build_manifest(repo_root, queue_dir, video_root)

        loaded_path, manifest = queue.load_manifest(manifest_path)
        changed = queue.migrate_manifest(manifest, loaded_path)
        assert changed
        for stage_name in ("sync", "stats", "mono", "stereo"):
            assert queue.calibration_stage_complete(manifest, stage_name)
        migrated_sync_fp = manifest["preprocessing"]["calibration"]["stages"]["sync"]["fingerprint"]
        migrated_stats_fp = manifest["preprocessing"]["calibration"]["stages"]["stats"]["fingerprint"]

        Path(manifest["preprocessing"]["calibration"]["sync_json"]).write_bytes(b"changed-sync")
        queue.migrate_manifest(manifest, loaded_path)
        assert manifest["preprocessing"]["calibration"]["stages"]["stats"]["fingerprint"] == migrated_stats_fp
        assert not queue.calibration_stage_complete(manifest, "stats")
        queue.save_manifest(loaded_path, manifest)

        raw_saved = json.loads(loaded_path.read_text())
        assert raw_saved["version"] == 3
        assert raw_saved["preprocessing"]["splitter"]["manifest"].startswith("queue:")
        assert raw_saved["jobs"][0]["setup_dir"].startswith("queue:")
        assert raw_saved["jobs"][0]["result_dir"].startswith("repo:")

        copied_queue = queue.QUEUE_ROOT / "queue_b"
        shutil.copytree(queue_dir, copied_queue)
        new_video_root = tmp_root / "videos_b"
        new_video_root.mkdir()
        shutil.move(str(left_video), new_video_root / left_video.name)
        shutil.move(str(right_video), new_video_root / right_video.name)
        shutil.rmtree(video_root)
        write_json(queue.MACHINE_PATHS_FILE, {"video_roots": [str(new_video_root)]})

        hopped_path, hopped = queue.load_manifest(copied_queue)
        queue.resolve_source_videos(hopped_path, hopped)
        queue.migrate_manifest(hopped, hopped_path)
        assert Path(hopped["preprocessing"]["raw_left"]).parent == new_video_root
        assert queue.calibration_stage_fingerprint(hopped, "sync") == migrated_sync_fp
        assert queue.setup_complete(hopped["jobs"][0], hopped)

        missing_setup = json.loads(json.dumps(hopped))
        Path(missing_setup["jobs"][0]["setup_json"]).unlink()
        missing, command = queue.process_only_missing(hopped_path, missing_setup)
        assert any("setup is incomplete" in item for item in missing)
        assert "--setup-only" in command
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("queue portability tests passed")


if __name__ == "__main__":
    main()
