#!/usr/bin/env python3
"""Plain assert checks for safe remote pipeline pulls."""

import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import remote_pipeline as remote


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def base_manifest():
    return {
        "queue_id": "queue_a",
        "setup_mode": "grid",
        "grid_cols": 3,
        "grid_rows": 2,
        "preprocessing": {
            "raw_left": "C:/local/raw_left.mp4",
            "raw_right": "C:/local/raw_right.mp4",
            "calibration": {
                "left_video": "C:/local/cal_left.mp4",
                "right_video": "C:/local/cal_right.mp4",
                "stages": {
                    "sync": {"status": "complete"},
                    "stats": {"status": "pending"},
                },
            },
        },
        "settings": {"scale": 0.25},
        "jobs": [
            {
                "index": 1,
                "velocity_slug": "v1",
                "start_frame": 10,
                "end_frame": 20,
                "left_video": "C:/local/job_left.mp4",
                "right_video": "C:/local/job_right.mp4",
                "stages": {
                    "setup": {"status": "complete"},
                    "sam2": {"status": "pending"},
                    "triangulation": {"status": "pending"},
                },
            },
            {
                "index": 2,
                "velocity_slug": "v2",
                "start_frame": 20,
                "end_frame": 30,
                "left_video": "C:/local/job2_left.mp4",
                "right_video": "C:/local/job2_right.mp4",
                "stages": {"setup": {"status": "pending"}},
            },
        ],
    }


def verdict_cases():
    local = base_manifest()

    progress = copy.deepcopy(local)
    progress["preprocessing"]["calibration"]["stages"]["stats"]["status"] = "complete"
    progress["jobs"][0]["stages"]["sam2"]["status"] = "complete"

    offset_regression = copy.deepcopy(local)
    offset_regression["preprocessing"]["calibration"]["stages"]["sync"]["status"] = "pending"
    offset_regression["preprocessing"]["calibration"]["stages"]["stats"]["status"] = "complete"

    missing_job = copy.deepcopy(local)
    missing_job["jobs"].pop()

    differing_velocity = copy.deepcopy(local)
    differing_velocity["jobs"][0]["velocity_slug"] = "other"

    differing_range = copy.deepcopy(local)
    differing_range["jobs"][0]["end_frame"] = 21

    differing_grid = copy.deepcopy(local)
    differing_grid["grid_cols"] = 4

    differing_scale = copy.deepcopy(local)
    differing_scale["settings"]["scale"] = 0.5

    failed_after_complete = copy.deepcopy(local)
    failed_after_complete["jobs"][0]["stages"]["setup"]["status"] = "failed"

    return [
        ("pure progress", progress, True, None),
        ("offset regression", offset_regression, False, "calibration stage 'sync' regressed"),
        ("missing remote job", missing_job, False, "missing job indices"),
        ("differing velocity", differing_velocity, False, "velocity_slug differs"),
        ("differing frame range", differing_range, False, "end_frame differs"),
        ("differing grid", differing_grid, False, "grid_cols differs"),
        ("differing scale", differing_scale, False, "settings.scale differs"),
        ("failed after complete", failed_after_complete, False, "stage 'setup' regressed"),
    ]


def test_superset_verdicts():
    local = base_manifest()
    for label, candidate, expected, reason_fragment in verdict_cases():
        verdict, reasons = remote.remote_manifest_superset_verdict(local, candidate)
        assert verdict is expected, (label, reasons)
        assert remote.remote_manifest_superset(local, candidate) is expected, label
        if reason_fragment:
            assert any(reason_fragment in reason for reason in reasons), (label, reasons)
        else:
            assert reasons == [], (label, reasons)


def test_pull_merge(tmp_root):
    local_queue = tmp_root / "local" / "queue_a"
    remote_queue = tmp_root / "stage" / "queue_a"
    local_manifest = base_manifest()
    remote_manifest = copy.deepcopy(local_manifest)
    remote_manifest["preprocessing"]["raw_left"] = "/remote/raw_left.mp4"
    remote_manifest["preprocessing"]["raw_right"] = "/remote/raw_right.mp4"
    remote_manifest["preprocessing"]["calibration"]["left_video"] = "/remote/cal_left.mp4"
    remote_manifest["preprocessing"]["calibration"]["right_video"] = "/remote/cal_right.mp4"
    remote_manifest["preprocessing"]["calibration"]["stages"]["stats"]["status"] = "complete"
    for job in remote_manifest["jobs"]:
        job["left_video"] = f"/remote/job{job['index']}_left.mp4"
        job["right_video"] = f"/remote/job{job['index']}_right.mp4"

    write_json(local_queue / "queue_manifest.json", local_manifest)
    write_json(remote_queue / "queue_manifest.json", remote_manifest)
    corrections_rel = Path("run_001_v1/setup/prompts/corrections.json")
    write_json(local_queue / corrections_rel, {"owner": "local"})
    write_json(remote_queue / corrections_rel, {"owner": "remote"})
    write_json(local_queue / "split/split_manifest.json", {"left_source": {"path": "C:/local/left.mp4"}})
    write_json(remote_queue / "split/split_manifest.json", {"left_source": {"path": "/remote/left.mp4"}})

    result = remote.merge_staged_pull(local_queue, remote_queue)

    assert result.compatible
    assert result.reasons == []
    assert result.backup_dir and result.backup_dir.is_dir()
    assert json.loads((local_queue / corrections_rel).read_text())["owner"] == "remote"
    backed_up = result.backup_dir / corrections_rel
    assert json.loads(backed_up.read_text())["owner"] == "local"
    assert json.loads((local_queue / "split/split_manifest.json").read_text())["left_source"]["path"] == "C:/local/left.mp4"

    adopted = json.loads((local_queue / "queue_manifest.json").read_text())
    local_preprocessing = local_manifest["preprocessing"]
    adopted_preprocessing = adopted["preprocessing"]
    assert adopted_preprocessing["raw_left"] == local_preprocessing["raw_left"]
    assert adopted_preprocessing["raw_right"] == local_preprocessing["raw_right"]
    assert adopted_preprocessing["calibration"]["left_video"] == local_preprocessing["calibration"]["left_video"]
    assert adopted_preprocessing["calibration"]["right_video"] == local_preprocessing["calibration"]["right_video"]
    local_jobs = {job["index"]: job for job in local_manifest["jobs"]}
    for job in adopted["jobs"]:
        assert job["left_video"] == local_jobs[job["index"]]["left_video"]
        assert job["right_video"] == local_jobs[job["index"]]["right_video"]
    assert adopted_preprocessing["calibration"]["stages"]["stats"]["status"] == "complete"


def test_conflicting_pull_copies_only_remote_manifest(tmp_root):
    local_queue = tmp_root / "conflict_local" / "queue_a"
    remote_queue = tmp_root / "conflict_stage" / "queue_a"
    local_manifest = base_manifest()
    remote_manifest = copy.deepcopy(local_manifest)
    remote_manifest["preprocessing"]["calibration"]["stages"]["sync"]["status"] = "pending"
    corrections_rel = Path("run_001_v1/setup/prompts/corrections.json")
    write_json(local_queue / "queue_manifest.json", local_manifest)
    write_json(remote_queue / "queue_manifest.json", remote_manifest)
    write_json(local_queue / corrections_rel, {"owner": "local"})
    write_json(remote_queue / corrections_rel, {"owner": "remote"})

    result = remote.merge_staged_pull(local_queue, remote_queue)

    assert not result.compatible
    assert result.backup_dir is None
    assert json.loads((local_queue / corrections_rel).read_text())["owner"] == "local"
    assert json.loads((local_queue / "queue_manifest.json").read_text()) == local_manifest
    assert json.loads((local_queue / "queue_manifest.remote.json").read_text()) == remote_manifest


def test_configure_remote_video_roots():
    captured = {}
    original_run_command = remote.run_command

    def fake_run_command(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return remote.CommandResult(0, "/remote/repo/machine_paths.json\n", "")

    config = {
        "host": "gpu-box",
        "remote_repo": "/remote/repo",
        "remote_python": "python3",
        "ssh_extra_args": ["-i", "gpu-key"],
    }
    try:
        remote.run_command = fake_run_command
        remote.configure_remote_video_roots(
            config,
            ["/data/videos", "/data/videos", "/mnt/archive"],
            verbose=True,
        )
    finally:
        remote.run_command = original_run_command

    assert captured["command"][:4] == ["ssh", "-i", "gpu-key", "gpu-box"]
    assert captured["command"][4].startswith("cd /remote/repo && python3 -c ")
    assert json.loads(captured["input_text"]) == {
        "video_roots": ["/data/videos", "/mnt/archive"]
    }

    args = remote.build_parser().parse_args(
        ["configure-paths", "/data/videos", "/mnt/archive", "--dry-run"]
    )
    assert args.func is remote.cmd_configure_paths
    assert args.video_roots == ["/data/videos", "/mnt/archive"]
    assert args.dry_run is True


def test_derive_scp_args():
    assert remote.derive_scp_args(["-p", "2222"]) == ["-P", "2222"]
    assert remote.derive_scp_args(["-p2222"]) == ["-P", "2222"]
    passthrough = ["-i", "key.pem", "-J", "jump", "-o", "key=value", "-4", "-6"]
    assert remote.derive_scp_args(passthrough) == passthrough
    assert remote.derive_scp_args(["-t", "-T", "-p", "22"]) == ["-P", "22"]
    assert remote.derive_scp_args([]) == []


def test_scp_command_honors_explicit_extra_args():
    config = {
        "host": "gpu",
        "ssh_extra_args": ["-p", "2222", "-t"],
        "scp_extra_args": ["-P", "2200", "-o", "Compression=no"],
    }
    assert remote.scp_command(config, "source", "dest") == [
        "scp",
        "-P",
        "2200",
        "-o",
        "Compression=no",
        "source",
        "dest",
    ]


def main():
    test_derive_scp_args()
    test_scp_command_honors_explicit_extra_args()
    test_superset_verdicts()
    test_configure_remote_video_roots()
    tmp_parent = Path(__file__).resolve().parent / "_tmp"
    tmp_root = tmp_parent / "remote_pipeline"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    try:
        test_pull_merge(tmp_root)
        test_conflicting_pull_copies_only_remote_manifest(tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("remote pipeline tests passed")


if __name__ == "__main__":
    main()
