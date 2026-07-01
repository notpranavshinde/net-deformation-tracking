#!/usr/bin/env python3
"""Prepare multiple stereo runs interactively, then process them unattended."""

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SAM2_DIR = REPO_ROOT / "sam2" / "sam2"
MARKERS_SCRIPT = SAM2_DIR / "run_sam2_markers.py"
OBJECTWISE_SCRIPT = SAM2_DIR / "run_sam2_objectwise.py"
TRIANGULATION_SCRIPT = REPO_ROOT / "triangulation" / "points_to_3d.py"
QUEUE_ROOT = REPO_ROOT / "work" / "pipeline_queue"
RESULTS_ROOT = REPO_ROOT / "triangulation" / "results"
SAM2_MODEL_ID = "facebook/sam2-hiera-large"
QUEUE_SAM2_META = "queue_sam2_meta.json"
QUEUE_TRIANGULATION_META = "queue_triangulation_meta.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def prompt_positive_int(label):
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if value > 0:
            return value
        print("Enter a number greater than zero.")


def prompt_video_path(label):
    while True:
        raw = input(f"{label}: ").strip().strip('"').strip("'")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if path.is_file():
            return path
        print(f"Video not found: {path}")


def velocity_slug(value):
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def stable_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_signature(path, include_hash=False):
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    sig = {
        "path": str(p.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        sig["sha256"] = file_sha256(p)
    return sig


def video_signature(path):
    p = Path(path)
    sig = file_signature(p)
    if not sig.get("exists"):
        return sig
    try:
        import cv2

        cap = cv2.VideoCapture(str(p))
        sig.update({
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else -1,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else -1,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else -1.0,
        })
        cap.release()
    except Exception:
        sig.update({"frame_count": -1, "width": -1, "height": -1, "fps": -1.0})
    return sig


def objectwise_video_signature(path):
    p = Path(path)
    stat = p.stat()
    try:
        import cv2

        cap = cv2.VideoCapture(str(p))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else -1
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else -1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else -1.0
        cap.release()
    except Exception:
        frame_count = -1
        width = -1
        height = -1
        fps = -1.0
    return {
        "path": str(p.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
    }


def same_path(a, b):
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except Exception:
        return False


def count_csv_rows(path):
    p = Path(path)
    if not p.is_file():
        return -1
    with p.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def save_manifest(manifest_path, manifest):
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def run_logged(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$", " ".join(str(part) for part in command))
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] $ {' '.join(str(part) for part in command)}\n")
        popen_kwargs = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **popen_kwargs,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        except KeyboardInterrupt:
            if hasattr(os, "killpg"):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
            log.write(f"[{utc_now()}] interrupted\n")
            raise
        log.write(f"[{utc_now()}] exit_code={return_code}\n")
    return return_code


def setup_complete(job, manifest=None):
    setup_json = Path(job["setup_json"])
    setup_manifest = Path(job["setup_dir"]) / "setup_manifest.json"
    if not setup_json.is_file() or not setup_manifest.is_file():
        return False
    try:
        points_payload = read_json(setup_json)
        setup_meta = read_json(setup_manifest)
    except Exception:
        return False
    if not same_path(setup_meta.get("left_input"), job["left_video"]):
        return False
    if not same_path(setup_meta.get("right_input"), job["right_video"]):
        return False
    if not same_path(setup_meta.get("points_json"), setup_json):
        return False

    left_points = points_payload.get("left")
    right_points = points_payload.get("right")
    if not isinstance(left_points, list) or not isinstance(right_points, list):
        return False
    expected = None
    if manifest is not None:
        expected = int(manifest["grid_cols"]) * int(manifest["grid_rows"])
    if expected is not None and (len(left_points) != expected or len(right_points) != expected):
        return False
    if int(setup_meta.get("left_count", -1)) != len(left_points):
        return False
    if int(setup_meta.get("right_count", -1)) != len(right_points):
        return False
    return True


def load_setup_payload(job):
    payload = read_json(job["setup_json"])
    crops = payload.get("crops", {})

    def side_crop(side):
        crop_meta = crops.get(side, {})
        if not crop_meta or not crop_meta.get("crop_applied", False):
            return None
        return [
            int(crop_meta["x"]),
            int(crop_meta["y"]),
            int(crop_meta["w"]),
            int(crop_meta["h"]),
        ]

    return {
        "left": {
            "points": payload["left"],
            "crop": side_crop("left"),
            "video": job["left_video"],
        },
        "right": {
            "points": payload["right"],
            "crop": side_crop("right"),
            "video": job["right_video"],
        },
    }


def load_corrections_payload(job):
    path = Path(job["corrections_json"])
    if not path.is_file():
        return {
            "version": 1,
            "coordinate_space": "cropped_original_scale",
            "corrections": {"left": [], "right": []},
        }
    payload = read_json(path)
    corrections = payload.setdefault("corrections", {})
    corrections.setdefault("left", [])
    corrections.setdefault("right", [])
    payload.setdefault("version", 1)
    payload.setdefault("coordinate_space", "cropped_original_scale")
    return payload


def corrections_for_side(payload, side_name):
    return list(payload.get("corrections", {}).get(str(side_name).lower(), []))


def chunk_ids(ids, batch_size):
    batch_size = max(1, int(batch_size))
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def expected_object_ids(manifest):
    total = int(manifest["grid_cols"]) * int(manifest["grid_rows"])
    return list(range(total))


def batch_fingerprint(side_name, batch_ids, setup_side, manifest, corrections):
    wanted = {int(obj_id) for obj_id in batch_ids}
    batch_corrections = [
        corr for corr in corrections_for_side(corrections, side_name)
        if int(corr.get("obj_id", -1)) in wanted
    ]
    return stable_hash({
        "version": 2,
        "side": side_name,
        "object_ids": [int(obj_id) for obj_id in batch_ids],
        "video": objectwise_video_signature(setup_side["video"]),
        "crop": list(setup_side["crop"]) if setup_side["crop"] is not None else None,
        "scale": float(manifest["settings"]["scale"]),
        "points": [setup_side["points"][obj_id] for obj_id in batch_ids],
        "corrections": batch_corrections,
        "model_id": SAM2_MODEL_ID,
        "batch_size": int(manifest["settings"]["batch_size"]),
    })


def sam2_stage_fingerprint(job, manifest):
    return stable_hash({
        "version": 1,
        "stage": "sam2",
        "left_video": video_signature(job["left_video"]),
        "right_video": video_signature(job["right_video"]),
        "setup_json": file_signature(job["setup_json"], include_hash=True),
        "corrections_json": file_signature(job["corrections_json"], include_hash=True),
        "settings": {
            "scale": float(manifest["settings"]["scale"]),
            "gpu_mode": manifest["settings"]["gpu_mode"],
            "batch_size": int(manifest["settings"]["batch_size"]),
            "preview": bool(manifest["settings"]["preview"]),
            "grid_cols": int(manifest["grid_cols"]),
            "grid_rows": int(manifest["grid_rows"]),
            "model_id": SAM2_MODEL_ID,
        },
    })


def validate_sam2_side(job, manifest, side_name, root_summary, setup, corrections):
    side_out = Path(job["sam2_out"]) / side_name
    side_summary_path = side_out / "objectwise_summary.json"
    side_tracks = side_out / "tracks_2d.csv"
    frame_meta_path = side_out / "frames_objectwise_meta.json"
    if not side_summary_path.is_file() or not side_tracks.is_file() or not frame_meta_path.is_file():
        return False
    try:
        side_summary = read_json(side_summary_path)
        frame_meta = read_json(frame_meta_path)
    except Exception:
        return False

    if root_summary.get(side_name) != side_summary:
        return False
    if side_summary.get("failed_batches") or side_summary.get("stopped_batches"):
        return False
    if side_summary.get("side") != side_name:
        return False

    expected_ids = expected_object_ids(manifest)
    object_ids = [int(v) for v in side_summary.get("object_ids", [])]
    if object_ids != expected_ids:
        return False

    frames = int(side_summary.get("frames", -1))
    if frames <= 0:
        return False
    if int(frame_meta.get("video", {}).get("frame_count", -1)) != frames:
        return False
    if not same_path(frame_meta.get("video", {}).get("path"), job[f"{side_name}_video"]):
        return False
    if abs(float(frame_meta.get("scale", -1.0)) - float(manifest["settings"]["scale"])) > 1e-9:
        return False

    expected_total_rows = frames * len(expected_ids)
    if int(side_summary.get("rows", -1)) != expected_total_rows:
        return False
    if count_csv_rows(side_tracks) != expected_total_rows:
        return False

    expected_batches = chunk_ids(expected_ids, int(manifest["settings"]["batch_size"]))
    if [[int(v) for v in batch] for batch in side_summary.get("batches", [])] != expected_batches:
        return False

    result_by_batch = {
        item.get("batch"): item for item in side_summary.get("results", [])
        if isinstance(item, dict)
    }
    for batch_ids in expected_batches:
        batch_name = f"batch_{batch_ids[0]:03d}_{batch_ids[-1]:03d}"
        batch_dir = side_out / "objectwise" / batch_name
        batch_meta_path = batch_dir / "batch_meta.json"
        batch_tracks = batch_dir / "tracks_2d.csv"
        if not batch_meta_path.is_file() or not batch_tracks.is_file():
            return False
        try:
            batch_meta = read_json(batch_meta_path)
        except Exception:
            return False
        expected_rows = frames * len(batch_ids)
        expected_fp = batch_fingerprint(side_name, batch_ids, setup[side_name], manifest, corrections)
        if batch_meta.get("status") != "ok":
            return False
        if batch_meta.get("fingerprint") != expected_fp:
            return False
        if int(batch_meta.get("frames", -1)) != frames:
            return False
        if int(batch_meta.get("rows", -1)) != expected_rows:
            return False
        if int(batch_meta.get("expected_rows", -1)) != expected_rows:
            return False
        if [int(v) for v in batch_meta.get("object_ids", [])] != batch_ids:
            return False
        if count_csv_rows(batch_tracks) != expected_rows:
            return False
        summary_result = result_by_batch.get(batch_name)
        if not summary_result or summary_result.get("status") not in ("ok", "skipped"):
            return False
        if int(summary_result.get("rows", -1)) != expected_rows:
            return False
    return True


def sam2_complete(job, manifest=None):
    if manifest is None:
        return False
    if not setup_complete(job, manifest):
        return False
    summary_path = Path(job["sam2_out"]) / "objectwise_summary.json"
    if not summary_path.is_file():
        return False
    try:
        root_summary = read_json(summary_path)
        setup = load_setup_payload(job)
        corrections = load_corrections_payload(job)
    except Exception:
        return False
    for side in ("left", "right"):
        if not validate_sam2_side(job, manifest, side, root_summary, setup, corrections):
            return False

    expected_fp = sam2_stage_fingerprint(job, manifest)
    meta_path = Path(job["sam2_out"]) / QUEUE_SAM2_META
    if meta_path.is_file():
        try:
            return read_json(meta_path).get("fingerprint") == expected_fp
        except Exception:
            return False
    return True


def write_sam2_queue_meta(job, manifest):
    payload = {
        "version": 1,
        "stage": "sam2",
        "queue_id": manifest.get("queue_id"),
        "job_index": int(job["index"]),
        "velocity": job["velocity"],
        "fingerprint": sam2_stage_fingerprint(job, manifest),
        "written_at": utc_now(),
    }
    atomic_write_json(Path(job["sam2_out"]) / QUEUE_SAM2_META, payload)


def visualization_outputs(result_dir):
    return [
        result_dir / "triangulated_3d_viz_left.mp4",
        result_dir / "triangulated_3d_viz_iso.mp4",
        result_dir / "triangulated_3d_viz_topdown.mp4",
        result_dir / "triangulated_3d_viz_viewer.html",
    ]


def triangulation_data_fingerprint(job, manifest):
    sam2_out = Path(job["sam2_out"])
    return stable_hash({
        "version": 1,
        "stage": "triangulation-data",
        "left_tracks": file_signature(sam2_out / "left" / "tracks_2d.csv", include_hash=True),
        "right_tracks": file_signature(sam2_out / "right" / "tracks_2d.csv", include_hash=True),
        "left_video": video_signature(job["left_video"]),
        "right_video": video_signature(job["right_video"]),
        "stereo": file_signature(REPO_ROOT / "calibration" / "work" / "stereo.npz", include_hash=True),
        "sync_json": file_signature(REPO_ROOT / "calibration" / "work" / "sync.json", include_hash=True),
        "quality_min": 0.0,
        "max_reproj": 20.0,
        "sync_mode": "default",
        "sync_json_arg": "calibration/work/sync.json",
    })


def triangulation_full_fingerprint(job, manifest):
    return stable_hash({
        "version": 1,
        "stage": "triangulation-full",
        "data_fingerprint": triangulation_data_fingerprint(job, manifest),
        "visualize": bool(manifest["settings"]["visualize"]),
        "workers": int(manifest["settings"]["visualization_workers"]),
        "viz_grid_cols": int(manifest["grid_cols"]),
        "viz_grid_rows": int(manifest["grid_rows"]),
        "viz_mode": "scene",
    })


def triangulation_data_reusable(job, manifest):
    result_dir = Path(job["result_dir"])
    out_csv = result_dir / "triangulated_3d.csv"
    out_summary = result_dir / "summary.json"
    meta_path = result_dir / QUEUE_TRIANGULATION_META
    if not out_csv.is_file() or not out_summary.is_file() or not meta_path.is_file():
        return False
    try:
        meta = read_json(meta_path)
    except Exception:
        return False
    return meta.get("data_fingerprint") == triangulation_data_fingerprint(job, manifest)


def triangulation_complete(job, manifest=None):
    if manifest is None:
        return False
    result_dir = Path(job["result_dir"])
    expected = [
        result_dir / "triangulated_3d.csv",
        result_dir / "summary.json",
        *visualization_outputs(result_dir),
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return False
    meta_path = result_dir / QUEUE_TRIANGULATION_META
    if not meta_path.is_file():
        return False
    try:
        meta = read_json(meta_path)
    except Exception:
        return False
    return (
        meta.get("data_fingerprint") == triangulation_data_fingerprint(job, manifest)
        and meta.get("full_fingerprint") == triangulation_full_fingerprint(job, manifest)
    )


def write_triangulation_queue_meta(job, manifest):
    payload = {
        "version": 1,
        "stage": "triangulation",
        "queue_id": manifest.get("queue_id"),
        "job_index": int(job["index"]),
        "velocity": job["velocity"],
        "data_fingerprint": triangulation_data_fingerprint(job, manifest),
        "full_fingerprint": triangulation_full_fingerprint(job, manifest),
        "written_at": utc_now(),
    }
    atomic_write_json(Path(job["result_dir"]) / QUEUE_TRIANGULATION_META, payload)


def create_manifest():
    run_count = prompt_positive_int("How many runs do you want to queue")
    jobs = []
    used_velocities = set()

    for index in range(1, run_count + 1):
        print(f"\nRun {index}/{run_count}")
        left_video = prompt_video_path("  LEFT video path")
        right_video = prompt_video_path("  RIGHT video path")
        while True:
            velocity = input("  Velocity label: ").strip()
            slug = velocity_slug(velocity)
            if not slug:
                print("Velocity must contain at least one usable character.")
                continue
            if slug in used_velocities:
                print(f"Velocity directory '{slug}' is already used in this queue.")
                continue
            result_dir = RESULTS_ROOT / slug
            if result_dir.exists() and any(result_dir.iterdir()):
                print(f"Result directory already exists and is not empty: {result_dir}")
                continue
            used_velocities.add(slug)
            break
        jobs.append(
            {
                "index": index,
                "velocity": velocity,
                "velocity_slug": slug,
                "left_video": str(left_video),
                "right_video": str(right_video),
                "stages": {
                    "setup": {"status": "pending"},
                    "sam2": {"status": "pending"},
                    "triangulation": {"status": "pending"},
                },
            }
        )

    print("\nMarker grid used by every queued run")
    grid_cols = prompt_positive_int("Grid columns")
    grid_rows = prompt_positive_int("Grid rows")
    if grid_cols < 2 or grid_rows < 2:
        raise RuntimeError("Grid columns and rows must both be at least 2.")

    queue_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    queue_dir = QUEUE_ROOT / queue_id
    for job in jobs:
        run_dir = queue_dir / f"run_{job['index']:03d}_{job['velocity_slug']}"
        setup_dir = run_dir / "setup"
        job.update(
            {
                "run_dir": str(run_dir),
                "setup_dir": str(setup_dir),
                "setup_json": str(setup_dir / "prompts" / "points_left_right.json"),
                "corrections_json": str(setup_dir / "prompts" / "corrections.json"),
                "sam2_out": str(run_dir / "sam2"),
                "logs_dir": str(run_dir / "logs"),
                "result_dir": str(RESULTS_ROOT / job["velocity_slug"]),
            }
        )

    manifest = {
        "version": 1,
        "queue_id": queue_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "settings": {
            "scale": 0.25,
            "gpu_mode": "dual",
            "batch_size": 36,
            "preview": False,
            "visualize": True,
            "visualization_workers": 16,
        },
        "jobs": jobs,
    }
    manifest_path = queue_dir / "queue_manifest.json"
    save_manifest(manifest_path, manifest)
    print(f"\n[QUEUE] Manifest: {manifest_path}")
    return manifest_path, manifest


def load_manifest(path_value):
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "queue_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Queue manifest not found: {path}")
    return path, json.loads(path.read_text())


def prepare_setups(manifest_path, manifest):
    print("\n[QUEUE] Complete the semi-auto setup for every run.")
    print("[QUEUE] Processing will begin only after this setup pass finishes.\n")
    for job in manifest["jobs"]:
        stage = job["stages"]["setup"]
        if setup_complete(job, manifest):
            stage.update({"status": "complete", "completed_at": stage.get("completed_at", utc_now())})
            save_manifest(manifest_path, manifest)
            print(f"[SKIP][SETUP] {job['velocity']} already prepared.")
            continue

        Path(job["run_dir"]).mkdir(parents=True, exist_ok=True)
        print(
            f"\n[SETUP] Run {job['index']}/{len(manifest['jobs'])}: "
            f"{job['velocity']}"
        )
        stage.update({"status": "running", "started_at": utc_now(), "error": None})
        save_manifest(manifest_path, manifest)
        command = [
            sys.executable,
            str(MARKERS_SCRIPT),
            "--left-input", job["left_video"],
            "--right-input", job["right_video"],
            "--out", job["setup_dir"],
            "--corrections-json", job["corrections_json"],
            "--semi-auto-setup",
            "--grid-cols", str(manifest["grid_cols"]),
            "--grid-rows", str(manifest["grid_rows"]),
        ]
        return_code = run_logged(
            command,
            SAM2_DIR,
            Path(job["logs_dir"]) / "setup.log",
        )
        if return_code == 0 and setup_complete(job, manifest):
            stage.update({"status": "complete", "completed_at": utc_now(), "error": None})
        else:
            stage.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "error": f"Setup exited with code {return_code}",
                }
            )
        save_manifest(manifest_path, manifest)


def run_objectwise(manifest_path, manifest, job):
    stage = job["stages"]["sam2"]
    if sam2_complete(job, manifest):
        write_sam2_queue_meta(job, manifest)
        stage.update({"status": "complete", "completed_at": stage.get("completed_at", utc_now())})
        save_manifest(manifest_path, manifest)
        print(f"[SKIP][SAM2] {job['velocity']} already complete.")
        return True

    stage.update({"status": "running", "started_at": utc_now(), "error": None})
    save_manifest(manifest_path, manifest)
    command = [
        sys.executable,
        str(OBJECTWISE_SCRIPT),
        "--left-input", job["left_video"],
        "--right-input", job["right_video"],
        "--setup-json", job["setup_json"],
        "--corrections-json", job["corrections_json"],
        "--out", job["sam2_out"],
        "--scale", "0.25",
        "--gpu-mode", "dual",
        "--batch-size", "36",
        "--preview", "false",
    ]
    return_code = run_logged(
        command,
        SAM2_DIR,
        Path(job["logs_dir"]) / "objectwise.log",
    )
    if return_code == 0 and sam2_complete(job, manifest):
        write_sam2_queue_meta(job, manifest)
        stage.update({"status": "complete", "completed_at": utc_now(), "error": None})
        save_manifest(manifest_path, manifest)
        return True

    stage.update(
        {
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"Objectwise tracking incomplete (exit code {return_code})",
        }
    )
    save_manifest(manifest_path, manifest)
    return False


def run_triangulation(manifest_path, manifest, job):
    stage = job["stages"]["triangulation"]
    if triangulation_complete(job, manifest):
        stage.update({"status": "complete", "completed_at": stage.get("completed_at", utc_now())})
        save_manifest(manifest_path, manifest)
        print(f"[SKIP][3D] {job['velocity']} already complete.")
        return True

    result_dir = Path(job["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    stage.update({"status": "running", "started_at": utc_now(), "error": None})
    save_manifest(manifest_path, manifest)
    sam2_out = Path(job["sam2_out"])
    out_csv = result_dir / "triangulated_3d.csv"
    out_summary = result_dir / "summary.json"
    command = [sys.executable, str(TRIANGULATION_SCRIPT)]
    if triangulation_data_reusable(job, manifest):
        print("[3D] Triangulation data fingerprint matches; resuming visualization only.")
        command.append("--viz-only")
    else:
        if out_csv.is_file() or out_summary.is_file():
            print("[3D] Existing triangulation data is missing or has a stale fingerprint; recomputing.")
        command.extend(
            [
                "--left", str(sam2_out / "left" / "tracks_2d.csv"),
                "--right", str(sam2_out / "right" / "tracks_2d.csv"),
                "--right-video", job["right_video"],
            ]
        )
    command.extend([
        "--left-video", job["left_video"],
        "--out-csv", str(out_csv),
        "--out-summary", str(out_summary),
        "--visualize",
        "--viz-out", str(result_dir / "triangulated_3d_viz.mp4"),
        "--workers", "16",
        "--viz-grid-cols", str(manifest["grid_cols"]),
        "--viz-grid-rows", str(manifest["grid_rows"]),
    ])
    return_code = run_logged(
        command,
        REPO_ROOT,
        Path(job["logs_dir"]) / "triangulation.log",
    )
    if return_code == 0:
        write_triangulation_queue_meta(job, manifest)
    if return_code == 0 and triangulation_complete(job, manifest):
        stage.update({"status": "complete", "completed_at": utc_now(), "error": None})
        save_manifest(manifest_path, manifest)
        return True

    stage.update(
        {
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"Triangulation incomplete (exit code {return_code})",
        }
    )
    save_manifest(manifest_path, manifest)
    return False


def process_jobs(manifest_path, manifest):
    print("\n[QUEUE] All available setups are finished. Starting unattended processing.")
    for job in manifest["jobs"]:
        print(
            f"\n[QUEUE] Run {job['index']}/{len(manifest['jobs'])}: "
            f"{job['velocity']}"
        )
        if not setup_complete(job, manifest):
            print("[SKIP] Setup is incomplete; fix it and resume this queue.")
            continue
        if not run_objectwise(manifest_path, manifest, job):
            print("[WARN] SAM2 failed or is incomplete; continuing to the next run.")
            continue
        if not run_triangulation(manifest_path, manifest, job):
            print("[WARN] Triangulation failed or is incomplete; continuing to the next run.")
            continue
        print(f"[OK] Results: {job['result_dir']}")

    complete = all(
        setup_complete(job, manifest) and sam2_complete(job, manifest) and triangulation_complete(job, manifest)
        for job in manifest["jobs"]
    )
    manifest["status"] = "complete" if complete else "incomplete"
    save_manifest(manifest_path, manifest)
    print(f"\n[QUEUE] Status: {manifest['status']}")
    print(f"[QUEUE] Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Queue stereo videos for semi-auto setup, objectwise SAM2, and triangulation."
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume an existing queue_manifest.json or its containing directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume:
        manifest_path, manifest = load_manifest(args.resume)
        print(f"[QUEUE] Resuming: {manifest_path}")
    else:
        manifest_path, manifest = create_manifest()
    prepare_setups(manifest_path, manifest)
    process_jobs(manifest_path, manifest)


if __name__ == "__main__":
    main()
