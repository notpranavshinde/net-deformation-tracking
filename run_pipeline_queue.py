#!/usr/bin/env python3
"""Prepare multiple stereo runs interactively, then process them unattended."""

import argparse
import codecs
import csv
import errno
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
SPLITTER_SCRIPT = REPO_ROOT / "dual_video_splitter_linux.py"
CALIBRATION_SCRIPT = REPO_ROOT / "calibration" / "stereo_checker_debug.py"
QUEUE_ROOT = REPO_ROOT / "work" / "pipeline_queue"
RESULTS_ROOT = REPO_ROOT / "triangulation" / "results"
DEFAULT_SAM2_SCALE = 0.2
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
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


def parse_positive_float(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


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


def restore_terminal_input():
    """Restore normal line input after an OpenCV/Qt interactive child exits."""
    if os.name == "nt" or not sys.stdin.isatty():
        return
    try:
        import termios

        attrs = termios.tcgetattr(sys.stdin.fileno())
        attrs[3] |= termios.ICANON | termios.ECHO
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attrs)
    except (ImportError, OSError):
        pass


def prompt_velocity(index, total, used_velocities):
    while True:
        velocity = input(f"  Clip {index}/{total} velocity label: ").strip()
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
        return velocity, slug


def prompt_setup_mode():
    while True:
        raw = input(
            "SAM2 setup mode: [g]rid rectangular or [s]ectioned local grids [g]: "
        ).strip().lower()
        if raw == "" or raw in {"g", "grid", "rect", "rectangular"}:
            return "grid"
        if raw in {"s", "section", "sections", "sectioned"}:
            return "sectioned"
        print("Enter g for one rectangular grid, or s for sectioned local grids.")


def prompt_section_layout():
    section_count = prompt_positive_int("Number of sections per experiment clip")
    sections = []
    for idx in range(1, section_count + 1):
        print(f"Section {idx}/{section_count}")
        cols = prompt_positive_int("  columns")
        rows = prompt_positive_int("  rows")
        if cols < 2 or rows < 2:
            raise RuntimeError("Section columns and rows must both be at least 2.")
        sections.append({"index": idx, "cols": cols, "rows": rows})
    return sections


def section_layout_arg(section_layout):
    return ",".join(f"{int(item['cols'])}x{int(item['rows'])}" for item in section_layout or [])


def derive_visualization_grid_from_sections(section_layout):
    if not section_layout:
        raise RuntimeError("Cannot derive visualization grid without at least one section.")
    cols = max(int(item["cols"]) for item in section_layout)
    rows = sum(int(item["rows"]) for item in section_layout)
    if cols < 2 or rows < 2:
        raise RuntimeError("Derived visualization grid must have columns and rows >= 2.")
    return cols, rows


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


def frame_range_signature(start_frame, end_frame):
    start = int(start_frame)
    end = int(end_frame)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid frame range [{start}, {end})")
    return {
        "start_frame": start,
        "end_frame": end,
        "frame_count": end - start,
    }


def frame_range_meta_matches(actual, expected):
    if not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        try:
            if int(actual.get(key, -1)) != int(value):
                return False
        except (TypeError, ValueError):
            return False
    return True


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


def track_grid_is_complete(path, frame_count, object_ids):
    expected_ids = {int(obj_id) for obj_id in object_ids}
    expected_rows = int(frame_count) * len(expected_ids)
    seen = set()
    try:
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                frame = int(row["frame"])
                obj_id = int(row["obj_id"])
                key = (frame, obj_id)
                if (
                    frame < 0
                    or frame >= int(frame_count)
                    or obj_id not in expected_ids
                    or key in seen
                ):
                    return False
                seen.add(key)
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return len(seen) == expected_rows


def save_manifest(manifest_path, manifest):
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def run_logged(command, cwd, log_path, use_pty=True):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$", " ".join(str(part) for part in command))
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] $ {' '.join(str(part) for part in command)}\n")
        popen_kwargs = {}
        master_fd = None
        slave_fd = None
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
            if use_pty:
                import pty

                master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd),
            stdout=slave_fd if slave_fd is not None else subprocess.PIPE,
            stderr=slave_fd if slave_fd is not None else subprocess.STDOUT,
            text=True,
            bufsize=1,
            **popen_kwargs,
        )
        if slave_fd is not None:
            os.close(slave_fd)
            slave_fd = None
        try:
            if master_fd is not None:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    try:
                        chunk = os.read(master_fd, 65536)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            break
                        raise
                    if not chunk:
                        break
                    output = decoder.decode(chunk)
                    if output:
                        sys.stdout.write(output)
                        sys.stdout.flush()
                        log.write(output)
                output = decoder.decode(b"", final=True)
                if output:
                    sys.stdout.write(output)
                    sys.stdout.flush()
                    log.write(output)
            else:
                assert process.stdout is not None
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
        finally:
            if master_fd is not None:
                os.close(master_fd)
            if slave_fd is not None:
                os.close(slave_fd)
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
    if len(left_points) <= 0 or len(left_points) != len(right_points):
        return False
    expected = None
    if manifest is not None and manifest.get("setup_mode", "grid") != "sectioned":
        expected = int(manifest["grid_cols"]) * int(manifest["grid_rows"])
    if expected is not None and (len(left_points) != expected or len(right_points) != expected):
        return False
    if int(setup_meta.get("left_count", -1)) != len(left_points):
        return False
    if int(setup_meta.get("right_count", -1)) != len(right_points):
        return False
    expected_range = frame_range_signature(job["start_frame"], job["end_frame"])
    recorded_range = setup_meta.get("frame_range", {})
    legacy_full_video = bool(
        manifest
        and manifest.get("preprocessing", {}).get("mode") == "legacy_existing_clips"
        and not recorded_range
        and expected_range["start_frame"] == 0
    )
    if not legacy_full_video:
        if (
            int(recorded_range.get("start_frame", -1)) != expected_range["start_frame"]
            or int(recorded_range.get("end_frame", -1)) != expected_range["end_frame"]
        ):
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
            "start_frame": int(job["start_frame"]),
            "end_frame": int(job["end_frame"]),
        },
        "right": {
            "points": payload["right"],
            "crop": side_crop("right"),
            "video": job["right_video"],
            "start_frame": int(job["start_frame"]),
            "end_frame": int(job["end_frame"]),
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


def expected_object_ids(manifest, job=None):
    total = None
    if manifest.get("setup_mode", "grid") == "sectioned" and job is not None:
        try:
            points_payload = read_json(job["setup_json"])
            left_count = len(points_payload.get("left", []))
            right_count = len(points_payload.get("right", []))
            if left_count != right_count or left_count <= 0:
                return []
            total = left_count
        except Exception:
            return []
    if total is None:
        total = int(manifest["grid_cols"]) * int(manifest["grid_rows"])
    return list(range(total))


def sam2_model_id(manifest):
    return manifest.get("settings", {}).get("model_id", SAM2_MODEL_ID)


def sam2_frame_step(manifest):
    return int(manifest.get("settings", {}).get("frame_step", 1))


def batch_fingerprint(side_name, batch_ids, setup_side, manifest, corrections):
    wanted = {int(obj_id) for obj_id in batch_ids}
    batch_corrections = [
        corr for corr in corrections_for_side(corrections, side_name)
        if int(corr.get("obj_id", -1)) in wanted
    ]
    return stable_hash({
        "version": 3,
        "side": side_name,
        "frame_cache_image_format": "jpg",
        "object_ids": [int(obj_id) for obj_id in batch_ids],
        "video": objectwise_video_signature(setup_side["video"]),
        "frame_range": {
            "start_frame": int(setup_side["start_frame"]),
            "end_frame": int(setup_side["end_frame"]),
            "frame_step": sam2_frame_step(manifest),
        },
        "crop": list(setup_side["crop"]) if setup_side["crop"] is not None else None,
        "scale": float(manifest["settings"]["scale"]),
        "points": [setup_side["points"][obj_id] for obj_id in batch_ids],
        "corrections": batch_corrections,
        "model_id": sam2_model_id(manifest),
        "batch_size": int(manifest["settings"]["batch_size"]),
    })


def sam2_stage_fingerprint(job, manifest):
    return stable_hash({
        "version": 2,
        "stage": "sam2",
        "frame_cache_image_format": "jpg",
        "setup_mode": manifest.get("setup_mode", "grid"),
        "left_video": video_signature(job["left_video"]),
        "right_video": video_signature(job["right_video"]),
        "frame_range": frame_range_signature(job["start_frame"], job["end_frame"]),
        "setup_json": file_signature(job["setup_json"], include_hash=True),
        "corrections_json": file_signature(job["corrections_json"], include_hash=True),
        "settings": {
            "scale": float(manifest["settings"]["scale"]),
            "gpu_mode": manifest["settings"]["gpu_mode"],
            "batch_size": int(manifest["settings"]["batch_size"]),
            "preview": bool(manifest["settings"]["preview"]),
            "grid_cols": int(manifest["grid_cols"]),
            "grid_rows": int(manifest["grid_rows"]),
            "model_id": sam2_model_id(manifest),
            "frame_step": sam2_frame_step(manifest),
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

    expected_ids = expected_object_ids(manifest, job)
    if not expected_ids:
        return False
    object_ids = [int(v) for v in side_summary.get("object_ids", [])]
    if object_ids != expected_ids:
        return False

    frames = int(side_summary.get("frames", -1))
    expected_range = frame_range_signature(job["start_frame"], job["end_frame"])
    if frames != expected_range["frame_count"]:
        return False
    if not frame_range_meta_matches(frame_meta.get("frame_range"), expected_range):
        return False
    if not same_path(frame_meta.get("video", {}).get("path"), job[f"{side_name}_video"]):
        return False
    if abs(float(frame_meta.get("scale", -1.0)) - float(manifest["settings"]["scale"])) > 1e-9:
        return False
    if frame_meta.get("image_format") != "jpg":
        return False

    expected_total_rows = frames * len(expected_ids)
    if int(side_summary.get("rows", -1)) != expected_total_rows:
        return False
    if not track_grid_is_complete(side_tracks, frames, expected_ids):
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
        if not track_grid_is_complete(batch_tracks, frames, batch_ids):
            return False
        summary_result = result_by_batch.get(batch_name)
        if not summary_result or summary_result.get("status") not in ("ok", "skipped"):
            return False
        if int(summary_result.get("rows", -1)) != expected_rows:
            return False
    return True


def sam2_outputs_valid(job, manifest):
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
    return True


def sam2_complete(job, manifest=None):
    if manifest is None:
        return False
    if not sam2_outputs_valid(job, manifest):
        return False
    expected_fp = sam2_stage_fingerprint(job, manifest)
    meta_path = Path(job["sam2_out"]) / QUEUE_SAM2_META
    if meta_path.is_file():
        try:
            if read_json(meta_path).get("fingerprint") == expected_fp:
                return True
        except Exception:
            return False
    write_sam2_queue_meta(job, manifest)
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
        result_dir / "triangulated_3d_viz_viewer.html",
    ]


def triangulation_data_fingerprint(job, manifest):
    sam2_out = Path(job["sam2_out"])
    sync_json, stereo_npz = calibration_paths(manifest)
    return stable_hash({
        "version": 1,
        "stage": "triangulation-data",
        "left_tracks": file_signature(sam2_out / "left" / "tracks_2d.csv", include_hash=True),
        "right_tracks": file_signature(sam2_out / "right" / "tracks_2d.csv", include_hash=True),
        "left_video": video_signature(job["left_video"]),
        "right_video": video_signature(job["right_video"]),
        "frame_range": frame_range_signature(job["start_frame"], job["end_frame"]),
        "stereo": file_signature(stereo_npz, include_hash=True),
        "sync_json": file_signature(sync_json, include_hash=True),
        "quality_min": 0.0,
        "max_reproj": 20.0,
        "sync_mode": "default",
        "sync_json_arg": str(sync_json),
    })


def triangulation_full_fingerprint(job, manifest):
    return stable_hash({
        "version": 1,
        "stage": "triangulation-full",
        "data_fingerprint": triangulation_data_fingerprint(job, manifest),
        "visualize": bool(manifest["settings"]["visualize"]),
        "viz_grid_cols": int(manifest["grid_cols"]),
        "viz_grid_rows": int(manifest["grid_rows"]),
        "viz_mode": "viewer-only",
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


def _new_job(
    queue_dir,
    index,
    velocity,
    slug,
    left_video,
    right_video,
    split_index,
    start_frame,
    end_frame,
):
    run_dir = queue_dir / f"run_{index:03d}_{slug}"
    setup_dir = run_dir / "setup"
    return {
        "index": int(index),
        "split_clip_index": int(split_index),
        "velocity": velocity,
        "velocity_slug": slug,
        "left_video": str(Path(left_video).resolve()),
        "right_video": str(Path(right_video).resolve()),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "run_dir": str(run_dir),
        "setup_dir": str(setup_dir),
        "setup_json": str(setup_dir / "prompts" / "points_left_right.json"),
        "corrections_json": str(setup_dir / "prompts" / "corrections.json"),
        "sam2_out": str(run_dir / "sam2"),
        "logs_dir": str(run_dir / "logs"),
        "result_dir": str(RESULTS_ROOT / slug),
        "stages": {
            "setup": {"status": "pending"},
            "sam2": {"status": "pending"},
            "triangulation": {"status": "pending"},
        },
    }


def migrate_manifest(manifest):
    changed = False
    settings = manifest.setdefault("settings", {})
    if "scale" not in settings:
        settings["scale"] = DEFAULT_SAM2_SCALE
        changed = True
    if "model_id" not in settings:
        settings["model_id"] = SAM2_MODEL_ID
        changed = True
    if "frame_step" not in settings:
        settings["frame_step"] = 1
        changed = True
    if "preprocessing" not in manifest:
        manifest["preprocessing"] = {
            "mode": "legacy_existing_clips",
            "calibration": {
                "mode": "existing",
                "sync_json": str(REPO_ROOT / "calibration" / "work" / "sync.json"),
                "stereo_npz": str(REPO_ROOT / "calibration" / "work" / "stereo.npz"),
            },
        }
        changed = True
    if int(manifest.get("version", 1)) < 2:
        manifest["version"] = 2
        changed = True
    for job in manifest.get("jobs", []):
        if "start_frame" not in job or "end_frame" not in job:
            counts = [
                int(video_signature(job[side])["frame_count"])
                for side in ("left_video", "right_video")
            ]
            job["start_frame"] = 0
            job["end_frame"] = min(count for count in counts if count > 0)
            changed = True
    return changed


def apply_cli_overrides(manifest, args):
    changed = False
    if args.sam2_scale is not None:
        current = float(manifest.setdefault("settings", {}).get("scale", DEFAULT_SAM2_SCALE))
        if abs(current - float(args.sam2_scale)) > 1e-12:
            manifest["settings"]["scale"] = float(args.sam2_scale)
            changed = True
    if args.sam2_model_id is not None:
        current = str(manifest.setdefault("settings", {}).get("model_id", SAM2_MODEL_ID))
        if current != str(args.sam2_model_id):
            manifest["settings"]["model_id"] = str(args.sam2_model_id)
            changed = True
    return changed


def print_queue_settings(manifest):
    settings = manifest.get("settings", {})
    print(
        "[QUEUE] SAM2 settings: "
        f"scale={float(settings.get('scale', DEFAULT_SAM2_SCALE))}, "
        f"gpu_mode={settings.get('gpu_mode', 'dual')}, "
        f"batch_size={int(settings.get('batch_size', 36))}, "
        f"model_id={sam2_model_id(manifest)}"
    )


def ensure_section_layout(manifest_path, manifest):
    if manifest.get("setup_mode", "grid") != "sectioned":
        return
    layout = manifest.get("section_layout")
    if isinstance(layout, list) and layout:
        return
    print("\n[QUEUE] Sectioned setup layout is missing from this queue.")
    print("[QUEUE] Enter it once; the same section count and dimensions will be reused for every experiment clip.")
    manifest["section_layout"] = prompt_section_layout()
    save_manifest(manifest_path, manifest)


def splitter_complete(manifest):
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("mode") == "legacy_existing_clips":
        return True
    splitter = preprocessing.get("splitter", {})
    path = Path(splitter.get("manifest", ""))
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    if payload.get("status") != "complete":
        return False
    if payload.get("mode") != "virtual":
        return False
    if not same_path(payload.get("output_root"), splitter.get("output_root")):
        return False
    for side in ("left", "right"):
        source_path = preprocessing.get(f"raw_{side}")
        recorded = payload.get(f"{side}_source", {})
        current = file_signature(source_path)
        if (
            not same_path(recorded.get("path"), source_path)
            or int(recorded.get("size_bytes", -1)) != int(current.get("size_bytes", -2))
            or int(recorded.get("mtime_ns", -1)) != int(current.get("mtime_ns", -2))
        ):
            return False
    clips = payload.get("clips", [])
    indices = [int(clip.get("index", -1)) for clip in clips]
    return indices == list(range(1, len(clips) + 1)) and len(clips) >= 2 and all(
        Path(clip.get(side, "")).is_file() and Path(clip[side]).stat().st_size > 0
        for clip in clips
        for side in ("left_video", "right_video")
    )


def prepare_splitter(manifest_path, manifest):
    preprocessing = manifest["preprocessing"]
    if preprocessing.get("mode") == "legacy_existing_clips":
        return True
    splitter = preprocessing["splitter"]
    stage = splitter["stage"]
    if splitter_complete(manifest):
        stage.update({"status": "complete", "completed_at": stage.get("completed_at", utc_now())})
        save_manifest(manifest_path, manifest)
        print("[SKIP][SPLIT] Virtual frame ranges already complete.")
        return True

    stage.update({"status": "running", "started_at": utc_now(), "error": None})
    save_manifest(manifest_path, manifest)
    command = [
        sys.executable,
        str(SPLITTER_SCRIPT),
        "--left", preprocessing["raw_left"],
        "--right", preprocessing["raw_right"],
        "--output-root", splitter["output_root"],
        "--manifest", splitter["manifest"],
        "--select-only",
    ]
    return_code = run_logged(
        command,
        REPO_ROOT,
        Path(manifest_path).parent / "split" / "splitter.log",
    )
    if return_code == 0 and splitter_complete(manifest):
        stage.update({"status": "complete", "completed_at": utc_now(), "error": None})
        save_manifest(manifest_path, manifest)
        return True
    stage.update({
        "status": "failed",
        "failed_at": utc_now(),
        "error": f"Splitter incomplete (exit code {return_code})",
    })
    save_manifest(manifest_path, manifest)
    return False


def ensure_jobs_from_split(manifest_path, manifest):
    preprocessing = manifest["preprocessing"]
    if preprocessing.get("mode") == "legacy_existing_clips":
        return True
    split_payload = read_json(preprocessing["splitter"]["manifest"])
    clips = sorted(split_payload.get("clips", []), key=lambda clip: int(clip["index"]))
    if [int(clip["index"]) for clip in clips] != list(range(1, len(clips) + 1)):
        raise RuntimeError("Split manifest clip indices are not contiguous from 1.")
    if len(clips) < 2:
        raise RuntimeError(
            "At least two clip pairs are required: clip 1 for calibration and one experiment clip."
        )

    calibration_clip = clips[0]
    calibration = preprocessing["calibration"]
    calibration["left_video"] = calibration_clip["left_video"]
    calibration["right_video"] = calibration_clip["right_video"]
    calibration["split_clip_index"] = int(calibration_clip["index"])
    calibration["start_frame"] = int(calibration_clip["start_frame"])
    calibration["end_frame"] = int(calibration_clip["end_frame"])

    experiment_clips = clips[1:]
    existing_jobs = manifest.get("jobs", [])
    if len(existing_jobs) > len(experiment_clips):
        raise RuntimeError("Split clip count is smaller than the labeled queue job count.")
    if existing_jobs:
        for job, clip in zip(existing_jobs, experiment_clips):
            if not same_path(job["left_video"], clip["left_video"]) or not same_path(
                job["right_video"], clip["right_video"]
            ):
                raise RuntimeError("Split clip paths changed after queue jobs were labeled.")
            if frame_range_signature(job["start_frame"], job["end_frame"]) != frame_range_signature(
                clip["start_frame"], clip["end_frame"]
            ):
                raise RuntimeError("Split frame ranges changed after queue jobs were labeled.")
        if len(existing_jobs) == len(experiment_clips):
            save_manifest(manifest_path, manifest)
            return True

    if not existing_jobs:
        print("\n[QUEUE] Clip pair 1 is reserved for calibration.")
        print(f"[CALIBRATION] LEFT:  {calibration['left_video']}")
        print(f"[CALIBRATION] RIGHT: {calibration['right_video']}")
        print("\nEnter velocity labels for the remaining experiment clips.")
    else:
        print(
            f"\n[QUEUE] Resuming velocity labels at clip "
            f"{len(existing_jobs) + 1}/{len(experiment_clips)}."
        )
    restore_terminal_input()

    queue_dir = Path(manifest_path).parent
    used_velocities = {job["velocity_slug"] for job in existing_jobs}
    total = len(experiment_clips)
    for index in range(len(existing_jobs) + 1, total + 1):
        clip = experiment_clips[index - 1]
        velocity, slug = prompt_velocity(index, total, used_velocities)
        manifest.setdefault("jobs", []).append(
            _new_job(
                queue_dir,
                index,
                velocity,
                slug,
                clip["left_video"],
                clip["right_video"],
                clip["index"],
                clip["start_frame"],
                clip["end_frame"],
            )
        )
        save_manifest(manifest_path, manifest)
    return True


def create_manifest(args):
    print("Enter the raw stereo videos that contain calibration first, then experiment runs.")
    left_video = prompt_video_path("Raw LEFT video path")
    right_video = prompt_video_path("Raw RIGHT video path")

    print("\nSAM2 marker setup")
    setup_mode = prompt_setup_mode()
    section_layout = None
    if setup_mode == "sectioned":
        print("\nSection layout used by every experiment clip")
        section_layout = prompt_section_layout()
        grid_cols, grid_rows = derive_visualization_grid_from_sections(section_layout)
        print(
            f"[INFO] Derived full visualization grid as {grid_cols} columns x {grid_rows} rows "
            "from stacked sections."
        )
    else:
        print("Marker grid used by every experiment run")
        grid_cols = prompt_positive_int("Grid columns")
        grid_rows = prompt_positive_int("Grid rows")
    if grid_cols < 2 or grid_rows < 2:
        raise RuntimeError("Grid columns and rows must both be at least 2.")

    queue_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    queue_dir = QUEUE_ROOT / queue_id
    split_dir = queue_dir / "split"
    calibration_dir = queue_dir / "calibration"
    manifest = {
        "version": 2,
        "queue_id": queue_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "setup_mode": setup_mode,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "section_layout": section_layout,
        "preprocessing": {
            "mode": "split_raw_pair",
            "raw_left": str(left_video),
            "raw_right": str(right_video),
            "splitter": {
                "output_root": str(split_dir / "clips"),
                "manifest": str(split_dir / "split_manifest.json"),
                "stage": {"status": "pending"},
            },
            "calibration": {
                "mode": "generated",
                "work_dir": str(calibration_dir),
                "sync_json": str(calibration_dir / "sync.json"),
                "stats_dir": str(calibration_dir / "stats"),
                "mono_npz": str(calibration_dir / "mono.npz"),
                "stereo_npz": str(calibration_dir / "stereo.npz"),
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
                    "sync": {"status": "pending"},
                    "stats": {"status": "pending"},
                    "mono": {"status": "pending"},
                    "stereo": {"status": "pending"},
                },
            },
        },
        "settings": {
            "scale": float(args.sam2_scale if args.sam2_scale is not None else DEFAULT_SAM2_SCALE),
            "gpu_mode": "dual",
            "batch_size": 36,
            "preview": False,
            "visualize": True,
            "model_id": str(args.sam2_model_id if args.sam2_model_id is not None else SAM2_MODEL_ID),
        },
        "jobs": [],
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


def calibration_paths(manifest):
    calibration = manifest["preprocessing"]["calibration"]
    return Path(calibration["sync_json"]), Path(calibration["stereo_npz"])


def calibration_stage_outputs(calibration, stage_name):
    if stage_name == "sync":
        return [Path(calibration["sync_json"])]
    if stage_name == "stats":
        stats_dir = Path(calibration["stats_dir"])
        return [stats_dir / "stats_left.json", stats_dir / "stats_right.json"]
    if stage_name == "mono":
        mono = Path(calibration["mono_npz"])
        return [mono, mono.with_name(mono.stem + "_report.json")]
    if stage_name == "stereo":
        stereo = Path(calibration["stereo_npz"])
        return [stereo, stereo.with_name(stereo.stem + "_report.json")]
    raise ValueError(f"Unknown calibration stage: {stage_name}")


def calibration_stage_fingerprint(manifest, stage_name):
    calibration = manifest["preprocessing"]["calibration"]
    payload = {
        "version": 1,
        "stage": stage_name,
        "left_video": video_signature(calibration["left_video"]),
        "right_video": video_signature(calibration["right_video"]),
        "frame_range": frame_range_signature(
            calibration["start_frame"], calibration["end_frame"]
        ),
        "sync_algorithm": "audio-frame-correlation-v1",
        "settings": calibration["settings"],
    }
    dependencies = {
        "sync": [],
        "stats": ["sync"],
        "mono": ["sync", "stats"],
        "stereo": ["sync", "stats", "mono"],
    }[stage_name]
    payload["dependencies"] = {
        dependency: [
            file_signature(path, include_hash=True)
            for path in calibration_stage_outputs(calibration, dependency)
        ]
        for dependency in dependencies
    }
    return stable_hash(payload)


def calibration_stage_complete(manifest, stage_name):
    calibration = manifest["preprocessing"]["calibration"]
    if calibration.get("mode") == "existing":
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in calibration_stage_outputs(calibration, stage_name)
        )
    stage = calibration["stages"][stage_name]
    outputs = calibration_stage_outputs(calibration, stage_name)
    return (
        stage.get("status") == "complete"
        and stage.get("fingerprint") == calibration_stage_fingerprint(manifest, stage_name)
        and all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    )


def calibration_complete(manifest):
    calibration = manifest["preprocessing"]["calibration"]
    if calibration.get("mode") == "existing":
        sync_json, stereo_npz = calibration_paths(manifest)
        return sync_json.is_file() and stereo_npz.is_file()
    return all(
        calibration_stage_complete(manifest, stage_name)
        for stage_name in ("sync", "stats", "mono", "stereo")
    )


def calibration_command(calibration, stage_name):
    settings = calibration["settings"]
    common = [
        "--left", calibration["left_video"],
        "--right", calibration["right_video"],
        "--start-frame", str(calibration["start_frame"]),
        "--end-frame", str(calibration["end_frame"]),
    ]
    board = [
        "--cols", str(settings["cols"]),
        "--rows", str(settings["rows"]),
        "--square-mm", str(settings["square_mm"]),
        "--scale", str(settings["scale"]),
    ]
    workers = ["--workers", str(settings["workers"])]
    command = [sys.executable, str(CALIBRATION_SCRIPT), stage_name, *common]
    if stage_name == "sync":
        command.extend([
            "--out", calibration["sync_json"],
            "--scale", str(settings["scale"]),
            "--sync-mode", settings["sync_mode"],
        ])
    elif stage_name == "stats":
        command.extend([
            "--sync", calibration["sync_json"],
            "--out", calibration["stats_dir"],
            *board,
            "--step", str(settings["stats_step"]),
            "--max-scan", str(settings["stats_max_scan"]),
            *workers,
            "--no-adaptive",
        ])
    elif stage_name == "mono":
        command.extend([
            "--sync", calibration["sync_json"],
            "--out", calibration["mono_npz"],
            *board,
            "--max-scan", str(settings["mono_max_scan"]),
            *workers,
            "--reuse-stats-indices", "true",
            "--stats-dir", calibration["stats_dir"],
        ])
    elif stage_name == "stereo":
        command.extend([
            "--sync", calibration["sync_json"],
            "--mono", calibration["mono_npz"],
            "--out", calibration["stereo_npz"],
            *board,
            "--max-pairs", str(settings["stereo_max_pairs"]),
            *workers,
            "--reuse-stats-indices", "true",
            "--stats-dir", calibration["stats_dir"],
        ])
    else:
        raise ValueError(f"Unknown calibration stage: {stage_name}")
    return command


def prepare_calibration(manifest_path, manifest):
    calibration = manifest["preprocessing"]["calibration"]
    if calibration.get("mode") == "existing":
        if not calibration_complete(manifest):
            sync_json, stereo_npz = calibration_paths(manifest)
            raise RuntimeError(
                f"Existing calibration is incomplete: sync={sync_json}, stereo={stereo_npz}"
            )
        print("[SKIP][CALIBRATION] Using calibration files recorded by this legacy queue.")
        return True

    print("\n[CALIBRATION] Clip pair 1: sync -> stats -> mono -> stereo")
    logs_dir = Path(calibration["work_dir"]) / "logs"
    for stage_name in ("sync", "stats", "mono", "stereo"):
        stage = calibration["stages"][stage_name]
        if calibration_stage_complete(manifest, stage_name):
            print(f"[SKIP][CALIBRATION:{stage_name.upper()}] Already complete.")
            continue
        stage.update({"status": "running", "started_at": utc_now(), "error": None})
        save_manifest(manifest_path, manifest)
        return_code = run_logged(
            calibration_command(calibration, stage_name),
            REPO_ROOT / "calibration",
            logs_dir / f"{stage_name}.log",
        )
        outputs = calibration_stage_outputs(calibration, stage_name)
        if return_code == 0 and all(
            path.is_file() and path.stat().st_size > 0 for path in outputs
        ):
            stage.update({
                "status": "complete",
                "completed_at": utc_now(),
                "error": None,
                "fingerprint": calibration_stage_fingerprint(manifest, stage_name),
            })
            save_manifest(manifest_path, manifest)
            continue
        stage.update({
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"Calibration {stage_name} incomplete (exit code {return_code})",
        })
        save_manifest(manifest_path, manifest)
        return False
    return calibration_complete(manifest)


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
        setup_flag = (
            "--semi-auto-setup-sections"
            if manifest.get("setup_mode", "grid") == "sectioned"
            else "--semi-auto-setup"
        )
        command = [
            sys.executable,
            str(MARKERS_SCRIPT),
            "--left-input", job["left_video"],
            "--right-input", job["right_video"],
            "--start-frame", str(job["start_frame"]),
            "--end-frame", str(job["end_frame"]),
            "--out", job["setup_dir"],
            "--corrections-json", job["corrections_json"],
            setup_flag,
            "--grid-cols", str(manifest["grid_cols"]),
            "--grid-rows", str(manifest["grid_rows"]),
        ]
        if manifest.get("setup_mode", "grid") == "sectioned":
            layout_arg = section_layout_arg(manifest.get("section_layout"))
            if not layout_arg:
                raise RuntimeError("Sectioned setup requires section_layout in the queue manifest.")
            command.extend(["--section-layout", layout_arg])
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
        "--start-frame", str(job["start_frame"]),
        "--end-frame", str(job["end_frame"]),
        "--setup-json", job["setup_json"],
        "--corrections-json", job["corrections_json"],
        "--out", job["sam2_out"],
        "--scale", str(manifest["settings"]["scale"]),
        "--model-id", str(sam2_model_id(manifest)),
        "--gpu-mode", str(manifest["settings"]["gpu_mode"]),
        "--batch-size", str(manifest["settings"]["batch_size"]),
        "--preview", str(bool(manifest["settings"]["preview"])).lower(),
    ]
    return_code = run_logged(
        command,
        SAM2_DIR,
        Path(job["logs_dir"]) / "objectwise.log",
    )
    if return_code == 0:
        write_sam2_queue_meta(job, manifest)
        if sam2_complete(job, manifest):
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
    sync_json, stereo_npz = calibration_paths(manifest)
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
        "--stereo", str(stereo_npz),
        "--sync-json", str(sync_json),
        "--left-video", job["left_video"],
        "--start-frame", str(job["start_frame"]),
        "--end-frame", str(job["end_frame"]),
        "--out-csv", str(out_csv),
        "--out-summary", str(out_summary),
        "--visualize",
        "--viewer-only",
        "--viz-out", str(result_dir / "triangulated_3d_viz.mp4"),
        "--viz-grid-cols", str(manifest["grid_cols"]),
        "--viz-grid-rows", str(manifest["grid_rows"]),
    ])
    return_code = run_logged(
        command,
        REPO_ROOT,
        Path(job["logs_dir"]) / "triangulation.log",
        use_pty=False,
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

    complete = calibration_complete(manifest) and bool(manifest["jobs"]) and all(
        setup_complete(job, manifest) and sam2_complete(job, manifest) and triangulation_complete(job, manifest)
        for job in manifest["jobs"]
    )
    manifest["status"] = "complete" if complete else "incomplete"
    save_manifest(manifest_path, manifest)
    print(f"\n[QUEUE] Status: {manifest['status']}")
    print(f"[QUEUE] Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select virtual ranges from one raw stereo pair, calibrate from range 1, "
            "then queue the remaining ranges for SAM2 tracking and triangulation."
        )
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume an existing queue_manifest.json or its containing directory.",
    )
    parser.add_argument(
        "--sam2-scale",
        type=parse_positive_float,
        default=None,
        help=(
            "SAM2 processing scale for a new queue. Default is 0.2. On "
            "--resume, passing this flag updates the queue scale and "
            "invalidates stale SAM2/3D outputs."
        ),
    )
    parser.add_argument(
        "--sam2-model-id",
        default=None,
        help="Hugging Face SAM2/SAM2.1 model id used by queued objectwise runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume:
        manifest_path, manifest = load_manifest(args.resume)
        changed = migrate_manifest(manifest)
        changed = apply_cli_overrides(manifest, args) or changed
        if changed:
            save_manifest(manifest_path, manifest)
        print(f"[QUEUE] Resuming: {manifest_path}")
    else:
        manifest_path, manifest = create_manifest(args)
    print_queue_settings(manifest)
    if not prepare_splitter(manifest_path, manifest):
        print("[QUEUE] Splitter is incomplete. Resume this manifest to try again.")
        return
    ensure_jobs_from_split(manifest_path, manifest)
    if not prepare_calibration(manifest_path, manifest):
        print("[QUEUE] Calibration is incomplete. Resume this manifest to try again.")
        return
    ensure_section_layout(manifest_path, manifest)
    prepare_setups(manifest_path, manifest)
    process_jobs(manifest_path, manifest)


if __name__ == "__main__":
    main()
