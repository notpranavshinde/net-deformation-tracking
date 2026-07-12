#!/usr/bin/env python3
"""Prepare portable stereo queue setup on one machine and processing on another."""

import argparse
import codecs
import copy
import csv
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
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
MACHINE_PATHS_FILE = REPO_ROOT / "machine_paths.json"
QUEUE_PATH_PREFIX = "queue:"
REPO_PATH_PREFIX = "repo:"


SETUP_REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "torch": "torch",
    "rich": "rich",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "PIL": "pillow",
    "hydra": "hydra-core",
    "iopath": "iopath",
}


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


def normalize_pasted_path(value):
    """Accept plain paths plus paths copied from a PowerShell invocation."""
    text = str(value).strip()
    if text.startswith("&"):
        text = text[1:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def prompt_video_path(label):
    while True:
        raw = normalize_pasted_path(input(f"{label}: "))
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


def format_command(command, windows=None):
    parts = [str(part) for part in command]
    use_windows = os.name == "nt" if windows is None else bool(windows)
    return subprocess.list2cmdline(parts) if use_windows else shlex.join(parts)


def setup_rerun_command(args, resume_path=None):
    command = ["python", str(Path(__file__).resolve()), "--setup-only"]
    if resume_path is not None:
        command.extend(["--resume", str(Path(resume_path))])
    elif getattr(args, "resume", None):
        command.extend(["--resume", str(args.resume)])
    if getattr(args, "sam2_scale", None) is not None:
        command.extend(["--sam2-scale", str(args.sam2_scale)])
    if getattr(args, "sam2_model_id", None) is not None:
        command.extend(["--sam2-model-id", str(args.sam2_model_id)])
    return format_command(command)


def collect_windows_setup_preflight_issues(
    python_version=None,
    module_finder=None,
    ui_framework_probe=None,
    executable_finder=None,
):
    """Return blocking issues and non-blocking warnings for native Windows setup."""
    if python_version is None:
        python_version = tuple(sys.version_info[:2])
    if module_finder is None:
        import importlib.util

        module_finder = importlib.util.find_spec
    if executable_finder is None:
        executable_finder = shutil.which

    issues = []
    warnings = []
    if tuple(python_version) != (3, 11):
        issues.append(
            f"Python 3.11 is required; this interpreter is "
            f"{python_version[0]}.{python_version[1]} ({sys.executable})."
        )

    missing_packages = []
    for module_name, package_name in SETUP_REQUIRED_MODULES.items():
        try:
            found = module_finder(module_name)
        except (ImportError, AttributeError, ValueError):
            found = None
        if found is None:
            missing_packages.append(package_name)
    if missing_packages:
        issues.append("Missing setup packages: " + ", ".join(sorted(set(missing_packages))))

    missing_scripts = [
        path for path in (SPLITTER_SCRIPT, CALIBRATION_SCRIPT, MARKERS_SCRIPT) if not path.is_file()
    ]
    if missing_scripts:
        issues.append("Missing setup scripts: " + ", ".join(str(path) for path in missing_scripts))

    if "opencv-python" not in missing_packages:
        try:
            if ui_framework_probe is None:
                import cv2

                framework = cv2.currentUIFramework() if hasattr(cv2, "currentUIFramework") else "unknown"
            else:
                framework = ui_framework_probe()
            if not framework or str(framework).lower() in {"none", "unknown"}:
                issues.append(
                    "OpenCV has no interactive UI backend; install opencv-python, not opencv-python-headless."
                )
        except Exception as exc:
            issues.append(f"Could not validate the OpenCV UI backend: {exc}")

    if executable_finder("ffprobe") is None:
        warnings.append("ffprobe was not found; splitter IMU-event navigation will be unavailable.")
    return issues, warnings


def validate_windows_setup_environment(args):
    if os.name != "nt":
        return True
    print("[PREFLIGHT] Checking the Windows --setup-only environment...")
    issues, warnings = collect_windows_setup_preflight_issues()
    for warning in warnings:
        print(f"[PREFLIGHT][WARN] {warning}")
    if not issues:
        print("[PREFLIGHT] Python 3.11, setup packages, scripts, and OpenCV GUI are ready.")
        return True
    print("[PREFLIGHT] Setup cannot start in the current environment:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nActivate the tested environment and rerun the same setup command:")
    print("  conda activate sam2py311")
    print(f"  {setup_rerun_command(args)}")
    return False


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


def path_basename(value):
    return re.split(r"[\\/]+", str(value).rstrip("\\/"))[-1]


def rel_posix(path, base):
    return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()


def path_is_absolute_string(value):
    text = str(value)
    return bool(re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("/", "\\")))


def resolve_relative_path(base, rel):
    return (Path(base) / Path(*str(rel).split("/"))).resolve()


def queue_rel_from_stored(value, queue_dir, queue_id=None):
    text = str(value)
    if text.startswith(QUEUE_PATH_PREFIX):
        return text[len(QUEUE_PATH_PREFIX):].lstrip("/")
    if not path_is_absolute_string(text):
        return text.replace("\\", "/")
    normalized = text.replace("\\", "/")
    marker = f"/pipeline_queue/{queue_id}/" if queue_id else None
    if marker and marker in normalized:
        return normalized.split(marker, 1)[1]
    if queue_id:
        fallback = f"/{queue_id}/"
        if fallback in normalized:
            return normalized.split(fallback, 1)[1]
    try:
        return Path(text).resolve().relative_to(Path(queue_dir).resolve()).as_posix()
    except Exception:
        return None


def resolve_queue_path(value, queue_dir, queue_id=None):
    rel = queue_rel_from_stored(value, queue_dir, queue_id)
    if rel is not None:
        return resolve_relative_path(queue_dir, rel)
    return Path(value).expanduser().resolve()


def repo_rel_from_stored(value):
    text = str(value)
    if text.startswith(REPO_PATH_PREFIX):
        return text[len(REPO_PATH_PREFIX):].lstrip("/")
    if not path_is_absolute_string(text):
        return text.replace("\\", "/")
    normalized = text.replace("\\", "/")
    marker = "/triangulation/results/"
    if marker in normalized:
        return "triangulation/results/" + normalized.split(marker, 1)[1]
    try:
        return Path(text).resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return None


def resolve_repo_path(value):
    rel = repo_rel_from_stored(value)
    if rel is not None:
        return resolve_relative_path(REPO_ROOT, rel)
    return Path(value).expanduser().resolve()


def relativize_queue_path(path, queue_dir):
    return QUEUE_PATH_PREFIX + rel_posix(path, queue_dir)


def relativize_repo_path(path):
    return REPO_PATH_PREFIX + rel_posix(path, REPO_ROOT)


def portable_file_signature(path, include_hash=False):
    sig = file_signature(path, include_hash=include_hash)
    portable = {
        key: value
        for key, value in sig.items()
        if key not in {"path", "mtime_ns"}
    }
    portable["name"] = path_basename(path)
    return portable


def portable_video_signature(path):
    sig = video_signature(path)
    portable = {
        key: value
        for key, value in sig.items()
        if key not in {"path", "mtime_ns"}
    }
    portable["name"] = path_basename(path)
    return portable


def signatures_match(recorded, current, require_video_meta=True):
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    recorded_name = path_basename(recorded.get("path") or recorded.get("name") or "")
    current_name = path_basename(current.get("path") or current.get("name") or "")
    if recorded_name and current_name and recorded_name != current_name:
        return False
    try:
        if int(recorded.get("size_bytes", -1)) != int(current.get("size_bytes", -2)):
            return False
    except (TypeError, ValueError):
        return False
    if require_video_meta:
        for key in ("frame_count", "width", "height"):
            if key in recorded and int(recorded.get(key, -1)) >= 0:
                try:
                    if int(recorded.get(key)) != int(current.get(key, -2)):
                        return False
                except (TypeError, ValueError):
                    return False
        if "fps" in recorded and float(recorded.get("fps", -1.0)) > 0:
            try:
                if abs(float(recorded.get("fps")) - float(current.get("fps", -2.0))) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
    return True


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
        "size_bytes": int(stat.st_size),
        "name": path_basename(p),
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


def same_video(a, b, recorded_signature=None):
    if recorded_signature is not None:
        return signatures_match(recorded_signature, video_signature(b))
    try:
        left = a if isinstance(a, dict) else video_signature(a)
        return signatures_match(left, video_signature(b))
    except Exception:
        return False


def manifest_video_signature(manifest, path):
    if not manifest:
        return None
    basename = path_basename(path)
    for sig in manifest.get("source_videos", {}).values():
        if isinstance(sig, dict) and path_basename(sig.get("path") or sig.get("name") or "") == basename:
            return sig
    return None


def same_recorded_video(recorded_path, current_path, manifest=None):
    if same_video(recorded_path, current_path):
        return True
    if path_basename(recorded_path) != path_basename(current_path):
        return False
    recorded_sig = manifest_video_signature(manifest, current_path)
    if recorded_sig is None:
        return Path(current_path).is_file()
    return signatures_match(recorded_sig, video_signature(current_path))


def same_points_json(recorded, expected, setup_dir):
    if not recorded:
        return False
    recorded_name = path_basename(recorded)
    expected_path = Path(expected)
    if recorded_name != expected_path.name:
        return False
    try:
        recorded_path = Path(recorded).expanduser()
        # pathlib follows the current OS, so a Windows absolute path is
        # considered relative on Linux (and vice versa). Treat either syntax
        # as absolute when validating a queue copied between machines.
        if path_is_absolute_string(recorded):
            if same_path(recorded_path, expected_path):
                return True
            recorded_parts = Path(str(recorded).replace("\\", "/")).parts
            expected_parts = expected_path.parts
            return len(recorded_parts) >= 2 and recorded_parts[-2:] == expected_parts[-2:]
        return same_path(Path(setup_dir) / recorded_path, expected_path)
    except Exception:
        return recorded_name == expected_path.name


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


def resolve_manifest_paths(manifest_path, manifest):
    queue_dir = Path(manifest_path).parent.resolve()
    queue_id = manifest.get("queue_id")
    preprocessing = manifest.get("preprocessing", {})
    splitter = preprocessing.get("splitter", {})
    for key in ("output_root", "manifest"):
        if key in splitter:
            splitter[key] = str(resolve_queue_path(splitter[key], queue_dir, queue_id))
    calibration = preprocessing.get("calibration", {})
    for key in ("work_dir", "sync_json", "stats_dir", "mono_npz", "stereo_npz"):
        if key in calibration:
            if calibration.get("mode") == "generated":
                calibration[key] = str(resolve_queue_path(calibration[key], queue_dir, queue_id))
            elif str(calibration[key]).startswith(REPO_PATH_PREFIX):
                calibration[key] = str(resolve_repo_path(calibration[key]))
            else:
                calibration[key] = str(Path(calibration[key]).expanduser().resolve())
    for job in manifest.get("jobs", []):
        for key in ("run_dir", "setup_dir", "setup_json", "corrections_json", "sam2_out", "logs_dir"):
            if key in job:
                job[key] = str(resolve_queue_path(job[key], queue_dir, queue_id))
        if "result_dir" in job:
            job["result_dir"] = str(resolve_repo_path(job["result_dir"]))


def manifest_for_save(manifest_path, manifest):
    payload = copy.deepcopy(manifest)
    queue_dir = Path(manifest_path).parent.resolve()
    preprocessing = payload.get("preprocessing", {})
    splitter = preprocessing.get("splitter", {})
    for key in ("output_root", "manifest"):
        if key in splitter:
            splitter[key] = relativize_queue_path(splitter[key], queue_dir)
    calibration = preprocessing.get("calibration", {})
    for key in ("work_dir", "sync_json", "stats_dir", "mono_npz", "stereo_npz"):
        if key in calibration:
            if calibration.get("mode") == "generated":
                calibration[key] = relativize_queue_path(calibration[key], queue_dir)
            else:
                try:
                    calibration[key] = relativize_repo_path(calibration[key])
                except Exception:
                    pass
    for job in payload.get("jobs", []):
        for key in ("run_dir", "setup_dir", "setup_json", "corrections_json", "sam2_out", "logs_dir"):
            if key in job:
                job[key] = relativize_queue_path(job[key], queue_dir)
        if "result_dir" in job:
            job["result_dir"] = relativize_repo_path(job["result_dir"])
    return payload


def save_manifest(manifest_path, manifest):
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest_for_save(manifest_path, manifest))


def read_machine_video_roots():
    if not MACHINE_PATHS_FILE.is_file():
        return []
    try:
        payload = read_json(MACHINE_PATHS_FILE)
    except Exception:
        return []
    roots = []
    for value in payload.get("video_roots", []):
        path = Path(value).expanduser()
        if path.is_dir():
            roots.append(path.resolve())
    return roots


def find_video_by_signature(recorded):
    basename = path_basename(recorded.get("path") or recorded.get("name") or "")
    if not basename:
        return None
    matches = []
    for root in read_machine_video_roots():
        try:
            candidates = root.rglob(basename)
        except OSError:
            continue
        for candidate in candidates:
            if candidate.is_file() and signatures_match(recorded, video_signature(candidate)):
                matches.append(candidate.resolve())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files under machine_paths.json video_roots match {basename}: "
            + ", ".join(str(path) for path in matches[:5])
        )
    return None


def collect_recorded_video_signatures(manifest, split_payload=None):
    signatures = {}
    for sig in manifest.get("source_videos", {}).values():
        if isinstance(sig, dict):
            key = sig.get("path") or sig.get("name")
            if key:
                signatures[path_basename(key)] = sig
    preprocessing = manifest.get("preprocessing", {})
    for key in ("raw_left", "raw_right"):
        path = preprocessing.get(key)
        if path and Path(path).is_file():
            signatures[path_basename(path)] = video_signature(path)
    if split_payload:
        for key in ("left_source", "right_source"):
            sig = split_payload.get(key)
            if isinstance(sig, dict):
                signatures[path_basename(sig.get("path") or sig.get("name") or "")] = sig
    return signatures


def resolve_source_value(value, signatures, missing):
    path = Path(str(value)).expanduser()
    if path.is_file():
        return str(path.resolve()), False
    recorded = signatures.get(path_basename(value))
    if not recorded:
        missing.append(str(value))
        return str(value), False
    resolved = find_video_by_signature(recorded)
    if resolved is None:
        missing.append(str(value))
        return str(value), False
    return str(resolved), True


def load_split_payload(manifest):
    splitter = manifest.get("preprocessing", {}).get("splitter", {})
    path = Path(splitter.get("manifest", ""))
    if not path.is_file():
        return None
    payload = read_json(path)
    queue_dir = path.parent.parent
    queue_id = manifest.get("queue_id")
    if "output_root" in payload:
        payload["output_root"] = str(resolve_queue_path(payload["output_root"], queue_dir, queue_id))
    return payload


def save_split_payload(manifest, payload):
    splitter = manifest.get("preprocessing", {}).get("splitter", {})
    path = Path(splitter.get("manifest", ""))
    if path.is_file() or path.parent.exists():
        payload_for_save = copy.deepcopy(payload)
        if "output_root" in payload_for_save:
            payload_for_save["output_root"] = relativize_queue_path(
                payload_for_save["output_root"],
                path.parent.parent,
            )
        atomic_write_json(path, payload_for_save)


def resolve_source_videos(manifest_path, manifest):
    changed = False
    missing = []
    split_payload = load_split_payload(manifest)
    signatures = collect_recorded_video_signatures(manifest, split_payload)
    preprocessing = manifest.get("preprocessing", {})
    for key in ("raw_left", "raw_right"):
        if key in preprocessing:
            resolved, did_change = resolve_source_value(preprocessing[key], signatures, missing)
            preprocessing[key] = resolved
            changed = changed or did_change
    calibration = preprocessing.get("calibration", {})
    for key in ("left_video", "right_video"):
        if key in calibration:
            resolved, did_change = resolve_source_value(calibration[key], signatures, missing)
            calibration[key] = resolved
            changed = changed or did_change
    for job in manifest.get("jobs", []):
        for key in ("left_video", "right_video"):
            if key in job:
                resolved, did_change = resolve_source_value(job[key], signatures, missing)
                job[key] = resolved
                changed = changed or did_change
    if split_payload:
        split_changed = False
        for key in ("left_source", "right_source"):
            sig = split_payload.get(key)
            if isinstance(sig, dict):
                resolved, did_change = resolve_source_value(sig.get("path", ""), signatures, missing)
                if did_change:
                    sig["path"] = resolved
                    split_changed = True
        for clip in split_payload.get("clips", []):
            for key in ("left_video", "right_video"):
                if key in clip:
                    resolved, did_change = resolve_source_value(clip[key], signatures, missing)
                    clip[key] = resolved
                    split_changed = split_changed or did_change
        if split_changed:
            split_payload["updated_at"] = utc_now()
            save_split_payload(manifest, split_payload)
            changed = True
    if missing:
        unique = sorted(set(missing))
        raise FileNotFoundError(
            "Could not resolve source video(s):\n  "
            + "\n  ".join(unique)
            + f"\nAdd local search roots to {MACHINE_PATHS_FILE} as "
            '{"video_roots": ["/data/videos", "..."]}.'
        )
    current_sources = {}
    for label, path in (
        ("raw_left", preprocessing.get("raw_left")),
        ("raw_right", preprocessing.get("raw_right")),
    ):
        if path and Path(path).is_file():
            current_sources[label] = video_signature(path)
    if current_sources and manifest.get("source_videos") != current_sources:
        manifest["source_videos"] = current_sources
        changed = True
    if changed:
        save_manifest(manifest_path, manifest)
    return changed


def write_console_text(output):
    try:
        sys.stdout.write(output)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_output = output.encode(encoding, errors="replace").decode(encoding)
        sys.stdout.write(safe_output)
    sys.stdout.flush()


def write_streamed_output(output, log):
    if not output:
        return
    log.write(output)
    log.flush()
    write_console_text(output)


def stop_process_tree(process, graceful_timeout=8):
    """Stop a logged child and any workers it created."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (AttributeError, OSError, ValueError):
            pass
        try:
            process.wait(timeout=graceful_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=graceful_timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait()


def run_logged(command, cwd, log_path, use_pty=True):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display_command = format_command(command)
    write_console_text(f"\n$ {display_command}\n")
    with open(log_path, "a", buffering=1, encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {display_command}\n")
        popen_kwargs = {}
        master_fd = None
        slave_fd = None
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
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
            text=False,
            bufsize=0,
            env=child_env,
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
                    write_streamed_output(output, log)
                output = decoder.decode(b"", final=True)
                write_streamed_output(output, log)
            else:
                assert process.stdout is not None
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    write_streamed_output(decoder.decode(chunk), log)
                write_streamed_output(decoder.decode(b"", final=True), log)
            return_code = process.wait()
        except KeyboardInterrupt:
            stop_process_tree(process)
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
    if not same_recorded_video(setup_meta.get("left_input"), job["left_video"], manifest):
        return False
    if not same_recorded_video(setup_meta.get("right_input"), job["right_video"], manifest):
        return False
    if not same_points_json(setup_meta.get("points_json"), setup_json, job["setup_dir"]):
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
        "version": 3,
        "stage": "sam2",
        "frame_cache_image_format": "jpg",
        "setup_mode": manifest.get("setup_mode", "grid"),
        "left_video": portable_video_signature(job["left_video"]),
        "right_video": portable_video_signature(job["right_video"]),
        "frame_range": frame_range_signature(job["start_frame"], job["end_frame"]),
        "setup_json": portable_file_signature(job["setup_json"], include_hash=True),
        "corrections_json": portable_file_signature(job["corrections_json"], include_hash=True),
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
    if not same_video(frame_meta.get("video", {}), job[f"{side_name}_video"]):
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


def sam2_complete_readonly(job, manifest=None):
    if manifest is None or not sam2_outputs_valid(job, manifest):
        return False
    meta_path = Path(job["sam2_out"]) / QUEUE_SAM2_META
    if not meta_path.is_file():
        return False
    try:
        return read_json(meta_path).get("fingerprint") == sam2_stage_fingerprint(job, manifest)
    except Exception:
        return False


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
        "left_tracks": portable_file_signature(sam2_out / "left" / "tracks_2d.csv", include_hash=True),
        "right_tracks": portable_file_signature(sam2_out / "right" / "tracks_2d.csv", include_hash=True),
        "left_video": portable_video_signature(job["left_video"]),
        "right_video": portable_video_signature(job["right_video"]),
        "frame_range": frame_range_signature(job["start_frame"], job["end_frame"]),
        "stereo": portable_file_signature(stereo_npz, include_hash=True),
        "sync_json": portable_file_signature(sync_json, include_hash=True),
        "quality_min": 0.0,
        "max_reproj": 20.0,
        "sync_mode": "default",
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


def migrate_manifest(manifest, manifest_path=None, refresh_fingerprints=True):
    changed = False
    was_pre_v3 = int(manifest.get("version", 1)) < 3
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
    if was_pre_v3:
        manifest["version"] = 3
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
    source_videos = manifest.setdefault("source_videos", {})
    preprocessing = manifest.get("preprocessing", {})
    for label, key in (("raw_left", "raw_left"), ("raw_right", "raw_right")):
        path = preprocessing.get(key)
        if path and Path(path).is_file():
            sig = video_signature(path)
            if source_videos.get(label) != sig:
                source_videos[label] = sig
                changed = True
    if refresh_fingerprints and was_pre_v3 and refresh_complete_fingerprints(manifest):
        changed = True
    return changed


def refresh_sam2_batch_fingerprints(job, manifest):
    if not Path(job["setup_json"]).is_file():
        return False
    try:
        setup = load_setup_payload(job)
        corrections = load_corrections_payload(job)
    except Exception:
        return False
    changed = False
    expected_ids = expected_object_ids(manifest, job)
    if not expected_ids:
        return False
    expected_batches = chunk_ids(expected_ids, int(manifest["settings"]["batch_size"]))
    for side_name in ("left", "right"):
        side_out = Path(job["sam2_out"]) / side_name / "objectwise"
        for batch_ids in expected_batches:
            batch_name = f"batch_{batch_ids[0]:03d}_{batch_ids[-1]:03d}"
            meta_path = side_out / batch_name / "batch_meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = read_json(meta_path)
            except Exception:
                continue
            expected = batch_fingerprint(side_name, batch_ids, setup[side_name], manifest, corrections)
            if meta.get("fingerprint") != expected:
                meta["fingerprint"] = expected
                meta["fingerprint_version"] = 2
                atomic_write_json(meta_path, meta)
                changed = True
    return changed


def refresh_complete_fingerprints(manifest):
    changed = False
    calibration = manifest.get("preprocessing", {}).get("calibration", {})
    if calibration.get("mode") == "generated":
        for stage_name in ("sync", "stats", "mono", "stereo"):
            stage = calibration.get("stages", {}).get(stage_name, {})
            outputs = calibration_stage_outputs(calibration, stage_name)
            if stage.get("status") == "complete" and all(
                path.is_file() and path.stat().st_size > 0 for path in outputs
            ):
                expected = calibration_stage_fingerprint(manifest, stage_name)
                if stage.get("fingerprint") != expected:
                    stage["fingerprint"] = expected
                    changed = True
    for job in manifest.get("jobs", []):
        if refresh_sam2_batch_fingerprints(job, manifest):
            changed = True
        if sam2_outputs_valid(job, manifest):
            meta_path = Path(job["sam2_out"]) / QUEUE_SAM2_META
            expected = sam2_stage_fingerprint(job, manifest)
            old = read_json(meta_path).get("fingerprint") if meta_path.is_file() else None
            if old != expected:
                write_sam2_queue_meta(job, manifest)
                changed = True
            stage = job.get("stages", {}).get("sam2", {})
            if stage.get("status") == "complete" and stage.get("fingerprint") != expected:
                stage["fingerprint"] = expected
                changed = True
        result_dir = Path(job.get("result_dir", ""))
        expected_outputs = [
            result_dir / "triangulated_3d.csv",
            result_dir / "summary.json",
            *visualization_outputs(result_dir),
        ]
        if all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs):
            meta_path = result_dir / QUEUE_TRIANGULATION_META
            data_fp = triangulation_data_fingerprint(job, manifest)
            full_fp = triangulation_full_fingerprint(job, manifest)
            try:
                meta = read_json(meta_path) if meta_path.is_file() else {}
            except Exception:
                meta = {}
            if meta.get("data_fingerprint") != data_fp or meta.get("full_fingerprint") != full_fp:
                write_triangulation_queue_meta(job, manifest)
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
    try:
        payload = load_split_payload(manifest)
    except Exception:
        return False
    if not payload:
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
        if not same_video(recorded, source_path):
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
    split_payload = load_split_payload(manifest)
    if not split_payload:
        raise RuntimeError("Split manifest is missing or unreadable.")
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
            if not same_video(job["left_video"], clip["left_video"]) or not same_video(
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
        "version": 3,
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
        "source_videos": {
            "raw_left": video_signature(left_video),
            "raw_right": video_signature(right_video),
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
    manifest = json.loads(path.read_text())
    resolve_manifest_paths(path, manifest)
    return path, manifest


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
        "left_video": portable_video_signature(calibration["left_video"]),
        "right_video": portable_video_signature(calibration["right_video"]),
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
            portable_file_signature(path, include_hash=True)
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


def prepare_calibration(manifest_path, manifest, stage_names=None):
    calibration = manifest["preprocessing"]["calibration"]
    stage_names = tuple(stage_names or ("sync", "stats", "mono", "stereo"))
    if calibration.get("mode") == "existing":
        if not calibration_complete(manifest):
            sync_json, stereo_npz = calibration_paths(manifest)
            raise RuntimeError(
                f"Existing calibration is incomplete: sync={sync_json}, stereo={stereo_npz}"
            )
        print("[SKIP][CALIBRATION] Using calibration files recorded by this legacy queue.")
        return True

    print("\n[CALIBRATION] Clip pair 1: " + " -> ".join(stage_names))
    logs_dir = Path(calibration["work_dir"]) / "logs"
    for stage_name in stage_names:
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
    return all(calibration_stage_complete(manifest, stage_name) for stage_name in stage_names)


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
        print("[SETUP] Starting the marker-setup program.")
        print(
            "[SETUP] It will load both frames and estimate markers before the OpenCV review "
            "windows open; high-resolution frames can take about a minute."
        )
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


def process_only_missing(manifest_path, manifest):
    missing = []
    setup_command = format_command(
        [
            "python",
            str(Path(__file__).resolve()),
            "--resume",
            str(Path(manifest_path).parent),
            "--setup-only",
        ]
    )
    preprocessing = manifest.get("preprocessing", {})
    if not splitter_complete(manifest):
        missing.append("splitter GUI virtual ranges are incomplete")
    try:
        split_payload = load_split_payload(manifest)
        expected_jobs = max(0, len(split_payload.get("clips", [])) - 1) if split_payload else 0
    except Exception:
        expected_jobs = 0
    jobs = manifest.get("jobs", [])
    if not jobs or (expected_jobs and len(jobs) < expected_jobs):
        missing.append(
            f"job creation/velocity labels are incomplete ({len(jobs)}/{expected_jobs or '?'} jobs labeled)"
        )
    calibration = preprocessing.get("calibration", {})
    if calibration.get("mode") == "generated" and not calibration_stage_complete(manifest, "sync"):
        missing.append("calibration sync/audio-offset confirmation is incomplete")
    elif calibration.get("mode") == "existing":
        sync_json, _ = calibration_paths(manifest)
        if not sync_json.is_file():
            missing.append(f"existing calibration sync JSON is missing: {sync_json}")
    if manifest.get("setup_mode", "grid") == "sectioned" and not manifest.get("section_layout"):
        missing.append("section layout is missing")
    for job in jobs:
        if not setup_complete(job, manifest):
            missing.append(f"setup is incomplete for job {job.get('index')}: {job.get('velocity')}")
    return missing, setup_command


def validate_process_only(manifest_path, manifest):
    missing, setup_command = process_only_missing(manifest_path, manifest)
    if not missing:
        return True
    print("[PROCESS-ONLY] Interactive setup is incomplete; refusing to run unattended processing.")
    print("[PROCESS-ONLY] Missing:")
    for item in missing:
        print(f"  - {item}")
    print("[PROCESS-ONLY] Run this on the setup machine:")
    print(f"  {setup_command}")
    return False


def stage_label(status):
    if status is True:
        return "complete"
    if status is False:
        return "pending"
    return str(status)


def print_check_status(manifest_path, manifest):
    rows = []
    rows.append(("split", stage_label(splitter_complete(manifest))))
    missing, _ = process_only_missing(manifest_path, manifest)
    rows.append(("process-ready", "complete" if not missing else "pending"))
    calibration = manifest.get("preprocessing", {}).get("calibration", {})
    if calibration.get("mode") == "generated":
        for stage_name in ("sync", "stats", "mono", "stereo"):
            stage = calibration.get("stages", {}).get(stage_name, {})
            status = "failed" if stage.get("status") == "failed" else stage_label(
                calibration_stage_complete(manifest, stage_name)
            )
            rows.append((f"calibration:{stage_name}", status))
    else:
        rows.append(("calibration:existing", stage_label(calibration_complete(manifest))))
    if manifest.get("setup_mode", "grid") == "sectioned":
        rows.append(("section-layout", "complete" if manifest.get("section_layout") else "pending"))
    for job in manifest.get("jobs", []):
        prefix = f"job {job.get('index')} {job.get('velocity')}"
        rows.append((f"{prefix}:setup", stage_label(setup_complete(job, manifest))))
        rows.append((f"{prefix}:sam2", stage_label(sam2_complete_readonly(job, manifest))))
        rows.append((f"{prefix}:triangulation", stage_label(triangulation_complete(job, manifest))))
    width = max(len(name) for name, _ in rows) if rows else 5
    print(f"[CHECK] Queue: {Path(manifest_path).parent}")
    print("[CHECK] Stage".ljust(width + 10) + "Status")
    for name, status in rows:
        print(f"[CHECK] {name.ljust(width)}  {status}")
    failed = [name for name, status in rows if status == "failed"]
    if missing:
        print("[CHECK] Process-only missing:")
        for item in missing:
            print(f"  - {item}")
    if failed:
        print("[CHECK] Failed stages: " + ", ".join(failed))
    return not failed


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
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--setup-only",
        action="store_true",
        help="Run only interactive setup stages, then print the processing handoff.",
    )
    phase.add_argument(
        "--process-only",
        action="store_true",
        help="Run only non-interactive calibration compute, SAM2, and triangulation.",
    )
    phase.add_argument(
        "--check-only",
        action="store_true",
        help="Validate queue paths and stage state without opening GUIs or prompts.",
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


def load_existing_queue(args):
    manifest_path, manifest = load_manifest(args.resume)
    changed = resolve_source_videos(manifest_path, manifest)
    changed = migrate_manifest(
        manifest,
        manifest_path,
        refresh_fingerprints=not args.check_only,
    ) or changed
    if not args.check_only:
        changed = apply_cli_overrides(manifest, args) or changed
    if changed:
        save_manifest(manifest_path, manifest)
    return manifest_path, manifest


def mark_setup_only_interrupted(manifest_path, manifest):
    interrupted = []

    def reset_stage(stage, label):
        if isinstance(stage, dict) and stage.get("status") == "running":
            stage.update(
                {
                    "status": "pending",
                    "interrupted_at": utc_now(),
                    "error": "Interrupted by user; resume setup to retry this stage.",
                }
            )
            interrupted.append(label)

    preprocessing = manifest.get("preprocessing", {})
    reset_stage(preprocessing.get("splitter", {}).get("stage"), "splitter")
    reset_stage(
        preprocessing.get("calibration", {}).get("stages", {}).get("sync"),
        "calibration sync",
    )
    for job in manifest.get("jobs", []):
        reset_stage(
            job.get("stages", {}).get("setup"),
            f"marker setup for {job.get('velocity', 'unknown run')}",
        )
    if interrupted:
        manifest["status"] = "incomplete"
        save_manifest(manifest_path, manifest)
    return interrupted


def main():
    args = parse_args()
    if (args.process_only or args.check_only) and not args.resume:
        print("[QUEUE] --process-only and --check-only require --resume.")
        return 2
    if args.setup_only and not validate_windows_setup_environment(args):
        return 2

    manifest_path = None
    manifest = None
    try:
        if args.resume:
            try:
                manifest_path, manifest = load_existing_queue(args)
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"[QUEUE] {exc}")
                return 1
            print(f"[QUEUE] Resuming: {manifest_path}")
        else:
            manifest_path, manifest = create_manifest(args)
        print_queue_settings(manifest)
        if args.check_only:
            return 0 if print_check_status(manifest_path, manifest) else 1
        if args.process_only:
            if not validate_process_only(manifest_path, manifest):
                return 1
            if bool(manifest.setdefault("settings", {}).get("preview", False)):
                print("[PROCESS-ONLY] Disabling SAM2 preview for non-interactive processing.")
                manifest["settings"]["preview"] = False
                save_manifest(manifest_path, manifest)
            if not prepare_calibration(manifest_path, manifest, ("stats", "mono", "stereo")):
                print("[QUEUE] Calibration compute is incomplete. Resume this manifest to try again.")
                return 1
            process_jobs(manifest_path, manifest)
            return 0
        if not prepare_splitter(manifest_path, manifest):
            print("[QUEUE] Splitter is incomplete. Resume this manifest to try again.")
            return 1
        ensure_jobs_from_split(manifest_path, manifest)
        calibration_stages = ("sync",) if args.setup_only else ("sync", "stats", "mono", "stereo")
        if not prepare_calibration(manifest_path, manifest, calibration_stages):
            print("[QUEUE] Calibration is incomplete. Resume this manifest to try again.")
            return 1
        ensure_section_layout(manifest_path, manifest)
        prepare_setups(manifest_path, manifest)
        if args.setup_only:
            missing, setup_command = process_only_missing(manifest_path, manifest)
            if missing:
                print("\n[QUEUE] Interactive setup is incomplete; queue is not ready for processing.")
                print("[QUEUE] Missing:")
                for item in missing:
                    print(f"  - {item}")
                print("[QUEUE] Re-run setup on this machine:")
                print(f"  {setup_command}")
                return 1
            print("\n[QUEUE] Interactive setup is complete.")
            print(f"[QUEUE] Queue dir: {Path(manifest_path).parent}")
            print("[QUEUE] Copy this queue directory to the processing machine and run:")
            print(f'  python run_pipeline_queue.py --resume "{Path(manifest_path).parent}" --process-only')
            print("[QUEUE] Or use the remote helper:")
            print("  python remote_pipeline.py push")
            print("  python remote_pipeline.py run")
            return 0
        process_jobs(manifest_path, manifest)
        return 0
    except KeyboardInterrupt:
        interrupted = []
        if args.setup_only and manifest_path is not None and manifest is not None:
            interrupted = mark_setup_only_interrupted(manifest_path, manifest)
        print("\n[QUEUE] Setup interrupted by user.")
        if interrupted:
            print("[QUEUE] Saved resumable stage state: " + ", ".join(interrupted))
        print("[QUEUE] Resume from the tested environment:")
        print("  conda activate sam2py311")
        resume_path = Path(manifest_path).parent if manifest_path is not None else None
        print(f"  {setup_rerun_command(args, resume_path=resume_path)}")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
