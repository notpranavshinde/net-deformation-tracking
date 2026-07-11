r"""
Manual SAM2 marker setup and tracking.

Paste-ready commands from this directory:

    python run_sam2_markers.py --setup
    python run_sam2_markers.py --semi-auto-setup
    python run_sam2_markers.py --semi-auto-setup-sections
    python run_sam2_markers.py --modify-setup
    python run_sam2_markers.py --reuse-setup

Defaults expect:
    in/left.mp4
    in/right.mp4

Use --setup on the local Windows machine for manual clicking.
Use --semi-auto-setup to generate grid prompts from detected painted markers
after clicking the four outer grid corners.
Use --semi-auto-setup-sections for irregular nets made from several local grids.
Default crop is full-frame. Add --select-crop only when you want to crop.
Use --modify-setup to load the saved setup and add more points without
re-clicking everything.

Local half-scale test with live mask preview:

    python run_sam2_markers.py --reuse-setup --scale 0.5 --frame-extractor auto --gpu-mode single --preview true --save-overlay false --save-masks false --save-tracks true
"""

import os
import cv2
import json
import csv
import shutil
import concurrent.futures
import multiprocessing
import queue
import time
import subprocess
import gc
import hashlib
import math
import re
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np
import torch
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

try:
    from auto_detect_sam2_prompts import detect_candidates_on_processed_frame
except Exception:
    detect_candidates_on_processed_frame = None

# ---- CONFIG ----
DEFAULT_OUT_DIR = Path("out")
DEFAULT_LOCAL_SETUP_DIR = Path("work") / "manual_sam2_setup"
DEFAULT_CORRECTIONS_PATH = DEFAULT_LOCAL_SETUP_DIR / "prompts" / "corrections.json"
NO_CROP_CORRECTIONS_PATH = DEFAULT_LOCAL_SETUP_DIR / "prompts" / "corrections_no_crop.json"
MODEL_ID = "facebook/sam2.1-hiera-large"  # or base-plus for speed
# MODEL_ID = "facebook/sam2.1-hiera-base-plus"
FRAME_CACHE_IMAGE_FORMAT = "jpg"
FRAME_CACHE_EXT = ".jpg"
FRAME_CACHE_GLOB = "*.jpg"
INTERACTIVE_ZOOM_STEP = 1.5

DOT_AREA_MIN = 10
DOT_AREA_MAX = 1_000_000
DOT_CORE_RADIUS_MIN = 1.0
DOT_SOLIDITY_MIN = 0.5
DOT_ERODE_KERNEL_SIZE = 3
TRACK_FIELDNAMES = [
    "frame",
    "obj_id",
    "u",
    "v",
    "u_local",
    "v_local",
    "method",
    "area_px",
    "core_radius_px",
    "solidity",
    "quality",
    "valid",
]

# ---- UTILS ----
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y", "on")


def resolve_video_path(video_path):
    path = Path(video_path)
    candidates = [path]
    if path.suffix:
        candidates.extend([path.with_suffix(path.suffix.lower()), path.with_suffix(path.suffix.upper())])
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return str(candidate)
    return str(path)


def parse_crop_arg(crop_arg: str):
    """
    Parse crop string in x,y,w,h format. The literal "none" disables cropping.
    Returns (x, y, w, h) as ints or None.
    """
    if crop_arg is None:
        return None
    if str(crop_arg).strip().lower() in ("none", "no", "false", "off", "full"):
        return None
    parts = [p.strip() for p in str(crop_arg).split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be in format x,y,w,h, or 'none'")

    try:
        x, y, w, h = [int(v) for v in parts]
    except ValueError as e:
        raise ValueError("--crop values must be integers in format x,y,w,h") from e

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("--crop requires x>=0, y>=0, w>0, h>0")

    return x, y, w, h


def is_crop_none_arg(crop_arg: str) -> bool:
    return crop_arg is not None and str(crop_arg).strip().lower() in (
        "none",
        "no",
        "false",
        "off",
        "full",
    )


def offset_points(points, crop):
    if crop is None:
        return points
    return translate_points(points, float(crop[0]), float(crop[1]))


def translate_points(points, x_off, y_off):
    x_off, y_off = float(x_off), float(y_off)
    shifted = []
    for point in points:
        x, y, label = _coerce_prompt_point(point)
        negatives = _coerce_negative_points(point)
        shifted_negatives = [[float(nx) + x_off, float(ny) + y_off] for nx, ny in negatives]
        shifted.append(_make_prompt_point(float(x) + x_off, float(y) + y_off, label, shifted_negatives))
    return shifted


def offset_corrections_payload(payload, side_crops):
    shifted = json.loads(json.dumps(payload))
    corrections = shifted.setdefault("corrections", {})
    for side, crop in side_crops.items():
        if crop is None:
            continue
        x_off, y_off = float(crop[0]), float(crop[1])
        for corr in corrections.get(side, []):
            corr["positive"] = [[float(x) + x_off, float(y) + y_off] for x, y in corr.get("positive", [])]
            corr["negative"] = [[float(x) + x_off, float(y) + y_off] for x, y in corr.get("negative", [])]
    shifted["coordinate_space"] = "original_full_frame_no_crop"
    return shifted


def _make_prompt_point(x, y, label=1, negative=None):
    point = {"x": float(x), "y": float(y), "label": int(label)}
    if negative:
        point["negative"] = [[float(nx), float(ny)] for nx, ny in negative]
    return point


def _coerce_prompt_point(row, prefer_local=True):
    """Read the main positive point. Old [x,y] rows are treated as positive."""
    if isinstance(row, dict):
        label = int(row.get("label", row.get("sam2_label", 1)))
        if prefer_local and "local_x" in row and "local_y" in row:
            return float(row["local_x"]), float(row["local_y"]), label
        if "x" in row and "y" in row:
            return float(row["x"]), float(row["y"]), label
        if "click_xy" in row and len(row["click_xy"]) >= 2:
            return float(row["click_xy"][0]), float(row["click_xy"][1]), label
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        label = int(row[2]) if len(row) >= 3 else 1
        return float(row[0]), float(row[1]), label
    raise ValueError(f"Could not parse point row: {row}")


def _coerce_negative_points(row):
    if not isinstance(row, dict):
        return []
    values = row.get("negative", row.get("negative_points", []))
    negatives = []
    for item in values:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                negatives.append([float(item["x"]), float(item["y"])])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            negatives.append([float(item[0]), float(item[1])])
    return negatives


def _coerce_point_pair(row, prefer_local=True):
    x, y, _label = _coerce_prompt_point(row, prefer_local=prefer_local)
    return [float(x), float(y)]


def load_points_file(path: str, side: str = None, prefer_local: bool = True):
    """Load points from CSV/JSON for headless SAM2 runs.

    Supported inputs:
    - auto_detect_sam2_prompts CSV with local_x/local_y or x/y columns
    - auto_detect_sam2_prompts JSON with a candidates list
    - shared points_left_right.json with {"left": [[x,y]], "right": [[x,y]]}
    - plain JSON list: [[x,y], ...]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Points file not found: {p}")

    if p.suffix.lower() == ".csv":
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        points = [
            _make_prompt_point(*_coerce_prompt_point(row, prefer_local=prefer_local), _coerce_negative_points(row))
            for row in rows
        ]
    else:
        payload = json.loads(p.read_text())
        if isinstance(payload, dict) and side and side in payload:
            rows = payload[side]
        elif isinstance(payload, dict) and "candidates" in payload:
            rows = payload["candidates"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(
                f"Unsupported points JSON format in {p}. Expected candidates, side key, or list."
            )
        points = [
            _make_prompt_point(*_coerce_prompt_point(row, prefer_local=prefer_local), _coerce_negative_points(row))
            for row in rows
        ]

    if not points:
        raise ValueError(f"No points loaded from {p}")
    return points


def load_crop_from_auto_points(path: str):
    """Return crop tuple from auto-detect JSON if present, otherwise None."""
    if not path or Path(path).suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(Path(path).read_text())
        crop = payload.get("crop")
        if not crop:
            return None
        return int(crop["x"]), int(crop["y"]), int(crop["w"]), int(crop["h"])
    except Exception:
        return None


def _empty_corrections_payload():
    return {
        "version": 1,
        "coordinate_space": "cropped_original_scale",
        "corrections": {"left": [], "right": []},
    }


def load_corrections(path: Path = DEFAULT_CORRECTIONS_PATH):
    """Load saved SAM2 correction prompts, if present."""
    p = Path(path)
    if not p.exists():
        return _empty_corrections_payload()

    payload = json.loads(p.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid corrections JSON: {p}")

    corrections = payload.setdefault("corrections", {})
    corrections.setdefault("left", [])
    corrections.setdefault("right", [])
    payload.setdefault("version", 1)
    payload.setdefault("coordinate_space", "cropped_original_scale")
    return payload


def save_corrections(payload, path: Path = DEFAULT_CORRECTIONS_PATH):
    """Persist correction prompts under the saved manual setup package."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Saved corrections: {p}")


def corrections_for_side(payload, side_name: str):
    side_key = str(side_name).lower()
    return list(payload.get("corrections", {}).get(side_key, []))


def filter_corrections_for_obj_ids(payload, side_name: str, obj_ids):
    wanted = {int(obj_id) for obj_id in obj_ids}
    filtered = _empty_corrections_payload()
    side_key = str(side_name).lower()
    filtered["corrections"][side_key] = [
        corr for corr in corrections_for_side(payload, side_key)
        if int(corr.get("obj_id", -1)) in wanted
    ]
    return filtered


def _crop_dict(crop, video_path: str):
    """Build crop metadata in the same format as side prompts/crop_roi.json."""
    if crop is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video to write crop metadata: {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "crop_applied": False,
            "x": 0,
            "y": 0,
            "w": int(w),
            "h": int(h),
            "coordinate_space": "original",
            "notes": "No crop applied; points are in original video coordinates.",
        }

    x, y, w, h = [int(v) for v in crop]
    return {
        "crop_applied": True,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "coordinate_space": "cropped",
        "notes": "Add x/y offsets back to tracked points to convert cropped coordinates to original video coordinates.",
    }


def write_setup_package(out_root: Path,
                        left_video: str,
                        right_video: str,
                        left_points,
                        right_points,
                        crop_left,
                        crop_right,
                        start_frame: int = 0,
                        end_frame: int = None):
    """Write portable interactive setup files for later headless SAM2 runs."""
    out_root.mkdir(parents=True, exist_ok=True)

    left_crop_meta = _crop_dict(crop_left, left_video)
    left_crop_meta.update({"side": "left", "video": str(left_video)})
    right_crop_meta = _crop_dict(crop_right, right_video)
    right_crop_meta.update({"side": "right", "video": str(right_video)})

    shared_prompts_dir = out_root / "prompts"
    shared_prompts_dir.mkdir(parents=True, exist_ok=True)
    shared_payload = {
        "left": left_points,
        "right": right_points,
        "crops": {
            "left": left_crop_meta,
            "right": right_crop_meta,
        },
        "frame_range": {
            "start_frame": int(start_frame),
            "end_frame": int(end_frame) if end_frame is not None else None,
        },
        "notes": (
            "Portable SAM2 setup package. Run headlessly with "
            "--points-json prompts/points_left_right.json --select-crop false --preview false."
        ),
    }
    with open(shared_prompts_dir / "points_left_right.json", "w") as f:
        json.dump(shared_payload, f, indent=2)

    for side, points, crop_meta in (
        ("left", left_points, left_crop_meta),
        ("right", right_points, right_crop_meta),
    ):
        side_prompts = out_root / side / "prompts"
        side_prompts.mkdir(parents=True, exist_ok=True)
        with open(side_prompts / "frame0_points.json", "w") as f:
            json.dump(points, f, indent=2)
        with open(side_prompts / "crop_roi.json", "w") as f:
            json.dump(crop_meta, f, indent=2)

    headless_points_rel = f"./work/{out_root.name}/prompts/points_left_right.json"
    manifest = {
        "left_input": str(left_video),
        "right_input": str(right_video),
        "points_json": str(shared_prompts_dir / "points_left_right.json"),
        "left_count": len(left_points),
        "right_count": len(right_points),
        "frame_range": shared_payload["frame_range"],
        "headless_command_example": (
            "python run_sam2_markers.py --left-input ./in/left.mp4 --right-input ./in/right.mp4 "
            f"--points-json {headless_points_rel} --out ./out "
            "--select-crop false --preview false --gpu-mode dual"
        ),
    }
    with open(out_root / "setup_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Wrote setup package: {out_root}")
    print(f"[OK] Shared prompts: {shared_prompts_dir / 'points_left_right.json'}")
    print(f"[OK] LEFT crop: {out_root / 'left' / 'prompts' / 'crop_roi.json'}")
    print(f"[OK] RIGHT crop: {out_root / 'right' / 'prompts' / 'crop_roi.json'}")


def load_crops_from_points_json(path: str):
    if not path:
        return None, None
    try:
        payload = json.loads(Path(path).read_text())
        crops = payload.get("crops", {})

        def parse_side(side):
            c = crops.get(side)
            if not c or not c.get("crop_applied", False):
                return None
            return int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"])

        return parse_side("left"), parse_side("right")
    except Exception as e:
        print(f"[WARN] Could not load crops from {path}: {e}")
        return None, None


def build_local_node_prompts(x: float, y: float, img_w: int, img_h: int):
    """
    Build the initial SAM2 prompt for a clicked node.
    Setup clicks are intentionally passed as positive points only.
    """
    x = float(x)
    y = float(y)
    pts_arr = np.asarray([[x, y]], dtype=np.float32)
    lbl_arr = np.asarray([1], dtype=np.int32)
    return pts_arr, lbl_arr, None


def build_setup_prompt(point, scale: float):
    x, y, label = _coerce_prompt_point(point, prefer_local=False)
    pts = [[float(x) * scale, float(y) * scale]]
    labels = [int(label)]
    for nx, ny in _coerce_negative_points(point):
        pts.append([float(nx) * scale, float(ny) * scale])
        labels.append(0)
    pts_arr = np.asarray(pts, dtype=np.float32)
    lbl_arr = np.asarray(labels, dtype=np.int32)
    return pts_arr, lbl_arr, None


def _prompt_int(prompt: str, default: int = None, minimum: int = 1) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return int(default)
        try:
            value = int(raw)
        except ValueError:
            print(f"Enter an integer >= {minimum}.")
            continue
        if value >= minimum:
            return value
        print(f"Enter an integer >= {minimum}.")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip().lower()
        if raw == "":
            return bool(default)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def _parse_section_layout(raw: str):
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    sections = []
    for idx, token in enumerate(re.split(r"[,;]+", text), start=1):
        token = token.strip().lower()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*(?:x|\*)\s*(\d+)", token)
        if not match:
            raise ValueError(
                f"Invalid section layout item {idx}: {token!r}. Use forms like '6x6,8x4'."
            )
        cols = int(match.group(1))
        rows = int(match.group(2))
        if cols < 2 or rows < 2:
            raise ValueError(f"Section layout item {idx} must have cols and rows >= 2.")
        sections.append({"cols": cols, "rows": rows})
    if not sections:
        return None
    return sections


def _open_cv_window(name: str, width: int = None, height: int = None):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    try:
        if width is not None and height is not None:
            cv2.resizeWindow(name, int(width), int(height))
        else:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    except Exception:
        pass


def _bilinear_grid_point(corners, col: int, row: int, cols: int, rows: int):
    tl, tr, bl, br = [np.asarray(pt, dtype=np.float32) for pt in corners]
    u = 0.0 if cols <= 1 else float(col) / float(cols - 1)
    v = 0.0 if rows <= 1 else float(row) / float(rows - 1)
    top = (1.0 - u) * tl + u * tr
    bottom = (1.0 - u) * bl + u * br
    xy = (1.0 - v) * top + v * bottom
    return float(xy[0]), float(xy[1])


def _make_detector_args(method: str, max_candidates: int = 0):
    return SimpleNamespace(
        method=method,
        scale=1.0,
        min_area=35.0,
        max_area=1600.0,
        min_distance=28.0,
        min_score=0.38,
        max_candidates=int(max_candidates),
        debug=False,
    )


def _detect_marker_candidates(side: str, frame_bgr: np.ndarray, expected_count: int):
    if detect_candidates_on_processed_frame is None:
        print("[WARN] auto_detect_sam2_prompts.py could not be imported; using grid fallback only.")
        return [], {"import_error": True}

    runs = []
    for method in ("color", "all"):
        args = _make_detector_args(method)
        _raw, counts, merged = detect_candidates_on_processed_frame(
            side,
            frame_bgr,
            (0, 0),
            args,
            debug_dir=None,
        )
        runs.append((method, counts, merged))
        if len(merged) >= max(4, int(round(expected_count * 0.75))):
            break

    method, counts, merged = runs[-1]
    print(f"[INFO][{side.upper()}] semi-auto detector method={method} candidates={len(merged)} expected={expected_count}")
    return list(merged), {"method": method, "counts": counts, "candidates": len(merged)}


def _grid_homographies(corners, cols: int, rows: int):
    image_pts = np.asarray(corners, dtype=np.float32)
    grid_pts = np.asarray(
        [[0.0, 0.0], [cols - 1.0, 0.0], [0.0, rows - 1.0], [cols - 1.0, rows - 1.0]],
        dtype=np.float32,
    )
    to_grid = cv2.getPerspectiveTransform(image_pts, grid_pts)
    to_image = cv2.getPerspectiveTransform(grid_pts, image_pts)
    if not np.isfinite(to_grid).all() or not np.isfinite(to_image).all():
        raise RuntimeError("Could not build a stable grid transform from the clicked corners.")
    return to_grid, to_image


def _apply_homography(points, homography):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    return out


def _grid_step_from_corners(corners, cols: int, rows: int) -> float:
    distances = []
    for row in range(rows):
        prev = None
        for col in range(cols):
            pt = np.asarray(_bilinear_grid_point(corners, col, row, cols, rows), dtype=np.float64)
            if prev is not None:
                distances.append(float(np.linalg.norm(pt - prev)))
            prev = pt
    for col in range(cols):
        prev = None
        for row in range(rows):
            pt = np.asarray(_bilinear_grid_point(corners, col, row, cols, rows), dtype=np.float64)
            if prev is not None:
                distances.append(float(np.linalg.norm(pt - prev)))
            prev = pt
    if not distances:
        return 40.0
    return max(float(np.median(distances)), 1.0)


def _candidate_grid_records(candidates, to_grid, cols: int, rows: int):
    if not candidates:
        return []
    image_xy = np.asarray([[float(c.x), float(c.y)] for c in candidates], dtype=np.float32)
    grid_xy = _apply_homography(image_xy, to_grid)
    records = []
    for idx, (cand, (gx, gy)) in enumerate(zip(candidates, grid_xy)):
        if -0.70 <= gx <= cols - 0.30 and -0.70 <= gy <= rows - 0.30:
            records.append({
                "idx": int(idx),
                "candidate": cand,
                "x": float(cand.x),
                "y": float(cand.y),
                "grid_x": float(gx),
                "grid_y": float(gy),
                "score": float(getattr(cand, "score", 0.0)),
                "area": float(getattr(cand, "area", 0.0)),
            })
    return records


def _assign_candidates_by_grid(records, cols: int, rows: int, max_grid_dist: float):
    pairs = []
    for rec in records:
        col = int(round(rec["grid_x"]))
        row = int(round(rec["grid_y"]))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        dist = math.hypot(rec["grid_x"] - float(col), rec["grid_y"] - float(row))
        if dist <= max_grid_dist:
            pairs.append((dist, -rec["score"], row, col, rec))

    assigned = {}
    used = set()
    for _dist, _neg_score, row, col, rec in sorted(pairs):
        cell = (row, col)
        if cell in assigned or rec["idx"] in used:
            continue
        assigned[cell] = rec
        used.add(rec["idx"])
    return assigned


def _fit_poly_predictor(samples, max_degree: int = 2):
    """Fit independent x/y polynomials. samples are (t, x, y)."""
    if len(samples) < 2:
        return None
    degree = min(max_degree, len(samples) - 1)
    ts = np.asarray([s[0] for s in samples], dtype=np.float64)
    xs = np.asarray([s[1] for s in samples], dtype=np.float64)
    ys = np.asarray([s[2] for s in samples], dtype=np.float64)
    try:
        px = np.polyfit(ts, xs, degree)
        py = np.polyfit(ts, ys, degree)
    except Exception:
        return None

    def predict(t):
        return np.array([float(np.polyval(px, t)), float(np.polyval(py, t))], dtype=np.float64)

    return predict


def _smooth_lattice_predictions(corners, assigned, cols: int, rows: int):
    row_models = {}
    col_models = {}

    for row in range(rows):
        samples = [
            (float(col), float(rec["x"]), float(rec["y"]))
            for (r, col), rec in assigned.items()
            if r == row
        ]
        model = _fit_poly_predictor(samples)
        if model is not None:
            row_models[row] = model

    for col in range(cols):
        samples = [
            (float(row), float(rec["x"]), float(rec["y"]))
            for (row, c), rec in assigned.items()
            if c == col
        ]
        model = _fit_poly_predictor(samples)
        if model is not None:
            col_models[col] = model

    predictions = {}
    for row in range(rows):
        for col in range(cols):
            pts = []
            sources = []
            if row in row_models:
                pts.append(row_models[row](float(col)))
                sources.append("row")
            if col in col_models:
                pts.append(col_models[col](float(row)))
                sources.append("col")
            if pts:
                pred = np.mean(np.vstack(pts), axis=0)
                source = "+".join(sources)
            else:
                pred = np.asarray(_bilinear_grid_point(corners, col, row, cols, rows), dtype=np.float64)
                source = "corner_grid"
            predictions[(row, col)] = {"xy": pred, "source": source}
    return predictions


def _assign_candidates_by_lattice(records, predictions, cols: int, rows: int, max_px_dist: float):
    pairs = []
    for cell, pred in predictions.items():
        row, col = cell
        px, py = pred["xy"]
        for rec in records:
            grid_dist = math.hypot(rec["grid_x"] - float(col), rec["grid_y"] - float(row))
            if grid_dist > 0.78:
                continue
            dist = math.hypot(rec["x"] - px, rec["y"] - py)
            if dist <= max_px_dist:
                pairs.append((dist, grid_dist, -rec["score"], row, col, rec))

    assigned = {}
    used = set()
    for _dist, _grid_dist, _neg_score, row, col, rec in sorted(pairs):
        cell = (row, col)
        if cell in assigned or rec["idx"] in used:
            continue
        assigned[cell] = rec
        used.add(rec["idx"])
    return assigned


def _prefer_larger_same_cell_candidates(assigned, records, predictions, cols: int, rows: int, max_px_dist: float):
    """Replace tiny same-cell detections with much larger nearby detections."""
    if not assigned:
        return assigned, 0
    updated = dict(assigned)
    used = {int(rec["idx"]) for rec in updated.values()}
    replacements = 0
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            current = updated.get(cell)
            pred = predictions.get(cell)
            if current is None or pred is None:
                continue
            px, py = pred["xy"]
            current_area = max(float(current.get("area", 0.0)), 1.0)
            current_dist = math.hypot(current["x"] - float(px), current["y"] - float(py))
            current_quality = current_area / max(1.0 + current_dist, 1.0)
            best = None
            best_quality = current_quality
            for rec in records:
                if int(rec["idx"]) in used:
                    continue
                grid_dist = math.hypot(rec["grid_x"] - float(col), rec["grid_y"] - float(row))
                if grid_dist > 0.92:
                    continue
                dist = math.hypot(rec["x"] - float(px), rec["y"] - float(py))
                if dist > max_px_dist * 1.12:
                    continue
                area = max(float(rec.get("area", 0.0)), 1.0)
                if area < max(current_area * 2.2, current_area + 18.0):
                    continue
                if float(rec.get("score", 0.0)) < float(current.get("score", 0.0)) * 0.45:
                    continue
                if dist > current_dist + max_px_dist * 0.42:
                    continue
                quality = area / max(1.0 + dist, 1.0)
                if quality > best_quality * 1.25:
                    best = rec
                    best_quality = quality
            if best is not None:
                used.discard(int(current["idx"]))
                used.add(int(best["idx"]))
                updated[cell] = best
                replacements += 1
    return updated, replacements


def _extract_orange_components(frame_bgr: np.ndarray,
                               search_points=None,
                               margin: int = 0,
                               grid_step: float = 40.0):
    """Segment complete painted marker bodies and find safe interior prompts."""
    h_img, w_img = frame_bgr.shape[:2]
    if search_points:
        xs = [float(point[0]) for point in search_points]
        ys = [float(point[1]) for point in search_points]
        pad = max(4, int(margin))
        x0 = max(0, int(math.floor(min(xs))) - pad)
        y0 = max(0, int(math.floor(min(ys))) - pad)
        x1 = min(w_img, int(math.ceil(max(xs))) + pad + 1)
        y1 = min(h_img, int(math.ceil(max(ys))) + pad + 1)
    else:
        x0, y0, x1, y1 = 0, 0, w_img, h_img

    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    _l, a, b = cv2.split(lab)

    strong_hue = ((h >= 4) & (h <= 42)) | (h >= 172)
    strong_mask = (
        strong_hue
        & (s >= 65)
        & (v >= 45)
        & (b >= 135)
        & (a >= 118)
    )

    # Grow strong orange seeds through dimmer red/orange paint. This keeps the
    # full marker body while excluding isolated green highlights and dark net.
    weak_hue = ((h <= 50) | (h >= 168))
    weak_mask = (
        weak_hue
        & (s >= 34)
        & (v >= 24)
        & (b >= 122)
        & (a >= 108)
    ).astype(np.uint8) * 255
    strong_u8 = strong_mask.astype(np.uint8) * 255
    weak_mask = cv2.medianBlur(weak_mask, 3)
    strong_u8 = cv2.medianBlur(strong_u8, 3)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    weak_mask = cv2.morphologyEx(weak_mask, cv2.MORPH_OPEN, kernel3, iterations=1)

    close_size = int(round(max(3.0, min(13.0, float(grid_step) * 0.035))))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    weak_mask = cv2.morphologyEx(
        weak_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    weak_count, weak_labels, weak_stats, _weak_centroids = cv2.connectedComponentsWithStats(
        (weak_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    mask = np.zeros_like(weak_mask)
    for label in range(1, weak_count):
        component_pixels = weak_labels == label
        area = int(weak_stats[label, cv2.CC_STAT_AREA])
        strong_overlap = int(np.count_nonzero(strong_u8[component_pixels]))
        if area >= 10 and strong_overlap >= max(3, int(round(area * 0.008))):
            mask[component_pixels] = 255

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    components = []
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 18.0:
            continue
        lx, ly, lw, lh = [float(vv) for vv in stats[label, :4]]
        aspect = max(lw / max(lh, 1.0), lh / max(lw, 1.0))
        extent = area / max(lw * lh, 1.0)
        if aspect > 6.0 or extent < 0.09:
            continue
        cx, cy = centroids[label]
        component_mask = (labels == label).astype(np.uint8)
        distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        _min_value, interior_radius, _min_loc, interior_xy = cv2.minMaxLoc(distance)
        interior_x, interior_y = interior_xy
        components.append(
            {
                "component_id": int(label),
                "x": float(x0) + float(interior_x),
                "y": float(y0) + float(interior_y),
                "centroid_x": float(x0) + float(cx),
                "centroid_y": float(y0) + float(cy),
                "area": area,
                "extent": float(extent),
                "interior_radius": float(interior_radius),
                "bbox": (
                    float(x0) + lx,
                    float(y0) + ly,
                    float(lw),
                    float(lh),
                ),
            }
        )
    return components


def _estimate_grid_corners_from_detection(
    side: str,
    frame_bgr: np.ndarray,
    cols: int,
    rows: int,
    allow_fragmented: bool = False,
):
    """Estimate TL/TR/BL/BR marker centers from the dominant orange lattice."""
    expected = int(cols * rows)
    min_component_count = min(expected, max(4, int(round(expected * 0.45))))
    min_kept_count = min(expected, max(4, int(round(expected * 0.40))))
    min_group_count = min(expected, max(4, int(round(expected * 0.55))))
    h_img, w_img = frame_bgr.shape[:2]
    rough_step = max(
        24.0,
        min(
            float(w_img) / float(max(cols + 4, 1)),
            float(h_img) / float(max(rows + 4, 1)),
        ),
    )
    components = _extract_orange_components(frame_bgr, grid_step=rough_step)
    if len(components) < min_component_count:
        raise RuntimeError(f"Only {len(components)} orange bodies were found.")

    areas = np.asarray([float(component["area"]) for component in components], dtype=np.float64)
    area_cap = float(np.percentile(areas, 95.0))
    typical_areas = areas[(areas >= 18.0) & (areas <= area_cap)]
    median_area = float(np.median(typical_areas)) if len(typical_areas) else float(np.median(areas))
    min_area = max(18.0, median_area * 0.16)
    max_area = max(min_area + 1.0, median_area * 5.0)
    kept_indices = [
        idx
        for idx, component in enumerate(components)
        if min_area <= float(component["area"]) <= max_area
    ]
    if len(kept_indices) < min_kept_count:
        raise RuntimeError("Too few consistently sized orange bodies remained after filtering.")

    xy = np.asarray(
        [
            [
                float(components[idx].get("centroid_x", components[idx]["x"])),
                float(components[idx].get("centroid_y", components[idx]["y"])),
            ]
            for idx in kept_indices
        ],
        dtype=np.float64,
    )
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    max_reasonable_step = max(80.0, float(min(h_img, w_img)) * 0.12)
    valid_nearest = nearest[(nearest >= 12.0) & (nearest <= max_reasonable_step)]
    if len(valid_nearest) < min(8, max(3, len(xy) - 1)):
        raise RuntimeError("Could not estimate consistent marker spacing.")
    typical_step = float(np.median(valid_nearest))
    connection_radius = max(
        typical_step * 1.30,
        min(typical_step * 1.55, float(min(h_img, w_img)) * 0.075),
    )

    unseen = set(range(len(xy)))
    groups = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        group = [seed]
        while stack:
            current = stack.pop()
            neighbors = np.flatnonzero(distances[current] <= connection_radius)
            for neighbor_value in neighbors:
                neighbor = int(neighbor_value)
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                stack.append(neighbor)
                group.append(neighbor)
        groups.append(group)

    group = max(groups, key=len)
    fragmented_group_fallback = False
    if len(group) < min_group_count:
        if not allow_fragmented:
            raise RuntimeError(
                f"The largest regular marker group contains only {len(group)}/{expected} bodies."
            )
        group = list(range(len(xy)))
        fragmented_group_fallback = True
    group_xy = xy[group]
    group_component_indices = np.asarray(kept_indices, dtype=np.int32)[group]

    top_limit = float(np.percentile(group_xy[:, 1], 25.0))
    top_pool = np.flatnonzero(group_xy[:, 1] <= top_limit)
    if len(top_pool) < 2:
        top_limit = float(np.percentile(group_xy[:, 1], 40.0))
        top_pool = np.flatnonzero(group_xy[:, 1] <= top_limit)
    hull_indices = cv2.convexHull(
        group_xy.astype(np.float32),
        returnPoints=False,
    ).reshape(-1)
    lower_limit = float(np.percentile(group_xy[:, 1], 50.0))
    lower_hull = hull_indices[group_xy[hull_indices, 1] >= lower_limit]
    if len(lower_hull) < 2:
        lower_limit = float(np.percentile(group_xy[:, 1], 35.0))
        lower_hull = hull_indices[group_xy[hull_indices, 1] >= lower_limit]
    if len(top_pool) < 2 or len(lower_hull) < 2:
        raise RuntimeError("Could not identify both upper and lower lattice boundaries.")

    top_left_idx = int(top_pool[np.argmin(group_xy[top_pool, 0])])
    top_right_idx = int(top_pool[np.argmax(group_xy[top_pool, 0])])
    bottom_left_idx = int(lower_hull[np.argmin(group_xy[lower_hull, 0])])
    bottom_right_idx = int(lower_hull[np.argmax(group_xy[lower_hull, 0])])

    all_centroids = np.asarray(
        [
            [
                float(component.get("centroid_x", component["x"])),
                float(component.get("centroid_y", component["y"])),
            ]
            for component in components
        ],
        dtype=np.float64,
    )

    def extrapolate_and_snap(group_idx: int, inward_x_sign: float):
        boundary = group_xy[group_idx]
        delta = group_xy - boundary
        inward_dx = delta[:, 0] * float(inward_x_sign)
        valid = np.flatnonzero(
            (delta[:, 1] > typical_step * 0.70)
            & (delta[:, 1] < typical_step * 2.50)
            & (inward_dx > -typical_step * 0.35)
            & (inward_dx < typical_step * 1.20)
        )
        if not len(valid):
            return int(group_component_indices[group_idx]), False
        scores = np.linalg.norm(delta[valid], axis=1) + np.maximum(0.0, -inward_dx[valid]) * 2.0
        next_point = group_xy[int(valid[int(np.argmin(scores))])]
        predicted = boundary - (next_point - boundary) * 1.12
        snap_distances = np.linalg.norm(all_centroids - predicted, axis=1)
        plausible = np.flatnonzero(
            (all_centroids[:, 1] < boundary[1] - typical_step * 0.35)
            & (snap_distances <= typical_step * 0.62)
            & (areas >= min_area)
            & (areas <= max_area)
        )
        if not len(plausible):
            return int(group_component_indices[group_idx]), False
        best = int(plausible[int(np.argmin(snap_distances[plausible]))])
        return best, True

    top_left_component, top_left_extrapolated = extrapolate_and_snap(top_left_idx, 1.0)
    top_right_component, top_right_extrapolated = extrapolate_and_snap(top_right_idx, -1.0)
    selected_components = [
        top_left_component,
        top_right_component,
        int(group_component_indices[bottom_left_idx]),
        int(group_component_indices[bottom_right_idx]),
    ]
    if len(set(selected_components)) != 4:
        raise RuntimeError("Automatic corner estimation selected duplicate marker bodies.")

    corners = [
        (float(components[idx]["x"]), float(components[idx]["y"]))
        for idx in selected_components
    ]
    tl, tr, bl, br = [np.asarray(point, dtype=np.float64) for point in corners]
    top_width = float(np.linalg.norm(tr - tl))
    bottom_width = float(np.linalg.norm(br - bl))
    left_height = float(np.linalg.norm(bl - tl))
    right_height = float(np.linalg.norm(br - tr))
    if not (
        tl[0] < tr[0]
        and bl[0] < br[0]
        and (tl[1] + tr[1]) < (bl[1] + br[1])
        and min(top_width, bottom_width) >= typical_step * max(cols - 1, 1) * 0.42
        and min(left_height, right_height) >= typical_step * max(rows - 1, 1) * 0.42
    ):
        raise RuntimeError("Estimated corners did not form a plausible grid quadrilateral.")

    print(
        f"[INFO][{side.upper()}] auto corners: components={len(components)} "
        f"regular_group={len(group)} spacing={typical_step:.1f}px "
        f"fragmented_fallback={int(fragmented_group_fallback)} "
        f"top_extrapolated={int(top_left_extrapolated) + int(top_right_extrapolated)}/2"
    )
    meta = [
        {
            "obj_id": idx,
            "source": "anchor",
            "corner": name,
            "x": float(point[0]),
            "y": float(point[1]),
            "auto_estimated": True,
        }
        for idx, (name, point) in enumerate(zip(("TL", "TR", "BL", "BR"), corners))
    ]
    return [_make_prompt_point(x, y, 1) for x, y in corners], meta


def _assign_orange_components_globally(
    cell_states,
    components,
    search_radius: int,
    to_grid=None,
):
    """Match cells to dominant orange bodies one-to-one."""
    if not cell_states or not components:
        return {}

    radius = float(max(18, search_radius))
    invalid_cost = 1_000_000.0
    dummy_cost = 1.12
    component_grid_xy = None
    if to_grid is not None:
        try:
            component_xy = np.asarray(
                [
                    [
                        float(component.get("centroid_x", component["x"])),
                        float(component.get("centroid_y", component["y"])),
                    ]
                    for component in components
                ],
                dtype=np.float32,
            )
            component_grid_xy = _apply_homography(component_xy, to_grid)
        except Exception:
            component_grid_xy = None

    costs = np.full(
        (len(cell_states), len(components) + len(cell_states)),
        invalid_cost,
        dtype=np.float64,
    )
    costs[:, len(components):] = dummy_cost

    for cell_idx, state in enumerate(cell_states):
        px, py = state["pred_xy"]
        cx, cy = state["current_xy"]
        nearby = []
        for comp_idx, component in enumerate(components):
            center_x = float(component.get("centroid_x", component["x"]))
            center_y = float(component.get("centroid_y", component["y"]))
            dist_to_pred = math.hypot(center_x - px, center_y - py)
            dist_to_current = math.hypot(center_x - cx, center_y - cy)
            grid_dist = None
            grid_col_err = None
            grid_row_err = None
            if component_grid_xy is not None:
                gx, gy = component_grid_xy[comp_idx]
                grid_col_err = abs(float(gx) - float(state["col"]))
                grid_row_err = abs(float(gy) - float(state["row"]))
                grid_dist = math.hypot(grid_col_err, grid_row_err)
            area = float(component["area"])
            if component_grid_xy is None:
                if dist_to_pred > radius or dist_to_current > radius * 1.45:
                    continue
            else:
                near_grid_cell = (
                    grid_dist is not None
                    and grid_col_err <= 0.92
                    and grid_row_err <= 0.74
                    and grid_dist <= 1.02
                )
                near_lattice = dist_to_pred <= radius and dist_to_current <= radius * 1.70
                if not near_grid_cell and not near_lattice:
                    continue
            nearby.append((comp_idx, component, dist_to_pred, dist_to_current, grid_dist))

        if not nearby:
            continue
        dominant_area = max(float(item[1]["area"]) for item in nearby)
        for comp_idx, component, dist_to_pred, dist_to_current, grid_dist in nearby:
            area = float(component["area"])
            if area < max(18.0, dominant_area * 0.18):
                continue
            area_ratio = max(min(area / max(dominant_area, 1.0), 1.0), 1e-6)
            area_penalty = min(1.0, -math.log(area_ratio) / math.log(6.0))
            equivalent_radius = math.sqrt(max(area, 1.0) / math.pi)
            interior_ratio = min(
                1.0,
                float(component.get("interior_radius", 0.0))
                / max(equivalent_radius, 1.0),
            )
            if component_grid_xy is None:
                costs[cell_idx, comp_idx] = (
                    0.56 * (dist_to_pred / radius)
                    + 0.08 * (dist_to_current / (radius * 1.45))
                    + 0.62 * area_penalty
                    + 0.08 * (1.0 - interior_ratio)
                )
            else:
                grid_term = 1.0
                grid_term = min(1.0, float(grid_dist) / 0.88)
                lattice_term = min(1.0, float(dist_to_pred) / radius)
                current_term = min(1.0, float(dist_to_current) / (radius * 1.70))
                if grid_dist is not None and grid_dist <= 0.62:
                    costs[cell_idx, comp_idx] = (
                        0.54 * grid_term
                        + 0.18 * lattice_term
                        + 0.04 * current_term
                        + 0.34 * area_penalty
                        + 0.05 * (1.0 - interior_ratio)
                    )
                else:
                    costs[cell_idx, comp_idx] = (
                        0.50 * lattice_term
                        + 0.10 * current_term
                        + 0.18 * grid_term
                        + 0.48 * area_penalty
                        + 0.06 * (1.0 - interior_ratio)
                    )

    if linear_sum_assignment is not None:
        row_indices, col_indices = linear_sum_assignment(costs)
        pairs = zip(row_indices.tolist(), col_indices.tolist())
    else:
        candidates = []
        for row in range(len(cell_states)):
            for col in range(len(components)):
                if costs[row, col] < dummy_cost:
                    candidates.append((float(costs[row, col]), row, col))
        used_rows = set()
        used_cols = set()
        greedy_pairs = []
        for _cost, row, col in sorted(candidates):
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            greedy_pairs.append((row, col))
        pairs = greedy_pairs

    assignments = {}
    for row, col in pairs:
        if col >= len(components) or costs[row, col] >= dummy_cost:
            continue
        state = cell_states[row]
        component = dict(components[col])
        component["cost"] = float(costs[row, col])
        if component_grid_xy is not None:
            gx, gy = component_grid_xy[col]
            component["grid_x"] = float(gx)
            component["grid_y"] = float(gy)
            component["grid_dist"] = math.hypot(float(gx) - float(state["col"]), float(gy) - float(state["row"]))
        assignments[state["cell"]] = component
    return assignments


def _assign_orange_components_by_rows(cell_states, components, cols: int, rows: int, to_grid, search_radius: int):
    """Row-local assignment using grid coordinates and left-to-right order."""
    if not cell_states or not components or to_grid is None or cols < 2 or rows < 2:
        return {}
    try:
        component_xy = np.asarray(
            [
                [
                    float(component.get("centroid_x", component["x"])),
                    float(component.get("centroid_y", component["y"])),
                ]
                for component in components
            ],
            dtype=np.float32,
        )
        component_grid_xy = _apply_homography(component_xy, to_grid)
    except Exception:
        return {}

    state_by_cell = {tuple(state["cell"]): state for state in cell_states}
    assignments = {}
    used_components = set()
    radius = float(max(18, search_radius))
    dummy_cost = 1.05
    invalid_cost = 1_000_000.0

    for row in range(rows):
        row_states = [state_by_cell.get((row, col)) for col in range(cols)]
        if any(state is None for state in row_states):
            continue
        row_components = []
        for comp_idx, (component, (gx, gy)) in enumerate(zip(components, component_grid_xy)):
            if comp_idx in used_components:
                continue
            gx = float(gx)
            gy = float(gy)
            row_err = abs(gy - float(row))
            if row_err > 0.78:
                continue
            if gx < -0.88 or gx > float(cols - 1) + 0.88:
                continue
            nearest_col = int(round(gx))
            if nearest_col < 0 or nearest_col >= cols:
                continue
            state = state_by_cell[(row, nearest_col)]
            center_x = float(component.get("centroid_x", component["x"]))
            center_y = float(component.get("centroid_y", component["y"]))
            pred_x, pred_y = state["pred_xy"]
            if math.hypot(center_x - pred_x, center_y - pred_y) > radius * 2.05 and row_err > 0.46:
                continue
            row_components.append((comp_idx, component, gx, gy))

        if len(row_components) < max(2, min(cols, int(math.ceil(cols * 0.55)))):
            continue

        costs = np.full((cols, len(row_components) + cols), invalid_cost, dtype=np.float64)
        costs[:, len(row_components):] = dummy_cost
        dominant_area = max(float(item[1]["area"]) for item in row_components)
        for col in range(cols):
            state = state_by_cell[(row, col)]
            pred_x, pred_y = state["pred_xy"]
            for local_idx, (_comp_idx, component, gx, gy) in enumerate(row_components):
                col_err = abs(float(gx) - float(col))
                row_err = abs(float(gy) - float(row))
                if col_err > 0.90 or row_err > 0.78:
                    continue
                center_x = float(component.get("centroid_x", component["x"]))
                center_y = float(component.get("centroid_y", component["y"]))
                pred_dist = math.hypot(center_x - pred_x, center_y - pred_y)
                area = float(component["area"])
                if area < max(18.0, dominant_area * 0.12):
                    continue
                area_ratio = max(min(area / max(dominant_area, 1.0), 1.0), 1e-6)
                area_penalty = min(1.0, -math.log(area_ratio) / math.log(8.0))
                costs[col, local_idx] = (
                    0.62 * min(1.0, col_err / 0.90)
                    + 0.40 * min(1.0, row_err / 0.78)
                    + 0.10 * min(1.0, pred_dist / max(radius, 1.0))
                    + 0.22 * area_penalty
                )

        if linear_sum_assignment is not None:
            row_indices, col_indices = linear_sum_assignment(costs)
            pairs = zip(row_indices.tolist(), col_indices.tolist())
        else:
            candidates = []
            for cost_row in range(cols):
                for cost_col in range(len(row_components)):
                    if costs[cost_row, cost_col] < dummy_cost:
                        candidates.append((float(costs[cost_row, cost_col]), cost_row, cost_col))
            seen_rows = set()
            seen_cols = set()
            greedy = []
            for _cost, cost_row, cost_col in sorted(candidates):
                if cost_row in seen_rows or cost_col in seen_cols:
                    continue
                seen_rows.add(cost_row)
                seen_cols.add(cost_col)
                greedy.append((cost_row, cost_col))
            pairs = greedy

        row_assignments = {}
        for col, local_idx in pairs:
            if local_idx >= len(row_components) or costs[col, local_idx] >= dummy_cost:
                continue
            comp_idx, component, gx, gy = row_components[local_idx]
            component = dict(component)
            component["cost"] = float(costs[col, local_idx])
            component["grid_x"] = float(gx)
            component["grid_y"] = float(gy)
            component["grid_dist"] = math.hypot(float(gx) - float(col), float(gy) - float(row))
            component["row_ordered"] = True
            row_assignments[(row, col)] = (comp_idx, component)

        if len(row_assignments) < max(2, min(cols, int(math.ceil(cols * 0.50)))):
            continue
        for cell, (comp_idx, component) in row_assignments.items():
            assignments[cell] = component
            used_components.add(comp_idx)

    return assignments


def _assign_orange_components_by_elastic_rows(cell_states, components, cols: int, rows: int, to_grid, search_radius: int):
    """Assign orange bodies by fitting each row as a curved chain.

    This handles slack/curved net rows better than a straight homography band.
    Grid coordinates provide the rough row/column order; fitted image-space row
    curves handle the visible sag between anchors.
    """
    if not cell_states or not components or to_grid is None or cols < 2 or rows < 2:
        return {}
    try:
        component_xy = np.asarray(
            [
                [
                    float(component.get("centroid_x", component["x"])),
                    float(component.get("centroid_y", component["y"])),
                ]
                for component in components
            ],
            dtype=np.float32,
        )
        component_grid_xy = _apply_homography(component_xy, to_grid)
    except Exception:
        return {}

    state_by_cell = {tuple(state["cell"]): state for state in cell_states}
    component_records = []
    for comp_idx, (component, (gx, gy)) in enumerate(zip(components, component_grid_xy)):
        gx = float(gx)
        gy = float(gy)
        if gx < -1.10 or gx > float(cols - 1) + 1.10 or gy < -1.10 or gy > float(rows - 1) + 1.10:
            continue
        component_records.append(
            {
                "idx": int(comp_idx),
                "component": component,
                "gx": gx,
                "gy": gy,
                "x": float(component.get("centroid_x", component["x"])),
                "y": float(component.get("centroid_y", component["y"])),
            }
        )
    if not component_records:
        return {}

    row_models = {}
    for row in range(rows):
        row_records = [
            rec for rec in component_records
            if abs(float(rec["gy"]) - float(row)) <= 1.55
        ]
        if len(row_records) < 2:
            continue
        # Prefer the bodies closest to this row, but keep enough for a curve.
        row_records = sorted(
            row_records,
            key=lambda rec: (
                0.70 * abs(float(rec["gy"]) - float(row))
                + 0.30 * abs(float(rec["gx"]) - round(float(rec["gx"]))),
                abs(float(rec["gx"]) - round(float(rec["gx"]))),
            ),
        )[:max(cols + 5, min(len(row_records), cols * 3))]
        samples = [(float(rec["gx"]), float(rec["x"]), float(rec["y"])) for rec in row_records]
        model = _fit_poly_predictor(samples, max_degree=2)
        if model is not None:
            row_models[row] = model

    radius = float(max(18, search_radius))
    dummy_cost = 1.08
    invalid_cost = 1_000_000.0
    candidate_assignments = {}
    median_area = float(np.median([float(component["area"]) for component in components])) if components else 1.0

    for row in range(rows):
        row_states = [state_by_cell.get((row, col)) for col in range(cols)]
        if any(state is None for state in row_states):
            continue
        model = row_models.get(row)
        row_candidates = []
        for rec in component_records:
            gx = float(rec["gx"])
            gy = float(rec["gy"])
            row_err = abs(gy - float(row))
            nearest_col = int(round(gx))
            if nearest_col < -1 or nearest_col > cols:
                continue
            col_proximity = min(abs(gx - float(col)) for col in range(cols))
            if row_err > 1.65 or col_proximity > 1.15:
                continue
            curve_dist = None
            if model is not None:
                curve_xy = model(float(gx))
                curve_dist = math.hypot(float(rec["x"]) - float(curve_xy[0]), float(rec["y"]) - float(curve_xy[1]))
                if curve_dist > radius * 1.70 and row_err > 0.96:
                    continue
            row_candidates.append((rec, curve_dist))

        if len(row_candidates) < 2:
            continue

        costs = np.full((cols, len(row_candidates) + cols), invalid_cost, dtype=np.float64)
        costs[:, len(row_candidates):] = dummy_cost
        for col in range(cols):
            state = state_by_cell[(row, col)]
            pred_x, pred_y = state["pred_xy"]
            curve_xy_at_col = model(float(col)) if model is not None else np.asarray([pred_x, pred_y], dtype=np.float64)
            for local_idx, (rec, curve_dist) in enumerate(row_candidates):
                component = rec["component"]
                gx = float(rec["gx"])
                gy = float(rec["gy"])
                col_err = abs(gx - float(col))
                row_err = abs(gy - float(row))
                if col_err > 1.12 or row_err > 1.65:
                    continue
                center_x = float(rec["x"])
                center_y = float(rec["y"])
                pred_dist = math.hypot(center_x - pred_x, center_y - pred_y)
                col_curve_dist = math.hypot(center_x - float(curve_xy_at_col[0]), center_y - float(curve_xy_at_col[1]))
                curve_term_dist = curve_dist if curve_dist is not None else col_curve_dist
                area = max(float(component["area"]), 1.0)
                area_ratio = max(min(area / max(median_area, 1.0), 2.0), 1e-6)
                small_area_penalty = max(0.0, -math.log(min(area_ratio, 1.0)) / math.log(6.0))
                costs[col, local_idx] = (
                    0.34 * min(1.0, col_err / 1.12)
                    + 0.12 * min(1.0, row_err / 1.65)
                    + 0.42 * min(1.0, col_curve_dist / max(radius * 0.92, 1.0))
                    + 0.03 * min(1.0, pred_dist / max(radius * 2.25, 1.0))
                    + 0.34 * min(1.0, curve_term_dist / max(radius * 1.25, 1.0))
                    + 0.24 * small_area_penalty
                )

        if linear_sum_assignment is not None:
            row_indices, col_indices = linear_sum_assignment(costs)
            pairs = zip(row_indices.tolist(), col_indices.tolist())
        else:
            candidates = []
            for cost_row in range(cols):
                for cost_col in range(len(row_candidates)):
                    if costs[cost_row, cost_col] < dummy_cost:
                        candidates.append((float(costs[cost_row, cost_col]), cost_row, cost_col))
            used_rows = set()
            used_cols = set()
            greedy = []
            for _cost, cost_row, cost_col in sorted(candidates):
                if cost_row in used_rows or cost_col in used_cols:
                    continue
                used_rows.add(cost_row)
                used_cols.add(cost_col)
                greedy.append((cost_row, cost_col))
            pairs = greedy

        for col, local_idx in pairs:
            if local_idx >= len(row_candidates) or costs[col, local_idx] >= dummy_cost:
                continue
            rec, _curve_dist = row_candidates[local_idx]
            component = dict(rec["component"])
            component["cost"] = float(costs[col, local_idx])
            component["grid_x"] = float(rec["gx"])
            component["grid_y"] = float(rec["gy"])
            component["grid_dist"] = math.hypot(float(rec["gx"]) - float(col), float(rec["gy"]) - float(row))
            component["row_ordered"] = True
            component["elastic_row"] = True
            candidate_assignments[(row, col)] = (int(rec["idx"]), component)

    # Resolve conflicts where one orange body was plausible for adjacent rows.
    best_by_component = {}
    for cell, (comp_idx, component) in candidate_assignments.items():
        previous = best_by_component.get(comp_idx)
        if previous is None or float(component["cost"]) < float(previous[1]["cost"]):
            best_by_component[comp_idx] = (cell, component)

    assignments = {}
    for _comp_idx, (cell, component) in best_by_component.items():
        assignments[cell] = component
    return assignments


def _anchor_cells_from_corners(corners, cols: int, rows: int):
    if cols < 2 or rows < 2:
        return {}
    return {
        (0, 0): np.asarray(corners[0], dtype=np.float64),
        (0, cols - 1): np.asarray(corners[1], dtype=np.float64),
        (rows - 1, 0): np.asarray(corners[2], dtype=np.float64),
        (rows - 1, cols - 1): np.asarray(corners[3], dtype=np.float64),
    }


def _generate_grid_prompts_from_detection(
    side: str,
    frame_bgr: np.ndarray,
    corners,
    cols: int,
    rows: int,
    hole_net_mode: bool = False,
):
    expected = int(cols * rows)
    to_grid, _to_image = _grid_homographies(corners, cols, rows)
    candidates, detector_meta = _detect_marker_candidates(side, frame_bgr, expected)
    records = _candidate_grid_records(candidates, to_grid, cols, rows)
    grid_step = _grid_step_from_corners(corners, cols, rows)
    anchor_cells = _anchor_cells_from_corners(corners, cols, rows)

    assigned = _assign_candidates_by_grid(records, cols, rows, max_grid_dist=0.40)
    predictions = _smooth_lattice_predictions(corners, assigned, cols, rows)
    max_px_dist = max(10.0, min(55.0, grid_step * 0.42))
    for _iteration in range(2):
        refined = _assign_candidates_by_lattice(records, predictions, cols, rows, max_px_dist=max_px_dist)
        if len(refined) < max(4, len(assigned) // 2):
            break
        assigned = refined
        predictions = _smooth_lattice_predictions(corners, assigned, cols, rows)
    assigned, large_blob_replacements = _prefer_larger_same_cell_candidates(
        assigned,
        records,
        predictions,
        cols,
        rows,
        max_px_dist=max_px_dist,
    )
    if large_blob_replacements:
        predictions = _smooth_lattice_predictions(corners, assigned, cols, rows)

    detected_count = 0
    lattice_count = 0
    anchor_count = 0
    orange_search_radius = int(round(max(32.0, min(130.0, grid_step * 0.44))))
    cell_states = []

    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            pred = predictions.get(cell)
            pred_xy = pred["xy"] if pred is not None else np.asarray(_bilinear_grid_point(corners, col, row, cols, rows), dtype=np.float64)
            rec = assigned.get(cell)
            is_anchor = cell in anchor_cells
            if is_anchor:
                x, y = float(anchor_cells[cell][0]), float(anchor_cells[cell][1])
                source = "anchor"
                confidence = 1.0
                anchor_count += 1
            elif rec is not None:
                candidate_dist = math.hypot(rec["x"] - float(pred_xy[0]), rec["y"] - float(pred_xy[1]))
                if candidate_dist <= max_px_dist * 0.72:
                    x, y = float(rec["x"]), float(rec["y"])
                    source = "detected_lattice"
                    confidence = float(rec["score"])
                    detected_count += 1
                else:
                    x, y = float(pred_xy[0]), float(pred_xy[1])
                    source = "lattice"
                    confidence = max(float(rec["score"]) * 0.5, 0.0)
                    lattice_count += 1
            elif pred is not None:
                x, y = float(pred_xy[0]), float(pred_xy[1])
                if pred["source"] == "corner_grid":
                    source = "inferred"
                else:
                    source = "lattice"
                    lattice_count += 1
                confidence = 0.0
            else:
                x, y = _bilinear_grid_point(corners, col, row, cols, rows)
                source = "inferred"
                confidence = 0.0

            if rec is not None:
                candidate_dx = float(rec["grid_x"] - float(col))
                candidate_dy = float(rec["grid_y"] - float(row))
                candidate_grid_dist = math.hypot(candidate_dx, candidate_dy)
                candidate_px_dist = math.hypot(rec["x"] - float(pred_xy[0]), rec["y"] - float(pred_xy[1]))
            else:
                candidate_grid_dist = None
                candidate_px_dist = None

            if source == "detected_lattice":
                source = "detected"

            cell_states.append(
                {
                    "cell": cell,
                    "row": row,
                    "col": col,
                    "pred_xy": (float(pred_xy[0]), float(pred_xy[1])),
                    "current_xy": (float(x), float(y)),
                    "current_area": float(rec.get("area", 1.0)) if rec is not None else 1.0,
                    "source": source,
                    "confidence": confidence,
                    "lattice_source": pred["source"] if pred is not None else "corner_grid",
                    "candidate_grid_dist": candidate_grid_dist,
                    "candidate_lattice_dist_px": candidate_px_dist,
                    "candidate_area": float(rec.get("area", 0.0)) if rec is not None else None,
                    "anchor_locked": bool(is_anchor),
                }
            )

    refinable_states = [state for state in cell_states if not state["anchor_locked"]]
    orange_components = _extract_orange_components(
        frame_bgr,
        search_points=[state["pred_xy"] for state in refinable_states],
        margin=orange_search_radius,
        grid_step=grid_step,
    )
    orange_assignments = _assign_orange_components_globally(
        refinable_states,
        orange_components,
        orange_search_radius,
        to_grid=to_grid if hole_net_mode else None,
    )
    row_ordered_assignments = {}
    if hole_net_mode:
        row_ordered_assignments = _assign_orange_components_by_rows(
            refinable_states,
            orange_components,
            cols,
            rows,
            to_grid,
            orange_search_radius,
        )
        elastic_row_assignments = _assign_orange_components_by_elastic_rows(
            refinable_states,
            orange_components,
            cols,
            rows,
            to_grid,
            orange_search_radius,
        )
        for cell, elastic_refine in elastic_row_assignments.items():
            current_refine = row_ordered_assignments.get(cell)
            use_elastic = current_refine is None
            if current_refine is not None:
                elastic_grid_dist = float(elastic_refine.get("grid_dist", 99.0))
                current_grid_dist = float(current_refine.get("grid_dist", 99.0))
                elastic_cost = float(elastic_refine.get("cost", 99.0))
                current_cost = float(current_refine.get("cost", 99.0))
                if elastic_grid_dist <= current_grid_dist + 0.34 and elastic_cost <= current_cost + 0.30:
                    use_elastic = True
                elif elastic_grid_dist <= 0.72 and current_grid_dist > 0.72:
                    use_elastic = True
                elif elastic_grid_dist <= 0.58 and elastic_cost <= current_cost + 0.42:
                    use_elastic = True
            if use_elastic:
                row_ordered_assignments[cell] = elastic_refine

        best_row_assignment_by_component = {}
        componentless_row_assignments = {}
        for cell, refine in row_ordered_assignments.items():
            component_id = int(refine.get("component_id", -1))
            if component_id < 0:
                componentless_row_assignments[cell] = refine
                continue
            score = (
                0 if bool(refine.get("elastic_row", False)) else 1,
                float(refine.get("cost", 99.0)),
                float(refine.get("grid_dist", 99.0)),
            )
            previous = best_row_assignment_by_component.get(component_id)
            if previous is None or score < previous[0]:
                best_row_assignment_by_component[component_id] = (score, cell, refine)
        row_ordered_assignments = {
            cell: refine
            for _component_id, (_score, cell, refine) in best_row_assignment_by_component.items()
        }
        row_ordered_assignments.update(componentless_row_assignments)

    row_ordered_used = 0
    elastic_row_used = 0
    for cell, row_refine in row_ordered_assignments.items():
        current_refine = orange_assignments.get(cell)
        use_row_refine = current_refine is None
        if current_refine is not None:
            current_grid_dist = float(current_refine.get("grid_dist", 99.0))
            row_grid_dist = float(row_refine.get("grid_dist", 99.0))
            current_cost = float(current_refine.get("cost", 99.0))
            row_cost = float(row_refine.get("cost", 99.0))
            if row_grid_dist <= 0.72 and current_grid_dist > row_grid_dist + 0.12:
                use_row_refine = True
            elif row_grid_dist <= 0.58 and row_cost <= current_cost + 0.34:
                use_row_refine = True
            elif bool(row_refine.get("elastic_row", False)) and row_grid_dist <= 0.76 and row_cost <= current_cost + 0.46:
                use_row_refine = True
        if use_row_refine:
            row_component_id = int(row_refine.get("component_id", -1))
            if row_component_id >= 0:
                for other_cell, other_refine in list(orange_assignments.items()):
                    if other_cell == cell:
                        continue
                    if int(other_refine.get("component_id", -2)) == row_component_id:
                        del orange_assignments[other_cell]
            orange_assignments[cell] = row_refine
            row_ordered_used += 1
            if bool(row_refine.get("elastic_row", False)):
                elastic_row_used += 1

    prompts = []
    meta_points = []
    for state in cell_states:
        x, y = state["current_xy"]
        orange_refine = orange_assignments.get(state["cell"])
        if orange_refine is not None:
            x, y = orange_refine["x"], orange_refine["y"]

        x = max(0.0, min(float(frame_bgr.shape[1] - 1), x))
        y = max(0.0, min(float(frame_bgr.shape[0] - 1), y))
        prompts.append(_make_prompt_point(x, y, 1))
        meta_points.append(
            {
                "obj_id": len(prompts) - 1,
                "row": state["row"],
                "col": state["col"],
                "source": state["source"],
                "confidence": state["confidence"],
                "x": x,
                "y": y,
                "lattice_source": state["lattice_source"],
                "candidate_grid_dist": state["candidate_grid_dist"],
                "candidate_lattice_dist_px": state["candidate_lattice_dist_px"],
                "candidate_area": state["candidate_area"],
                "orange_body_refined": orange_refine is not None,
                "orange_body_area": float(orange_refine["area"]) if orange_refine is not None else None,
                "orange_component_id": int(orange_refine["component_id"]) if orange_refine is not None else None,
                "orange_assignment_cost": float(orange_refine["cost"]) if orange_refine is not None else None,
                "orange_grid_x": float(orange_refine["grid_x"]) if orange_refine is not None and "grid_x" in orange_refine else None,
                "orange_grid_y": float(orange_refine["grid_y"]) if orange_refine is not None and "grid_y" in orange_refine else None,
                "orange_grid_dist": float(orange_refine["grid_dist"]) if orange_refine is not None and "grid_dist" in orange_refine else None,
                "orange_row_ordered": bool(orange_refine.get("row_ordered", False)) if orange_refine is not None else False,
                "orange_elastic_row": bool(orange_refine.get("elastic_row", False)) if orange_refine is not None else False,
                "anchor_locked": state["anchor_locked"],
            }
        )

    print(
        f"[INFO][{side.upper()}] semi-auto mode={'hole-net' if hole_net_mode else 'full-grid'}; "
        f"lattice assigned {detected_count}/{expected} from detected centroids; "
        f"{lattice_count} lattice fallback; {anchor_count} locked anchors; "
        f"{expected - detected_count - lattice_count - anchor_count} corner fallback; "
        f"large-blob swaps={large_blob_replacements}; "
        f"orange components={len(orange_components)}; one-to-one refinements={len(orange_assignments)}; "
        f"row-ordered fixes={row_ordered_used}; elastic-row fixes={elastic_row_used}."
    )
    meta = {
        "side": side,
        "cols": int(cols),
        "rows": int(rows),
        "detector": detector_meta,
        "assignment": {
            "hole_net_mode": bool(hole_net_mode),
            "candidate_records": int(len(records)),
            "lattice_px_threshold": float(max_px_dist),
            "large_blob_replacements": int(large_blob_replacements),
            "orange_components": int(len(orange_components)),
            "orange_body_replacements": int(len(orange_assignments)),
            "orange_row_ordered_replacements": int(row_ordered_used),
            "orange_elastic_row_replacements": int(elastic_row_used),
            "orange_search_radius_px": int(orange_search_radius),
            "orange_assignment_method": (
                "hungarian" if linear_sum_assignment is not None else "greedy_unique"
            ),
        },
        "assigned_detected": int(detected_count),
        "assigned_lattice": int(lattice_count),
        "assigned_anchors": int(anchor_count),
        "assigned_inferred": int(expected - detected_count - lattice_count - anchor_count),
        "points": meta_points,
    }
    return prompts, meta


def _sync_meta_positions_for_points(points, meta):
    synced = json.loads(json.dumps(meta or {}))
    meta_points = synced.get("points", [])
    for idx, point in enumerate(points):
        if idx >= len(meta_points):
            continue
        x, y, _label = _coerce_prompt_point(point, prefer_local=False)
        if (
            abs(float(meta_points[idx].get("x", x)) - x) > 1e-3
            or abs(float(meta_points[idx].get("y", y)) - y) > 1e-3
        ):
            meta_points[idx]["source"] = "edited"
        meta_points[idx]["obj_id"] = int(idx)
        meta_points[idx]["x"] = float(x)
        meta_points[idx]["y"] = float(y)
    return synced


def _offset_section_meta(meta, section_index: int, obj_offset: int):
    adjusted = json.loads(json.dumps(meta or {}))
    adjusted["section_index"] = int(section_index)
    for point in adjusted.get("points", []):
        local_obj_id = int(point.get("obj_id", 0))
        point["section_index"] = int(section_index)
        point["section_local_obj_id"] = local_obj_id
        point["obj_id"] = int(obj_offset) + local_obj_id
    return adjusted


def _append_section_with_overlap_dedupe(
    left_points,
    right_points,
    left_meta_points,
    right_meta_points,
    section_left_points,
    section_right_points,
    section_left_meta,
    section_right_meta,
    threshold_px: float = 12.0,
):
    kept_left = []
    kept_right = []
    kept_left_meta = []
    kept_right_meta = []
    skipped = 0
    threshold2 = float(threshold_px) ** 2

    existing_pairs = []
    for left_point, right_point in zip(left_points, right_points):
        lx, ly, _ = _coerce_prompt_point(left_point, prefer_local=False)
        rx, ry, _ = _coerce_prompt_point(right_point, prefer_local=False)
        existing_pairs.append((float(lx), float(ly), float(rx), float(ry)))

    for local_idx, (left_point, right_point) in enumerate(zip(section_left_points, section_right_points)):
        lx, ly, _ = _coerce_prompt_point(left_point, prefer_local=False)
        rx, ry, _ = _coerce_prompt_point(right_point, prefer_local=False)
        duplicate = False
        for ex_lx, ex_ly, ex_rx, ex_ry in existing_pairs:
            left_d2 = (float(lx) - ex_lx) ** 2 + (float(ly) - ex_ly) ** 2
            right_d2 = (float(rx) - ex_rx) ** 2 + (float(ry) - ex_ry) ** 2
            if left_d2 <= threshold2 and right_d2 <= threshold2:
                duplicate = True
                break
        if duplicate:
            skipped += 1
            continue
        kept_left.append(left_point)
        kept_right.append(right_point)
        kept_left_meta.append(section_left_meta[local_idx])
        kept_right_meta.append(section_right_meta[local_idx])
        existing_pairs.append((float(lx), float(ly), float(rx), float(ry)))

    obj_offset = len(left_points)
    for local_idx, (left_point, right_point) in enumerate(zip(kept_left, kept_right)):
        global_idx = obj_offset + local_idx
        left_points.append(left_point)
        right_points.append(right_point)
        left_meta = dict(kept_left_meta[local_idx])
        right_meta = dict(kept_right_meta[local_idx])
        left_meta["obj_id"] = int(global_idx)
        right_meta["obj_id"] = int(global_idx)
        left_meta_points.append(left_meta)
        right_meta_points.append(right_meta)

    return len(kept_left), skipped


def _run_sectioned_grid_setup(left_first, right_first, default_cols=None, default_rows=None, section_layout=None):
    left_points = []
    right_points = []
    left_meta = {
        "side": "left",
        "mode": "sectioned_grid",
        "sections": [],
        "points": [],
    }
    right_meta = {
        "side": "right",
        "mode": "sectioned_grid",
        "sections": [],
        "points": [],
    }
    last_cols = default_cols
    last_rows = default_rows
    section_index = 0
    fixed_layout = list(section_layout or [])
    precollected_left_corners = None
    precollected_right_corners = None
    if fixed_layout:
        total_corners = 4 * len(fixed_layout)
        print("")
        print(
            f"[INFO] Click all section corner anchors once: {total_corners} points per side."
        )
        print(
            "[INFO] Order is 4 points per section: top-left, top-right, bottom-left, bottom-right. "
            "For two sections, points 0-3 are section 1 and 4-7 are section 2."
        )
        precollected_left_corners, precollected_right_corners = click_points_dual(
            left_first,
            right_first,
            corner_collection=True,
            required_count=total_corners,
        )

    while True:
        section_index += 1
        print("")
        print(f"[SECTION {section_index}] Define one locally rectangular net patch.")
        if fixed_layout:
            if section_index > len(fixed_layout):
                break
            layout_item = fixed_layout[section_index - 1]
            cols = int(layout_item["cols"])
            rows = int(layout_item["rows"])
            print(f"[INFO] Section layout from command: {cols} columns x {rows} rows")
        else:
            cols = _prompt_int("Section grid columns", default=last_cols, minimum=2)
            rows = _prompt_int("Section grid rows", default=last_rows, minimum=2)
        last_cols = cols
        last_rows = rows
        expected_points = int(cols) * int(rows)
        corner_source = "manual"
        if precollected_left_corners is not None and precollected_right_corners is not None:
            corner_source = "manual_batch"
            start = 4 * (section_index - 1)
            end = start + 4
            left_corners = precollected_left_corners[start:end]
            right_corners = precollected_right_corners[start:end]
            print(
                f"[INFO] Section {section_index}: using pre-clicked anchors "
                f"{start}-{end - 1}."
            )
        else:
            print(
                "[INFO] Click exactly 4 anchors for this section in each window, in order: "
                "top-left, top-right, bottom-left, bottom-right. Press q when done."
            )
            left_corners, right_corners = click_points_dual(left_first, right_first)
        if len(left_corners) != 4 or len(right_corners) != 4:
            raise RuntimeError(
                f"Section {section_index} needs exactly 4 anchors per side; "
                f"got LEFT={len(left_corners)}, RIGHT={len(right_corners)}."
            )

        left_corner_xy = [_coerce_point_pair(p, prefer_local=False) for p in left_corners]
        right_corner_xy = [_coerce_point_pair(p, prefer_local=False) for p in right_corners]
        section_left_points, section_left_meta = _generate_grid_prompts_from_detection(
            "left",
            left_first,
            left_corner_xy,
            cols,
            rows,
            hole_net_mode=True,
        )
        section_right_points, section_right_meta = _generate_grid_prompts_from_detection(
            "right",
            right_first,
            right_corner_xy,
            cols,
            rows,
            hole_net_mode=True,
        )

        section_left_meta = _offset_section_meta(section_left_meta, section_index, len(left_points))
        section_right_meta = _offset_section_meta(section_right_meta, section_index, len(right_points))

        kept, skipped = _append_section_with_overlap_dedupe(
            left_points,
            right_points,
            left_meta["points"],
            right_meta["points"],
            section_left_points,
            section_right_points,
            section_left_meta.get("points", []),
            section_right_meta.get("points", []),
        )
        section_record = {
            "section_index": int(section_index),
            "cols": int(cols),
            "rows": int(rows),
            "expected_points": int(expected_points),
            "accepted_points": int(kept),
            "deduped_overlap_points": int(skipped),
            "first_obj_id": int(len(left_points) - kept) if kept else None,
            "last_obj_id": int(len(left_points) - 1) if kept else None,
            "corner_source": corner_source,
            "left_corners": [[float(x), float(y)] for x, y in left_corner_xy],
            "right_corners": [[float(x), float(y)] for x, y in right_corner_xy],
            "left_assignment": section_left_meta.get("assignment", {}),
            "right_assignment": section_right_meta.get("assignment", {}),
        }
        left_meta["sections"].append(dict(section_record))
        right_meta["sections"].append(dict(section_record))

        print(
            f"[INFO] Section {section_index}: generated {expected_points}, "
            f"kept {kept}, skipped {skipped} overlap duplicate(s)."
        )
        print(
            f"[INFO] Review all accepted prompts so far ({len(left_points)} per side). "
            "Use two-click move for corrections, then press q to accept this section."
        )
        left_points, right_points = click_points_dual(
            left_first,
            right_first,
            initial_left_points=left_points,
            initial_right_points=right_points,
            fixed_count_review=True,
            initial_left_meta=left_meta["points"],
            initial_right_meta=right_meta["points"],
        )
        if len(left_points) != len(right_points):
            raise RuntimeError(
                f"Sectioned setup count mismatch after review: LEFT={len(left_points)}, RIGHT={len(right_points)}"
            )
        left_meta = _sync_meta_positions_for_points(left_points, left_meta)
        right_meta = _sync_meta_positions_for_points(right_points, right_meta)

        if fixed_layout:
            if section_index >= len(fixed_layout):
                break
            continue
        if not _prompt_yes_no("Add another section?", default=True):
            break

    if not left_points:
        raise RuntimeError("No sectioned setup points were accepted.")
    return left_points, right_points, left_meta, right_meta


def _save_semi_auto_debug(out_root: Path, left_img, right_img, left_points, right_points, left_meta, right_meta):
    debug_dir = out_root / "prompts" / "semi_auto_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    left_meta = _sync_meta_positions_for_points(left_points, left_meta)
    right_meta = _sync_meta_positions_for_points(right_points, right_meta)

    def draw_preview(img, points, meta, path):
        disp = img.copy()
        meta_by_idx = {int(p["obj_id"]): p for p in meta.get("points", [])}
        for idx, point in enumerate(points):
            x, y, _label = _coerce_prompt_point(point, prefer_local=False)
            point_meta = meta_by_idx.get(idx, {})
            source = point_meta.get("source", "unknown")
            verified = source in {"detected", "edited"} or bool(point_meta.get("orange_body_refined"))
            if source == "anchor":
                color = (255, 0, 255)
            else:
                color = (0, 220, 0) if verified else (0, 180, 255)
            center = (int(round(x)), int(round(y)))
            cv2.circle(disp, center, 6, color, 2)
            cv2.circle(disp, center, 2, color, -1)
            cv2.putText(disp, str(idx), (int(round(x)) + 8, int(round(y)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imwrite(str(path), disp)

    draw_preview(left_img, left_points, left_meta, debug_dir / "left_preview.png")
    draw_preview(right_img, right_points, right_meta, debug_dir / "right_preview.png")
    with open(debug_dir / "semi_auto_meta.json", "w") as f:
        json.dump({"left": left_meta, "right": right_meta}, f, indent=2)


def select_crop_roi(video_path: str):
    """
    Open the first video frame and let the user drag a crop ROI.
    Returns (x, y, w, h) in original frame coordinates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError("Could not read first frame for crop selection.")

    h, w = frame.shape[:2]
    max_w, max_h = 1600, 900
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    disp = frame if scale == 1.0 else cv2.resize(
        frame,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    win = "Select Crop ROI (Enter/Space confirm, c cancel)"
    _open_cv_window(win, disp.shape[1], disp.shape[0])
    roi = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)

    x, y, rw, rh = [int(v) for v in roi]
    if rw <= 0 or rh <= 0:
        raise RuntimeError("No crop ROI selected. Drag a region and press Enter to confirm.")

    if scale != 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        rw = int(round(rw / scale))
        rh = int(round(rh / scale))

    # Clamp to frame bounds.
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    rw = max(1, min(w - x, rw))
    rh = max(1, min(h - y, rh))

    print(f"[INFO] Selected crop ROI x={x}, y={y}, w={rw}, h={rh}")
    return x, y, rw, rh


def _draw_crop_hud(display, frame_idx, total_frames, fps, crop, g_mode, digit_buf):
    x, y, w, h = crop
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 255), 2)

    hud_h = 150
    hud = np.zeros((hud_h, display.shape[1], 3), dtype=np.uint8)
    cv2.putText(hud, f"Frame: {frame_idx}/{max(0, total_frames - 1)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(hud, f"Time: {frame_idx / max(fps, 1e-6):.2f}s", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(hud, "a/d frame  A/D ~1s  Ctrl+A/Ctrl+D ~10s  g goto", (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(hud, "q accept crop  r reselect crop  Esc cancel", (20, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if g_mode:
        cv2.putText(hud, f"Go to frame: {digit_buf}_", (650, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    return np.vstack([hud, display])


def review_crop_over_video(video_path: str, crop, label: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for crop review: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid frame count during crop review.")
    if fps <= 0:
        fps = 30.0

    jump_small = max(1, int(round(fps)))
    jump_large = max(10, int(round(10 * fps)))

    window = f"Crop Review [{label}]"
    _open_cv_window(window, 1600, 1050)
    cv2.createTrackbar("Frame", window, 0, max(1, total_frames - 1), lambda _: None)

    current_frame = 0
    last_tb = 0
    g_mode = False
    digit_buf = ""

    while True:
        current_frame = max(0, min(current_frame, total_frames - 1))

        tb_val = cv2.getTrackbarPos("Frame", window)
        if tb_val != last_tb:
            current_frame = tb_val
            last_tb = tb_val
        else:
            cv2.setTrackbarPos("Frame", window, current_frame)
            last_tb = current_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            current_frame += 1
            continue

        disp = _draw_crop_hud(frame.copy(), current_frame, total_frames, fps, crop, g_mode, digit_buf)
        cv2.imshow(window, disp)

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        if key == 27:
            cap.release()
            cv2.destroyWindow(window)
            raise RuntimeError("Crop review canceled by user.")

        if key == ord("q"):
            cap.release()
            cv2.destroyWindow(window)
            return "accept"

        if key == ord("r"):
            cap.release()
            cv2.destroyWindow(window)
            return "reselect"

        if key == ord("g"):
            g_mode = True
            digit_buf = ""
            continue

        if g_mode:
            if key == 13:  # Enter
                try:
                    current_frame = max(0, min(int(digit_buf), total_frames - 1))
                except ValueError:
                    pass
                g_mode = False
                digit_buf = ""
            elif key == 8:  # Backspace
                digit_buf = digit_buf[:-1]
            elif ord("0") <= key <= ord("9"):
                digit_buf += chr(key)
            continue

        if key == ord("d"):
            current_frame += 1
        elif key == ord("a"):
            current_frame -= 1
        elif key == ord("D"):
            current_frame += jump_small
        elif key == ord("A"):
            current_frame -= jump_small
        elif key == 4:  # Ctrl+D
            current_frame += jump_large
        elif key == 1:  # Ctrl+A
            current_frame -= jump_large


def select_crop_with_review(video_path: str, label: str):
    while True:
        crop = select_crop_roi(video_path)
        action = review_crop_over_video(video_path, crop, label)
        if action == "accept":
            return crop


def load_first_frame_from_video(video_path: str, crop=None, frame_idx: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_idx = max(0, int(frame_idx))
    if frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    if crop is not None:
        x, y, w, h = crop
        fh, fw = frame.shape[:2]
        if x + w > fw or y + h > fh:
            raise ValueError(f"Crop {crop} is outside frame bounds {fw}x{fh}")
        frame = frame[y:y + h, x:x + w]
    return frame


def click_points_dual(
    image_left_bgr,
    image_right_bgr,
    initial_left_points=None,
    initial_right_points=None,
    fixed_count_review: bool = False,
    move_existing_points: bool = False,
    anchor_review: bool = False,
    corner_collection: bool = False,
    required_count: int = None,
    initial_left_meta=None,
    initial_right_meta=None,
):
    left_points = json.loads(json.dumps(initial_left_points or []))
    right_points = json.loads(json.dumps(initial_right_points or []))
    left_meta = json.loads(json.dumps(initial_left_meta or []))
    right_meta = json.loads(json.dumps(initial_right_meta or []))
    active_view_name = "left"
    hud_h = 150

    def setup_view(img):
        h, w = img.shape[:2]
        max_w, max_h = 1800, 1000
        base_scale = min(max_w / float(w), max_h / float(h), 1.0)
        disp_w = max(1, int(round(w * base_scale)))
        disp_h = max(1, int(round(h * base_scale)))
        return {
            "img": img,
            "h": h,
            "w": w,
            "base_scale": base_scale,
            "disp_w": disp_w,
            "disp_h": disp_h,
            "zoom": 1.0,
            "min_zoom": 1.0,
            "max_zoom": 8.0,
            "center_x": w / 2.0,
            "center_y": h / 2.0,
            "last_mouse": (disp_w // 2, hud_h + disp_h // 2),
            "dragging": False,
            "last_drag": (disp_w // 2, hud_h + disp_h // 2),
            "moving_idx": None,
            "selected_idx": None,
            "add_next": False,
        }

    left_view = setup_view(image_left_bgr)
    right_view = setup_view(image_right_bgr)

    count_locked_review = fixed_count_review or anchor_review
    move_mode = count_locked_review or move_existing_points

    if corner_collection:
        left_win = "LEFT section corner anchors"
        right_win = "RIGHT section corner anchors"
    elif anchor_review:
        left_win = "LEFT auto corner guesses (0=TL 1=TR 2=BL 3=BR)"
        right_win = "RIGHT auto corner guesses (0=TL 1=TR 2=BL 3=BR)"
    elif fixed_count_review:
        left_win = "LEFT semi-auto prompts (two-click move, RMB negative-nearest)"
        right_win = "RIGHT semi-auto prompts (two-click move, RMB negative-nearest)"
    elif move_existing_points:
        left_win = "LEFT modify prompts (two-click move, n add next, RMB negative-nearest)"
        right_win = "RIGHT modify prompts (two-click move, n add next, RMB negative-nearest)"
    else:
        left_win = "LEFT prompts (LMB positive, RMB negative-nearest)"
        right_win = "RIGHT prompts (LMB positive, RMB negative-nearest)"

    def clamp_center(view, cx, cy):
        view_w = view["w"] / view["zoom"]
        view_h = view["h"] / view["zoom"]
        half_w = view_w / 2.0
        half_h = view_h / 2.0
        cx = max(half_w, min(view["w"] - half_w, cx))
        cy = max(half_h, min(view["h"] - half_h, cy))
        return cx, cy

    def get_view_rect(view):
        view_w = view["w"] / view["zoom"]
        view_h = view["h"] / view["zoom"]
        cx, cy = clamp_center(view, view["center_x"], view["center_y"])
        view["center_x"], view["center_y"] = cx, cy

        x0 = int(round(cx - view_w / 2.0))
        y0 = int(round(cy - view_h / 2.0))
        x1 = int(round(x0 + view_w))
        y1 = int(round(y0 + view_h))

        x0 = max(0, min(view["w"] - 1, x0))
        y0 = max(0, min(view["h"] - 1, y0))
        x1 = max(x0 + 1, min(view["w"], x1))
        y1 = max(y0 + 1, min(view["h"], y1))
        return x0, y0, x1, y1

    def map_disp_to_orig(view, x, y):
        img_x = max(0, min(view["disp_w"] - 1, int(round(x))))
        img_y = max(0, min(view["disp_h"] - 1, int(round(y)) - hud_h))
        x0, y0, x1, y1 = get_view_rect(view)
        view_w = x1 - x0
        view_h = y1 - y0
        ox = x0 + (img_x / max(view["disp_w"] - 1, 1)) * view_w
        oy = y0 + (img_y / max(view["disp_h"] - 1, 1)) * view_h
        ox = int(round(ox))
        oy = int(round(oy))
        ox = max(0, min(view["w"] - 1, ox))
        oy = max(0, min(view["h"] - 1, oy))
        return ox, oy

    def map_disp_delta_to_orig(view, ddx, ddy):
        x0, y0, x1, y1 = get_view_rect(view)
        view_w = x1 - x0
        view_h = y1 - y0
        ox = (ddx / max(view["disp_w"] - 1, 1)) * view_w
        oy = (ddy / max(view["disp_h"] - 1, 1)) * view_h
        return ox, oy

    def zoom_at_disp_point(view, x, y, new_zoom):
        img_x = max(0, min(view["disp_w"] - 1, int(round(x))))
        img_y = max(0, min(view["disp_h"] - 1, int(round(y)) - hud_h))
        new_zoom = max(view["min_zoom"], min(view["max_zoom"], new_zoom))
        if new_zoom == view["zoom"]:
            return
        ox, oy = map_disp_to_orig(view, img_x, hud_h + img_y)
        fx = max(0.0, min(1.0, img_x / max(view["disp_w"] - 1, 1)))
        fy = max(0.0, min(1.0, img_y / max(view["disp_h"] - 1, 1)))
        view["zoom"] = new_zoom
        new_view_w = view["w"] / view["zoom"]
        new_view_h = view["h"] / view["zoom"]
        cx = ox + (0.5 - fx) * new_view_w
        cy = oy + (0.5 - fy) * new_view_h
        view["center_x"], view["center_y"] = clamp_center(view, cx, cy)
        view["last_mouse"] = (img_x, hud_h + img_y)

    def pan_view_with_key(view, key):
        step = max(10.0, 60.0 / float(view["zoom"]))
        if key == ord("a"):
            view["center_x"] -= step
        elif key == ord("d"):
            view["center_x"] += step
        elif key == ord("w"):
            view["center_y"] -= step
        elif key == ord("s"):
            view["center_y"] += step
        view["center_x"], view["center_y"] = clamp_center(view, view["center_x"], view["center_y"])

    def render(view, points, title, other_count, meta=None):
        x0, y0, x1, y1 = get_view_rect(view)
        crop = view["img"][y0:y1, x0:x1]
        disp = cv2.resize(crop, (view["disp_w"], view["disp_h"]), interpolation=cv2.INTER_LINEAR)
        meta_by_idx = {}
        for item in meta or []:
            try:
                meta_by_idx[int(item.get("obj_id", len(meta_by_idx)))] = item
            except Exception:
                pass

        for idx, point in enumerate(points):
            x, y, label = _coerce_prompt_point(point, prefer_local=False)
            if x0 <= x < x1 and y0 <= y < y1:
                dx = int(round((x - x0) * (view["disp_w"] / max(x1 - x0, 1))))
                dy = int(round((y - y0) * (view["disp_h"] / max(y1 - y0, 1))))
                if move_mode:
                    point_meta = meta_by_idx.get(idx, {})
                    source = point_meta.get("source", "detected")
                    verified = source in {"detected", "edited"} or bool(point_meta.get("orange_body_refined"))
                    if count_locked_review and verified:
                        color = (0, 220, 0)
                    elif count_locked_review and source == "anchor":
                        color = (255, 0, 255)
                    else:
                        color = (0, 180, 255)
                    if idx == view.get("selected_idx"):
                        color = (255, 0, 255)
                else:
                    color = (0, 255, 255)
                cv2.circle(disp, (dx, dy), 6, color, 2)
                cv2.circle(disp, (dx, dy), 2, color, -1)
                if move_mode and idx == view.get("selected_idx"):
                    cv2.circle(disp, (dx, dy), 12, color, 2)
                cv2.putText(
                    disp,
                    f"{idx}+",
                    (dx + 8, dy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            for nx, ny in _coerce_negative_points(point):
                if x0 <= nx < x1 and y0 <= ny < y1:
                    dx = int(round((nx - x0) * (view["disp_w"] / max(x1 - x0, 1))))
                    dy = int(round((ny - y0) * (view["disp_h"] / max(y1 - y0, 1))))
                    cv2.drawMarker(
                        disp,
                        (dx, dy),
                        (0, 0, 255),
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=14,
                        thickness=2,
                    )
                    cv2.putText(
                        disp,
                        f"{idx}-",
                        (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        hud = np.zeros((hud_h, view["disp_w"], 3), dtype=np.uint8)
        cv2.putText(hud, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(hud, f"count={len(points)}  other={other_count}  zoom={view['zoom']:.1f}x", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if corner_collection:
            count_text = f"{required_count} total" if required_count is not None else "4 per section"
            cv2.putText(hud, f"Click section corners in groups of 4: TL TR BL BR  ({count_text})", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(hud, "For two sections: 0-3 section 1, 4-7 section 2. x undo pair  q accept  Esc cancel", (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)
        elif anchor_review:
            cv2.putText(hud, "Review corners: 0=TL  1=TR  2=BL  3=BR  two-click move  q accept", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(hud, "MMB drag/WASD pan  c clear selection  Esc cancel  1/2 focus", (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif fixed_count_review:
            cv2.putText(hud, "LMB select point, LMB again places it  RMB negative  green=verified orange=unverified magenta=anchor", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(hud, "MMB drag/WASD pan  c clear selection  z undo neg  q accept  Esc cancel  1/2 focus", (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif move_existing_points:
            cv2.putText(hud, "LMB select point, LMB again places it  n add next click  RMB negative", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(hud, "MMB drag/WASD pan  c clear selection  z undo neg  x undo object pair  q done  Esc cancel", (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        else:
            cv2.putText(hud, "LMB add object  RMB add negative to nearest object  wheel/+/- zoom", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(hud, "MMB drag/WASD pan  z undo neg  x undo object pair  q done  Esc cancel  1/2 focus", (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return np.vstack([hud, disp])

    def add_negative_to_nearest(points, px, py):
        if not points:
            print("Add at least one positive object before adding negative prompts.")
            return
        best_idx = None
        best_d2 = None
        for idx, point in enumerate(points):
            x, y, _label = _coerce_prompt_point(point, prefer_local=False)
            d2 = (float(px) - x) ** 2 + (float(py) - y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = idx
        points[best_idx].setdefault("negative", []).append([float(px), float(py)])
        print(f"Added negative prompt to object {best_idx}")

    def undo_last_negative(points):
        for idx in range(len(points) - 1, -1, -1):
            negatives = points[idx].get("negative", [])
            if negatives:
                negatives.pop()
                if not negatives:
                    points[idx].pop("negative", None)
                print(f"Removed negative prompt from object {idx}")
                return
        print("No negative prompt to undo in active window.")

    def nearest_point_index(points, px, py):
        best_idx = None
        best_d2 = None
        for idx, point in enumerate(points):
            x0, y0, _label = _coerce_prompt_point(point, prefer_local=False)
            d2 = (float(px) - x0) ** 2 + (float(py) - y0) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = idx
        return best_idx

    def move_point(points, idx, px, py, meta):
        if idx is None or idx < 0 or idx >= len(points):
            return
        negatives = _coerce_negative_points(points[idx])
        points[idx] = _make_prompt_point(px, py, 1, negatives)
        if idx < len(meta):
            meta[idx]["source"] = "edited"
            meta[idx]["x"] = float(px)
            meta[idx]["y"] = float(py)

    def _mouse_common(event, x, y, flags, view, points, which_name, meta):
        nonlocal active_view_name
        active_view_name = which_name
        if event == cv2.EVENT_MOUSEMOVE:
            view["last_mouse"] = (x, max(y, hud_h))
            if view["dragging"]:
                ddx = x - view["last_drag"][0]
                ddy = y - view["last_drag"][1]
                ox, oy = map_disp_delta_to_orig(view, ddx, ddy)
                view["center_x"] -= ox
                view["center_y"] -= oy
                view["last_drag"] = (x, y)
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            if y < hud_h:
                return
            px, py = map_disp_to_orig(view, x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                if move_mode:
                    if view.get("add_next"):
                        points.append(_make_prompt_point(px, py, 1))
                        view["add_next"] = False
                        view["selected_idx"] = None
                        print(f"[{which_name.upper()}] Added object {len(points) - 1} at x={px}, y={py}.")
                    elif view.get("selected_idx") is None:
                        view["selected_idx"] = nearest_point_index(points, px, py)
                        print(f"[{which_name.upper()}] Selected object {view['selected_idx']}; click the desired marker center to place it.")
                    else:
                        idx = view["selected_idx"]
                        move_point(points, idx, px, py, meta)
                        view["selected_idx"] = None
                        print(f"[{which_name.upper()}] Moved object {idx} to x={px}, y={py}.")
                else:
                    points.append(_make_prompt_point(px, py, 1))
            else:
                add_negative_to_nearest(points, px, py)
        elif event == cv2.EVENT_MBUTTONDOWN:
            if y < hud_h:
                return
            view["dragging"] = True
            view["last_drag"] = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            view["dragging"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if y < hud_h:
                return
            delta = INTERACTIVE_ZOOM_STEP if flags > 0 else (1 / INTERACTIVE_ZOOM_STEP)
            zoom_at_disp_point(view, x, y, view["zoom"] * delta)

    def on_left(event, x, y, flags, param):
        _mouse_common(event, x, y, flags, left_view, left_points, "left", left_meta)

    def on_right(event, x, y, flags, param):
        _mouse_common(event, x, y, flags, right_view, right_points, "right", right_meta)

    _open_cv_window(left_win, left_view["disp_w"], left_view["disp_h"] + hud_h)
    _open_cv_window(right_win, right_view["disp_w"], right_view["disp_h"] + hud_h)
    cv2.setMouseCallback(left_win, on_left)
    cv2.setMouseCallback(right_win, on_right)

    while True:
        cv2.imshow(left_win, render(left_view, left_points, "LEFT", len(right_points), left_meta))
        cv2.imshow(right_win, render(right_view, right_points, "RIGHT", len(left_points), right_meta))
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow(left_win)
            cv2.destroyWindow(right_win)
            raise RuntimeError("Point selection canceled by user.")

        if key == ord("x") and not count_locked_review:
            if left_points and right_points:
                left_points.pop()
                right_points.pop()
            elif left_points:
                left_points.pop()
            elif right_points:
                right_points.pop()

        if key == ord("z"):
            undo_last_negative(left_points if active_view_name == "left" else right_points)

        if move_mode and key == ord("c"):
            target = left_view if active_view_name == "left" else right_view
            target["selected_idx"] = None
            target["add_next"] = False
            print(f"[{active_view_name.upper()}] Cleared selected point.")

        if move_existing_points and key == ord("n"):
            target = left_view if active_view_name == "left" else right_view
            target["selected_idx"] = None
            target["add_next"] = True
            print(f"[{active_view_name.upper()}] Next left click will add a new object.")

        if key == ord("1"):
            active_view_name = "left"
        elif key == ord("2"):
            active_view_name = "right"

        if key in (ord("+"), ord("="), ord("-"), ord("_")):
            target = left_view if active_view_name == "left" else right_view
            if key in (ord("+"), ord("=")):
                zoom_at_disp_point(target, *target["last_mouse"], target["zoom"] * INTERACTIVE_ZOOM_STEP)
            else:
                zoom_at_disp_point(target, *target["last_mouse"], target["zoom"] / INTERACTIVE_ZOOM_STEP)

        if key in (ord("w"), ord("a"), ord("s"), ord("d")):
            target = left_view if active_view_name == "left" else right_view
            pan_view_with_key(target, key)

        if key == ord("q"):
            if len(left_points) == 0:
                print("Select at least one point before finishing.")
                continue
            if len(left_points) != len(right_points):
                print(f"Point count mismatch: LEFT={len(left_points)}, RIGHT={len(right_points)}")
                continue
            if required_count is not None and (
                len(left_points) != int(required_count) or len(right_points) != int(required_count)
            ):
                print(
                    f"Expected exactly {int(required_count)} points per side; "
                    f"LEFT={len(left_points)}, RIGHT={len(right_points)}"
                )
                continue
            if count_locked_review and len(left_points) != len(initial_left_points or []):
                print("Fixed-count review point count changed unexpectedly; cancel and regenerate.")
                continue
            break

        if cv2.getWindowProperty(left_win, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(right_win, cv2.WND_PROP_VISIBLE) < 1:
            raise RuntimeError("Point selection window closed before finishing.")

    cv2.destroyWindow(left_win)
    cv2.destroyWindow(right_win)
    return left_points, right_points


def _video_size(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Could not read video dimensions: {video_path}")
    return w, h


def _video_signature(video_path: str):
    path = Path(video_path)
    stat = path.stat()
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else -1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else -1
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else -1.0
    cap.release()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
    }


def _stable_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_cached_frames(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    return sum(1 for _ in frames_dir.glob(FRAME_CACHE_GLOB))


def _frame_cache_status(frames_dir: Path, expected_count: int):
    cached_count = _count_cached_frames(frames_dir)
    if cached_count <= 0:
        return False, cached_count, "no JPEG frames found"
    if expected_count <= 0:
        return True, cached_count, "OpenCV could not report the video frame count"
    if cached_count != expected_count:
        return False, cached_count, f"expected {expected_count} JPEG frames, found {cached_count}"
    first_frame = frames_dir / f"000000{FRAME_CACHE_EXT}"
    last_frame = frames_dir / f"{expected_count - 1:06d}{FRAME_CACHE_EXT}"
    if not first_frame.exists():
        return False, cached_count, f"missing first frame {first_frame.name}"
    if not last_frame.exists():
        return False, cached_count, f"missing last frame {last_frame.name}"
    return True, cached_count, ""


def _frame_cache_meta(
    video_path: str,
    crop,
    scale: float,
    frame_extractor: str,
    start_frame: int = 0,
    end_frame: int = None,
    frame_step: int = 1,
):
    video = _video_signature(video_path)
    end = int(end_frame) if end_frame is not None else int(video["frame_count"])
    step = max(1, int(frame_step))
    source_frame_count = end - int(start_frame)
    return {
        "video": video,
        "frame_range": {
            "start_frame": int(start_frame),
            "end_frame": end,
            "frame_step": step,
            "source_frame_count": source_frame_count,
            "frame_count": int(math.ceil(source_frame_count / float(step))),
        },
        "crop": list(crop) if crop is not None else None,
        "scale": float(scale),
        "frame_extractor": str(frame_extractor),
        "image_format": FRAME_CACHE_IMAGE_FORMAT,
    }


def prepare_frame_cache(video_path: str,
                        frames_dir: Path,
                        crop=None,
                        scale: float = 1.0,
                        frame_extractor: str = "auto",
                        meta_name: str = "frames_meta.json",
                        label: str = "",
                        start_frame: int = 0,
                        end_frame: int = None,
                        frame_step: int = 1):
    frames_dir = Path(frames_dir)
    meta_path = frames_dir.parent / meta_name
    frame_meta = _frame_cache_meta(
        video_path,
        crop,
        scale,
        frame_extractor,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
    )
    frame_hash = _stable_hash(frame_meta)
    old_hash = None
    if meta_path.exists():
        try:
            old_hash = json.loads(meta_path.read_text()).get("hash")
        except Exception:
            old_hash = None

    expected_count = int(frame_meta["frame_range"]["frame_count"])
    cache_complete, cached_count, cache_reason = _frame_cache_status(frames_dir, expected_count)
    prefix = f"[{label}] " if label else ""
    if old_hash != frame_hash or not cache_complete:
        if old_hash == frame_hash and not cache_complete and cached_count > 0:
            print(f"[WARN]{prefix}Ignoring incomplete frame cache: {cache_reason}")
        elif old_hash is not None and old_hash != frame_hash:
            print(f"[INFO]{prefix}Frame cache metadata changed; rebuilding {frames_dir}")
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        extract_frames(
            video_path,
            frames_dir,
            crop=crop,
            scale=scale,
            frame_extractor=frame_extractor,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
        )
        cache_complete, cached_count, cache_reason = _frame_cache_status(frames_dir, expected_count)
        if not cache_complete:
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            raise RuntimeError(f"{prefix}Frame extraction incomplete: {cache_reason}. Expected video={video_path}")
        meta_path.write_text(json.dumps({"hash": frame_hash, **frame_meta}, indent=2))
    else:
        print(f"[INFO]{prefix}Reusing extracted frames from {frames_dir}")
    return frames_dir, cached_count


def _validate_crop_for_video(video_path: str, crop):
    if crop is None:
        return None
    x, y, w, h = [int(v) for v in crop]
    fw, fh = _video_size(video_path)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
        raise ValueError(f"Crop x,y,w,h={crop} is outside frame bounds {fw}x{fh}.")
    return x, y, w, h


def _ffmpeg_filter(crop, scale: float):
    filters = []
    if crop is not None:
        x, y, w, h = [int(v) for v in crop]
        filters.append(f"crop={w}:{h}:{x}:{y}")
    if scale != 1.0:
        filters.append(f"scale=iw*{scale:g}:ih*{scale:g}:flags=area")
    return ",".join(filters)


def extract_frames_ffmpeg(
    video_path: str,
    frames_dir: Path,
    crop=None,
    scale: float = 1.0,
    start_frame: int = 0,
    end_frame: int = None,
):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    crop_roi = _validate_crop_for_video(video_path, crop)
    if crop_roi is not None:
        x, y, w, h = crop_roi
        print(f"[INFO] Applying crop ROI x={x}, y={y}, w={w}, h={h}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    signature = _video_signature(video_path)
    source_count = int(signature["frame_count"])
    fps = float(signature["fps"])
    start_frame = int(start_frame)
    end_frame = int(end_frame) if end_frame is not None else source_count
    if fps <= 0 or start_frame < 0 or end_frame <= start_frame or end_frame > source_count:
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}) for {video_path}")
    frame_count = end_frame - start_frame
    start_seconds = start_frame / fps
    pre_seek = min(5.0, start_seconds)
    output_pattern = str(frames_dir / f"%06d{FRAME_CACHE_EXT}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds - pre_seek:.9f}",
        "-i",
        str(video_path),
        "-ss",
        f"{pre_seek:.9f}",
    ]
    vf = _ffmpeg_filter(crop_roi, scale)
    if vf:
        cmd.extend(["-vf", vf])
    cmd.extend([
        "-frames:v", str(frame_count),
        "-q:v", "2",
        "-start_number", "0",
        output_pattern,
    ])

    print(f"[INFO] Using ffmpeg JPEG extraction for source frames [{start_frame}, {end_frame})")
    subprocess.run(cmd, check=True)
    n_frames = len(list(frames_dir.glob(FRAME_CACHE_GLOB)))
    if n_frames <= 0:
        raise RuntimeError(f"ffmpeg did not write any frames to {frames_dir}")
    print(f"[OK] Extracted {n_frames} frames to {frames_dir}")


def extract_frames_opencv(
    video_path: str,
    frames_dir: Path,
    crop=None,
    scale: float = 1.0,
    start_frame: int = 0,
    end_frame: int = None,
    frame_step: int = 1,
):
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_frame)
    end_frame = int(end_frame) if end_frame is not None else source_count
    frame_step = max(1, int(frame_step))
    if start_frame < 0 or end_frame <= start_frame or end_frame > source_count:
        cap.release()
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}) for {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    crop_roi = None
    source_i = 0
    written_i = 0
    while source_i < end_frame - start_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if source_i % frame_step != 0:
            source_i += 1
            continue

        if crop is not None:
            if crop_roi is None:
                crop_roi = _validate_crop_for_video(video_path, crop)
                x, y, w, h = crop_roi
                print(f"[INFO] Applying crop ROI x={x}, y={y}, w={w}, h={h}")

            x, y, w, h = crop_roi
            frame = frame[y:y + h, x:x + w]

        if scale != 1.0:
            new_w = max(1, int(round(frame.shape[1] * scale)))
            new_h = max(1, int(round(frame.shape[0] * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(
            str(frames_dir / f"{written_i:06d}{FRAME_CACHE_EXT}"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        written_i += 1
        source_i += 1
    cap.release()
    print(f"[OK] Extracted {written_i} frames to {frames_dir}")


def extract_frames(video_path: str,
                   frames_dir: Path,
                   crop=None,
                   scale: float = 1.0,
                   frame_extractor: str = "auto",
                   start_frame: int = 0,
                   end_frame: int = None,
                   frame_step: int = 1):
    print(f"[INFO] Extracting frames from {video_path} to {frames_dir}")
    frame_step = max(1, int(frame_step))
    if scale <= 0:
        raise ValueError("--scale must be > 0")
    if scale != 1.0:
        print(f"[INFO] Applying processing scale={scale:g} after crop")

    method = str(frame_extractor).lower()
    if method not in ("auto", "ffmpeg", "opencv"):
        raise ValueError("--frame-extractor must be auto, ffmpeg, or opencv")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    if frame_step > 1:
        print(f"[INFO] Using frame_step={frame_step}; extracting every {frame_step}th source frame with OpenCV")
        return extract_frames_opencv(
            video_path,
            frames_dir,
            crop=crop,
            scale=scale,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
        )

    if method in ("auto", "ffmpeg"):
        try:
            return extract_frames_ffmpeg(
                video_path,
                frames_dir,
                crop=crop,
                scale=scale,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        except Exception as e:
            if method == "ffmpeg":
                raise
            print(f"[WARN] ffmpeg extraction failed, falling back to OpenCV: {e}")
            if frames_dir.exists():
                shutil.rmtree(frames_dir)

    print("[INFO] Using OpenCV frame extraction")
    return extract_frames_opencv(
        video_path,
        frames_dir,
        crop=crop,
        scale=scale,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
    )

def load_first_frame(frames_dir: Path):
    first = frames_dir / f"000000{FRAME_CACHE_EXT}"
    if not first.exists():
        raise RuntimeError(f"Missing {first}")
    img = cv2.imread(str(first))
    if img is None:
        raise RuntimeError(f"Failed to read {first}")
    return img

def click_points(image_bgr, n_points):
    """
    Click exactly n_points in order.
    Left-click: add point
    Right-click: undo last point
    Press 'q' to quit (after selecting enough points).
    """
    points = []
    hud_h = 90

    # Scale display to fit screen (if needed) while preserving aspect ratio
    h, w = image_bgr.shape[:2]
    base_scale = 1.0
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        max_w = int(screen_w * 0.95)
        max_h = max(1, int(screen_h * 0.90) - hud_h)
        base_scale = min(max_w / w, max_h / h, 1.0)
    except Exception:
        base_scale = 1.0

    display_w = int(w * base_scale)
    display_h = int(h * base_scale)
    display_w = max(display_w, 1)
    display_h = max(display_h, 1)

    zoom = 1.0
    min_zoom = 1.0
    max_zoom = 8.0
    center_x = w / 2.0
    center_y = h / 2.0
    last_mouse = (display_w // 2, hud_h + display_h // 2)
    dragging = False
    last_drag = (display_w // 2, hud_h + display_h // 2)

    def clamp_center(cx, cy, view_w, view_h):
        half_w = view_w / 2.0
        half_h = view_h / 2.0
        cx = max(half_w, min(w - half_w, cx))
        cy = max(half_h, min(h - half_h, cy))
        return cx, cy

    def get_view_rect():
        view_w = w / zoom
        view_h = h / zoom
        cx, cy = clamp_center(center_x, center_y, view_w, view_h)
        x0 = int(round(cx - view_w / 2.0))
        y0 = int(round(cy - view_h / 2.0))
        x1 = int(round(x0 + view_w))
        y1 = int(round(y0 + view_h))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        return x0, y0, x1, y1

    def map_display_to_original(dx, dy):
        img_x = max(0, min(display_w - 1, int(round(dx))))
        img_y = max(0, min(display_h - 1, int(round(dy)) - hud_h))
        x0, y0, x1, y1 = get_view_rect()
        view_w = x1 - x0
        view_h = y1 - y0
        ox = x0 + (img_x / max(display_w - 1, 1)) * view_w
        oy = y0 + (img_y / max(display_h - 1, 1)) * view_h
        return int(round(ox)), int(round(oy))

    def map_display_delta_to_original(ddx, ddy):
        x0, y0, x1, y1 = get_view_rect()
        view_w = x1 - x0
        view_h = y1 - y0
        ox = (ddx / max(display_w - 1, 1)) * view_w
        oy = (ddy / max(display_h - 1, 1)) * view_h
        return ox, oy

    def zoom_at_display_point(dx, dy, new_zoom):
        nonlocal zoom, center_x, center_y, last_mouse
        img_x = max(0, min(display_w - 1, int(round(dx))))
        img_y = max(0, min(display_h - 1, int(round(dy)) - hud_h))
        new_zoom = max(min_zoom, min(max_zoom, new_zoom))
        if new_zoom == zoom:
            return
        ox, oy = map_display_to_original(img_x, hud_h + img_y)
        fx = max(0.0, min(1.0, img_x / max(display_w - 1, 1)))
        fy = max(0.0, min(1.0, img_y / max(display_h - 1, 1)))
        zoom = new_zoom
        view_w = w / zoom
        view_h = h / zoom
        center_x, center_y = clamp_center(
            ox + (0.5 - fx) * view_w,
            oy + (0.5 - fy) * view_h,
            view_w,
            view_h,
        )
        last_mouse = (img_x, hud_h + img_y)

    def pan_display_with_key(key):
        nonlocal center_x, center_y
        step = max(10.0, 60.0 / float(zoom))
        if key == ord("a"):
            center_x -= step
        elif key == ord("d"):
            center_x += step
        elif key == ord("w"):
            center_y -= step
        elif key == ord("s"):
            center_y += step
        view_w = w / zoom
        view_h = h / zoom
        center_x, center_y = clamp_center(center_x, center_y, view_w, view_h)

    def render_display():
        x0, y0, x1, y1 = get_view_rect()
        crop = image_bgr[y0:y1, x0:x1]
        img = cv2.resize(crop, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        for idx, (x, y) in enumerate(points):
            if x0 <= x < x1 and y0 <= y < y1:
                sx = int(round((x - x0) * (display_w / max(x1 - x0, 1))))
                sy = int(round((y - y0) * (display_h / max(y1 - y0, 1))))
                cv2.circle(img, (sx, sy), 6, (0, 255, 255), -1)
                cv2.putText(img, str(idx), (sx + 8, sy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        hud = np.zeros((hud_h, display_w, 3), dtype=np.uint8)
        cv2.putText(hud, f"count={len(points)}/{n_points}  zoom={zoom:.1f}x", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(hud, "LMB add  RMB undo  wheel/+/- zoom  MMB drag/WASD pan  q done",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return np.vstack([hud, img])

    img = None

    def redraw():
        nonlocal img
        img = render_display()

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        nonlocal zoom, center_x, center_y, last_mouse, dragging, last_drag
        if event == cv2.EVENT_MOUSEMOVE:
            last_mouse = (x, max(y, hud_h))
            if dragging:
                ddx = x - last_drag[0]
                ddy = y - last_drag[1]
                ox, oy = map_display_delta_to_original(ddx, ddy)
                center_x -= ox
                center_y -= oy
                last_drag = (x, y)
                redraw()
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < hud_h:
                return
            if len(points) < n_points:
                ox, oy = map_display_to_original(x, y)
                points.append((ox, oy))
                redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if y < hud_h:
                return
            if points:
                points.pop()
                redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            if y < hud_h:
                return
            dragging = True
            last_drag = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if y < hud_h:
                return
            delta = INTERACTIVE_ZOOM_STEP if flags > 0 else 1 / INTERACTIVE_ZOOM_STEP
            old_zoom = zoom
            zoom_at_display_point(x, y, zoom * delta)
            if zoom != old_zoom:
                redraw()

    redraw()
    _open_cv_window("Click markers (LMB add, RMB undo, q when done)", display_w, display_h + hud_h)
    cv2.setMouseCallback("Click markers (LMB add, RMB undo, q when done)", on_mouse)

    while True:
        cv2.imshow("Click markers (LMB add, RMB undo, q when done)", img)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("+"), ord("=")):
            zoom_at_display_point(*last_mouse, zoom * INTERACTIVE_ZOOM_STEP)
            redraw()
        elif k in (ord("-"), ord("_")):
            zoom_at_display_point(*last_mouse, zoom / INTERACTIVE_ZOOM_STEP)
            redraw()
        elif k in (ord("w"), ord("a"), ord("s"), ord("d")):
            pan_display_with_key(k)
            redraw()
        if k == ord("q"):
            if len(points) == n_points:
                break
            else:
                print(f"Need {n_points} points, currently have {len(points)}")
        if cv2.getWindowProperty("Click markers (LMB add, RMB undo, q when done)", cv2.WND_PROP_VISIBLE) < 1:
            raise RuntimeError("Window closed before finishing point selection.")

    cv2.destroyAllWindows()
    return points

#     return out
def overlay_masks(frame_bgr, masks_by_obj):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if h == 0 or w == 0:
        return out

    rng = np.random.default_rng(0)

    for obj_id, m in masks_by_obj.items():
        if m is None:
            continue

        # Ensure we have a numpy array
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()
        
        m = np.squeeze(m) # Remove singleton dimensions like (1, H, W)

        # Check if mask is empty or invalid
        if m.size == 0 or m.shape[0] == 0:
            continue

        # Resize to match frame if necessary
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        m_bin = (m > 0).astype(np.uint8)
        
        # Color the mask
        color = [int(c) for c in rng.integers(0, 255, size=3)]
        alpha = 0.45
        
        mask_indices = m_bin > 0
        if np.any(mask_indices):
            # Optimized overlay
            roi = out[mask_indices]
            colored_roi = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
            out[mask_indices] = colored_roi

    return out


def _draw_correction_points(img, clicks):
    out = img.copy()
    for i, click in enumerate(clicks):
        x, y = int(round(click["x"])), int(round(click["y"]))
        positive = int(click["label"]) == 1
        color = (0, 255, 0) if positive else (0, 0, 255)
        cv2.circle(out, (x, y), 7, color, -1)
        cv2.circle(out, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(
            out,
            f"{i}:{'+' if positive else '-'}",
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return out


def _setup_zoom_view(img, max_w=1400, max_h=900):
    h, w = img.shape[:2]
    base_scale = min(max_w / float(w), max_h / float(h), 1.0)
    disp_w = max(1, int(round(w * base_scale)))
    disp_h = max(1, int(round(h * base_scale)))
    return {
        "img": img,
        "h": h,
        "w": w,
        "disp_w": disp_w,
        "disp_h": disp_h,
        "zoom": 1.0,
        "min_zoom": 1.0,
        "max_zoom": 16.0,
        "center_x": w / 2.0,
        "center_y": h / 2.0,
        "last_mouse": (disp_w // 2, disp_h // 2),
        "dragging": False,
        "last_drag": (disp_w // 2, disp_h // 2),
    }


def _clamp_zoom_center(view, cx, cy):
    view_w = view["w"] / view["zoom"]
    view_h = view["h"] / view["zoom"]
    half_w = view_w / 2.0
    half_h = view_h / 2.0
    cx = max(half_w, min(view["w"] - half_w, cx))
    cy = max(half_h, min(view["h"] - half_h, cy))
    return cx, cy


def _zoom_view_rect(view):
    view_w = view["w"] / view["zoom"]
    view_h = view["h"] / view["zoom"]
    cx, cy = _clamp_zoom_center(view, view["center_x"], view["center_y"])
    view["center_x"], view["center_y"] = cx, cy
    x0 = int(round(cx - view_w / 2.0))
    y0 = int(round(cy - view_h / 2.0))
    x1 = int(round(x0 + view_w))
    y1 = int(round(y0 + view_h))
    x0 = max(0, min(view["w"] - 1, x0))
    y0 = max(0, min(view["h"] - 1, y0))
    x1 = max(x0 + 1, min(view["w"], x1))
    y1 = max(y0 + 1, min(view["h"], y1))
    return x0, y0, x1, y1


def _zoom_disp_to_orig(view, x, y):
    x0, y0, x1, y1 = _zoom_view_rect(view)
    ox = x0 + (x / max(view["disp_w"] - 1, 1)) * (x1 - x0)
    oy = y0 + (y / max(view["disp_h"] - 1, 1)) * (y1 - y0)
    ox = max(0.0, min(float(view["w"] - 1), ox))
    oy = max(0.0, min(float(view["h"] - 1), oy))
    return ox, oy


def _zoom_disp_delta_to_orig(view, ddx, ddy):
    x0, y0, x1, y1 = _zoom_view_rect(view)
    ox = (ddx / max(view["disp_w"] - 1, 1)) * (x1 - x0)
    oy = (ddy / max(view["disp_h"] - 1, 1)) * (y1 - y0)
    return ox, oy


def _zoom_at_disp_point(view, x, y, new_zoom):
    """Zoom while keeping the image point under the display cursor stationary."""
    new_zoom = max(view["min_zoom"], min(view["max_zoom"], new_zoom))
    if new_zoom == view["zoom"]:
        return
    ox, oy = _zoom_disp_to_orig(view, x, y)
    fx = max(0.0, min(1.0, x / max(view["disp_w"] - 1, 1)))
    fy = max(0.0, min(1.0, y / max(view["disp_h"] - 1, 1)))
    view["zoom"] = new_zoom
    new_view_w = view["w"] / view["zoom"]
    new_view_h = view["h"] / view["zoom"]
    view["center_x"], view["center_y"] = _clamp_zoom_center(
        view,
        ox + (0.5 - fx) * new_view_w,
        oy + (0.5 - fy) * new_view_h,
    )
    view["last_mouse"] = (x, y)


def collect_correction_clicks(frame_bgr, obj_id: int, frame_idx: int, preview_max_width: int):
    """Collect positive/negative clicks and optional box in processed-frame coordinates."""
    clicks = []
    box = None
    box_start = None
    box_current = None
    mode = "positive"
    view = _setup_zoom_view(frame_bgr, max_w=preview_max_width if preview_max_width > 0 else 1400)
    win = f"Correct obj {obj_id} frame {frame_idx}"

    def normalize_box(p0, p1):
        x0 = max(0.0, min(float(view["w"] - 1), min(p0[0], p1[0])))
        y0 = max(0.0, min(float(view["h"] - 1), min(p0[1], p1[1])))
        x1 = max(0.0, min(float(view["w"] - 1), max(p0[0], p1[0])))
        y1 = max(0.0, min(float(view["h"] - 1), max(p0[1], p1[1])))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [x0, y0, x1, y1]

    def on_mouse(event, x, y, flags, userdata):
        nonlocal box, box_start, box_current
        view["last_mouse"] = (x, y)
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                _zoom_at_disp_point(view, x, y, view["zoom"] * INTERACTIVE_ZOOM_STEP)
            else:
                _zoom_at_disp_point(view, x, y, view["zoom"] / INTERACTIVE_ZOOM_STEP)
            return
        if mode == "box":
            if event == cv2.EVENT_LBUTTONDOWN:
                box_start = _zoom_disp_to_orig(view, x, y)
                box_current = box_start
                return
            if event == cv2.EVENT_MOUSEMOVE and box_start is not None:
                box_current = _zoom_disp_to_orig(view, x, y)
                return
            if event == cv2.EVENT_LBUTTONUP and box_start is not None:
                box_current = _zoom_disp_to_orig(view, x, y)
                box = normalize_box(box_start, box_current)
                box_start = None
                box_current = None
                return
        if event == cv2.EVENT_LBUTTONDOWN and mode in ("positive", "negative"):
            px, py = _zoom_disp_to_orig(view, x, y)
            clicks.append({
                "x": px,
                "y": py,
                "label": 1 if mode == "positive" else 0,
            })

    _open_cv_window(win, view["disp_w"], view["disp_h"])
    cv2.setMouseCallback(win, on_mouse)
    accepted = False
    try:
        while True:
            x0, y0, x1, y1 = _zoom_view_rect(view)
            crop = frame_bgr[y0:y1, x0:x1]
            vis = cv2.resize(crop, (view["disp_w"], view["disp_h"]), interpolation=cv2.INTER_LINEAR)
            visible_clicks = []
            for click in clicks:
                if x0 <= click["x"] < x1 and y0 <= click["y"] < y1:
                    visible_clicks.append({
                        "x": (click["x"] - x0) * view["disp_w"] / max(x1 - x0, 1),
                        "y": (click["y"] - y0) * view["disp_h"] / max(y1 - y0, 1),
                        "label": click["label"],
                    })
            vis = _draw_correction_points(vis, visible_clicks)
            visible_box = box
            if box_start is not None and box_current is not None:
                visible_box = normalize_box(box_start, box_current)
            if visible_box is not None:
                bx0, by0, bx1, by1 = visible_box
                if bx1 >= x0 and bx0 < x1 and by1 >= y0 and by0 < y1:
                    dx0 = int(round((bx0 - x0) * view["disp_w"] / max(x1 - x0, 1)))
                    dy0 = int(round((by0 - y0) * view["disp_h"] / max(y1 - y0, 1)))
                    dx1 = int(round((bx1 - x0) * view["disp_w"] / max(x1 - x0, 1)))
                    dy1 = int(round((by1 - y0) * view["disp_h"] / max(y1 - y0, 1)))
                    dx0 = max(0, min(view["disp_w"] - 1, dx0))
                    dy0 = max(0, min(view["disp_h"] - 1, dy0))
                    dx1 = max(0, min(view["disp_w"] - 1, dx1))
                    dy1 = max(0, min(view["disp_h"] - 1, dy1))
                    cv2.rectangle(vis, (dx0, dy0), (dx1, dy1), (0, 255, 255), 3)
                    cv2.rectangle(vis, (dx0, dy0), (dx1, dy1), (0, 0, 0), 1)
            cv2.putText(
                vis,
                "1 positive LMB   0 negative LMB   b box LMB-drag   x clear box   WASD pan   wheel/+/- zoom",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"obj={obj_id} frame={frame_idx} mode={mode} zoom={view['zoom']:.1f}x clicks={len(clicks)} box={'yes' if box else 'no'}  u undo Enter apply Esc cancel",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u") and clicks:
                clicks.pop()
            elif key == ord("b"):
                mode = "box"
            elif key == ord("1"):
                mode = "positive"
            elif key == ord("0"):
                mode = "negative"
            elif key == ord("x"):
                box = None
                box_start = None
                box_current = None
            elif key in (ord("+"), ord("="), ord("-"), ord("_")):
                if key in (ord("+"), ord("=")):
                    _zoom_at_disp_point(view, *view["last_mouse"], view["zoom"] * INTERACTIVE_ZOOM_STEP)
                else:
                    _zoom_at_disp_point(view, *view["last_mouse"], view["zoom"] / INTERACTIVE_ZOOM_STEP)
            elif key in (ord("w"), ord("a"), ord("s"), ord("d")):
                step = max(10.0, 60.0 / float(view["zoom"]))
                if key == ord("a"):
                    view["center_x"] -= step
                elif key == ord("d"):
                    view["center_x"] += step
                elif key == ord("w"):
                    view["center_y"] -= step
                elif key == ord("s"):
                    view["center_y"] += step
            elif key in (13, 10, 32):
                accepted = True
                break
            elif key == 27:
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyWindow(win)

    if not accepted or (not clicks and box is None):
        return None
    return {"clicks": clicks, "box": box}


def _apply_correction_to_state(predictor, state, correction, scale: float):
    pts = []
    labels = []
    for x, y in correction.get("positive", []):
        pts.append([float(x) * scale, float(y) * scale])
        labels.append(1)
    for x, y in correction.get("negative", []):
        pts.append([float(x) * scale, float(y) * scale])
        labels.append(0)
    box = correction.get("box")
    scaled_box = None
    if box is not None:
        scaled_box = np.asarray([float(v) * scale for v in box], dtype=np.float32)
    if not pts and scaled_box is None:
        return
    clear_old_points = scaled_box is not None
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=int(correction["frame"]),
        obj_id=int(correction["obj_id"]),
        points=np.asarray(pts, dtype=np.float32) if pts else None,
        labels=np.asarray(labels, dtype=np.int32) if labels else None,
        box=scaled_box,
        clear_old_points=clear_old_points,
    )


def _make_correction_record(frame_idx: int, obj_id: int, correction_data, scale: float):
    positive = []
    negative = []
    clicks = correction_data.get("clicks", []) if isinstance(correction_data, dict) else correction_data
    for click in clicks:
        pt = [float(click["x"]) / scale, float(click["y"]) / scale]
        if int(click["label"]) == 1:
            positive.append(pt)
        else:
            negative.append(pt)
    record = {
        "frame": int(frame_idx),
        "obj_id": int(obj_id),
        "positive": positive,
        "negative": negative,
    }
    if isinstance(correction_data, dict) and correction_data.get("box") is not None:
        record["box"] = [float(v) / scale for v in correction_data["box"]]
    return record


def _correction_frame_bounds(corrections_payload, side_key: str, active_obj_ids):
    active = set(int(v) for v in active_obj_ids)
    frames = [
        int(corr.get("frame", 0))
        for corr in corrections_for_side(corrections_payload, side_key)
        if int(corr.get("obj_id", -1)) in active
    ]
    if not frames:
        return None, None
    return min(frames), max(frames)


def _release_sam2_state(state, device: str):
    """Best-effort release of SAM2 inference state between clean restarts."""
    try:
        if isinstance(state, dict):
            state.clear()
    except Exception:
        pass
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_rich_progress():
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

# ---- MAIN ----
def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM2 marker tracking on LEFT and RIGHT videos.")
    parser.add_argument("--left-input", default=str(Path("in") / "left.mp4"), help="Path to LEFT input video file")
    parser.add_argument("--right-input", default=str(Path("in") / "right.mp4"), help="Path to RIGHT input video file")
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive source frame for a virtual clip")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive source frame for a virtual clip")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output root directory")
    parser.add_argument(
        "--corrections-json",
        default=str(DEFAULT_CORRECTIONS_PATH),
        help="Corrections JSON path. Use a per-run path when keeping multiple setups.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Interactively click points, save work/manual_sam2_setup, then exit before SAM2 compute.",
    )
    parser.add_argument(
        "--reuse-setup",
        action="store_true",
        help="Reuse saved crop/point setup from work/manual_sam2_setup and run headless.",
    )
    parser.add_argument(
        "--modify-setup",
        action="store_true",
        help="Load saved setup, let you add/edit prompt points, save it, then exit before SAM2 compute.",
    )
    parser.add_argument(
        "--semi-auto-setup",
        action="store_true",
        help="Generate a row/column prompt grid from detected painted markers, review/edit it, save setup, then exit before SAM2 compute.",
    )
    parser.add_argument(
        "--semi-auto-setup-sections",
        action="store_true",
        help="Build one setup from multiple locally rectangular semi-auto grid sections, then exit before SAM2 compute.",
    )
    parser.add_argument("--grid-cols", type=int, default=None, help="Semi-auto grid columns; prompts when omitted.")
    parser.add_argument("--grid-rows", type=int, default=None, help="Semi-auto grid rows; prompts when omitted.")
    parser.add_argument(
        "--section-layout",
        default=None,
        help="Sectioned semi-auto layout as 'colsxrows,colsxrows' (example: 6x6,8x4). Prompts when omitted.",
    )
    parser.add_argument(
        "--left-crop",
        default=None,
        help="Optional LEFT crop ROI in x,y,w,h format, or 'none' to force full frame",
    )
    parser.add_argument(
        "--right-crop",
        default=None,
        help="Optional RIGHT crop ROI in x,y,w,h format, or 'none' to force full frame",
    )
    parser.add_argument(
        "--select-crop",
        nargs="?",
        const=True,
        type=str_to_bool,
        default=False,
        help="Interactively select crop ROI for both videos, then review across frames",
    )
    parser.add_argument(
        "--points-json",
        default=None,
        help="Optional shared points JSON with {'left': [[x,y]], 'right': [[x,y]]}; skips interactive clicking",
    )
    parser.add_argument(
        "--left-points",
        default=None,
        help="Optional LEFT points CSV/JSON from auto_detect_sam2_prompts; skips interactive clicking when paired with --right-points",
    )
    parser.add_argument(
        "--right-points",
        default=None,
        help="Optional RIGHT points CSV/JSON from auto_detect_sam2_prompts; skips interactive clicking when paired with --left-points",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Keep loaded video frames in CPU memory to reduce GPU usage (recommended for high-res videos)",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Offload inference state to CPU (lower GPU memory, slower tracking)",
    )
    parser.add_argument(
        "--async-loading-frames",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Enable async frame loading during init_state",
    )
    parser.add_argument(
        "--save-masks",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Save per-object binary mask images. Usually unnecessary because --save-tracks writes mask-derived dot CSVs directly.",
    )
    parser.add_argument(
        "--save-tracks",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Save direct centroid tracks_2d.csv during SAM2 propagation",
    )
    parser.add_argument(
        "--save-overlay",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Save overlay.mp4 during SAM2 propagation",
    )
    parser.add_argument(
        "--preview",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Show live overlay preview during propagation (press q or Esc to stop early)",
    )
    parser.add_argument(
        "--preview-max-width",
        type=int,
        default=1400,
        help="Max width for preview window (keeps display responsive)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Processing scale applied after crop before SAM2. Tracks are written back in original full-frame coordinates.",
    )
    parser.add_argument(
        "--frame-extractor",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="Frame extraction backend. auto uses ffmpeg when available and falls back to OpenCV.",
    )
    parser.add_argument(
        "--gpu-mode",
        choices=["auto", "dual", "single", "4090-only"],
        default="auto",
        help="GPU scheduling mode: auto (default), dual (force 2 GPUs), single (force one GPU/CPU), 4090-only (force NVIDIA 4090 only)",
    )
    parser.add_argument(
        "--single-gpu-index",
        type=int,
        default=None,
        help="GPU index used when --gpu-mode=single. If omitted with multiple GPUs, prompt interactively.",
    )
    parser.add_argument(
        "--correction-restart-mode",
        choices=["full", "reseed-from-correction-frame"],
        default="full",
        help="Correction restart behavior. full restarts from frame 0; reseed-from-correction-frame is experimental and restarts from the earliest corrected frame using current tracked marker positions as prompts.",
    )
    parser.add_argument(
        "--auto-pause-missing",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Automatically pause preview when fewer valid markers are visible than expected.",
    )
    return parser.parse_args()


def format_cuda_device_line(idx: int, name: str) -> str:
    try:
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        return f"{idx}: {name} ({total_gb:.1f} GiB)"
    except Exception:
        return f"{idx}: {name}"


def choose_single_gpu_index(gpu_names):
    gpu_count = len(gpu_names)
    if gpu_count <= 0:
        return None
    if gpu_count == 1:
        print(f"[INFO] Single CUDA GPU detected: {format_cuda_device_line(0, gpu_names[0])}")
        return 0

    print("\nAvailable CUDA GPUs:")
    for idx, name in enumerate(gpu_names):
        print(f"  {format_cuda_device_line(idx, name)}")

    if not sys.stdin.isatty():
        print("[INFO] Non-interactive terminal detected; defaulting --gpu-mode single to cuda:0.")
        return 0

    while True:
        raw = input(f"Select GPU for --gpu-mode single [0-{gpu_count - 1}] (Enter=0): ").strip()
        if raw == "":
            return 0
        try:
            idx = int(raw)
        except ValueError:
            print(f"Invalid GPU index: {raw!r}")
            continue
        if 0 <= idx < gpu_count:
            return idx
        print(f"GPU index out of range. Choose 0-{gpu_count - 1}.")

def _video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps > 0 else 30.0


def _invalid_track_row(frame_idx: int, obj_id: int, method: str):
    return {
        "frame": int(frame_idx),
        "obj_id": int(obj_id),
        "u": "",
        "v": "",
        "u_local": "",
        "v_local": "",
        "method": method,
        "area_px": 0,
        "core_radius_px": 0.0,
        "solidity": 0.0,
        "quality": 0.0,
        "valid": 0,
    }


def _clean_largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _mask_solidity(mask: np.ndarray):
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0, 0.0

    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return area, 1.0

    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    if hull is None or len(hull) < 3:
        return area, 1.0

    hull_area = cv2.contourArea(hull)
    return area, float(area / max(hull_area, 1.0))


def _distance_transform_peak(mask: np.ndarray):
    if not np.any(mask):
        return None, None, 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return float(u), float(v), float(dt[v, u])


def _eroded_centroid(mask: np.ndarray, kernel_size: int = DOT_ERODE_KERNEL_SIZE):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ys, xs = np.nonzero(eroded)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _dot_quality(area: int, core_radius: float, solidity: float, bbox_h: int, bbox_w: int) -> float:
    quality = 1.0
    if area < DOT_AREA_MIN or area > DOT_AREA_MAX:
        quality *= 0.5
    if core_radius < DOT_CORE_RADIUS_MIN:
        quality *= 0.8
    if solidity < DOT_SOLIDITY_MIN:
        quality *= 0.7
    if bbox_h > 0 and bbox_w > 0:
        aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)
        if aspect > 5:
            quality *= 0.6
    return float(np.clip(quality, 0.0, 1.0))


def _mask_to_track_row(frame_idx: int, obj_id: int, mask_2d, crop, scale: float = 1.0):
    m = np.squeeze(mask_2d)
    if torch.is_tensor(m):
        m = m.detach().float().cpu().numpy()
    m_bin = m > 0
    x_off = int(crop[0]) if crop is not None else 0
    y_off = int(crop[1]) if crop is not None else 0
    if not np.any(m_bin):
        return _invalid_track_row(frame_idx, obj_id, "invalid")

    mask = _clean_largest_component(m_bin)
    if not np.any(mask):
        return _invalid_track_row(frame_idx, obj_id, "invalid_after_cleanup")

    u, v, core_radius = _distance_transform_peak(mask)
    method = "dt_peak"
    if u is None or v is None:
        fallback = _eroded_centroid(mask)
        if fallback is not None:
            u, v = fallback
            method = "fallback_eroded_centroid"
        else:
            fallback = _centroid(mask)
            if fallback is None:
                return _invalid_track_row(frame_idx, obj_id, "no_fallback")
            u, v = fallback
            method = "fallback_centroid"

    area_scaled, solidity = _mask_solidity(mask)
    ys, xs = np.nonzero(mask)
    bbox_h = int(np.max(ys) - np.min(ys) + 1) if len(ys) else 1
    bbox_w = int(np.max(xs) - np.min(xs) + 1) if len(xs) else 1

    inv_scale = 1.0 / float(scale)
    u_local = float(u * inv_scale)
    v_local = float(v * inv_scale)
    area_original = int(round(area_scaled * inv_scale * inv_scale))
    core_radius_original = float(core_radius * inv_scale)
    quality = _dot_quality(area_original, core_radius_original, solidity, bbox_h, bbox_w)

    return {
        "frame": int(frame_idx),
        "obj_id": int(obj_id),
        "u": float(u_local + x_off),
        "v": float(v_local + y_off),
        "u_local": u_local,
        "v_local": v_local,
        "method": method,
        "area_px": area_original,
        "core_radius_px": core_radius_original,
        "solidity": float(solidity),
        "quality": quality,
        "valid": 1 if quality > 0 and area_original > 0 else 0,
    }


def _valid_marker_count(rows_by_obj, active_obj_ids):
    count = 0
    missing = []
    for obj_id in active_obj_ids:
        row = rows_by_obj.get(int(obj_id))
        if row is not None and int(row.get("valid", 0)) == 1:
            count += 1
        else:
            missing.append(int(obj_id))
    return count, len(active_obj_ids), missing


def _pause_reason_text(auto_pause, valid_count=None, expected_count=None, missing_ids=None):
    if not auto_pause:
        return ""
    shown_missing = list(missing_ids or [])[:20]
    suffix = "..." if missing_ids is not None and len(missing_ids) > 20 else ""
    return f"  AUTO missing {shown_missing}{suffix} ({valid_count}/{expected_count})"


def draw_id_labels_on_processed_frame(frame_bgr,
                                      rows_by_obj,
                                      coordinate_scale: float,
                                      font_scale: float = 0.65,
                                      text_thickness: int = 2,
                                      marker_radius: int = 4,
                                      reserved_top_px: int = 0,
                                      visible_obj_ids=None):
    """Draw object IDs on a processed-frame preview from original-scale local track rows."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    visible_set = None if visible_obj_ids is None else {int(v) for v in visible_obj_ids}
    for obj_id, row in rows_by_obj.items():
        if visible_set is not None and int(obj_id) not in visible_set:
            continue
        if int(row.get("valid", 0)) != 1:
            continue
        try:
            x = int(round(float(row["u_local"]) * coordinate_scale))
            y = int(round(float(row["v_local"]) * coordinate_scale))
        except (TypeError, ValueError):
            continue

        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        label = str(obj_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(text_thickness))
        bg_pad = 3
        text_w, text_h = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = x + marker_radius + 4
        ty = y - marker_radius - 4
        tx = int(round(max(bg_pad, min(w - text_w - bg_pad, tx))))
        ty = int(round(max(max(text_h + bg_pad, reserved_top_px + text_h + bg_pad), min(h - bg_pad, ty))))

        cv2.circle(out, (x, y), marker_radius, (0, 255, 255), -1)
        cv2.circle(out, (x, y), marker_radius + 2, (0, 0, 0), 1)
        cv2.rectangle(
            out,
            (tx - bg_pad, ty - text_h - bg_pad),
            (tx + text_w + bg_pad, ty + bg_pad),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            out,
            label,
            (tx, ty),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def prompt_obj_id_in_preview(preview_win: str, preview_bgr, active_obj_ids, side_name: str, frame_idx: int):
    """Prompt for an object ID inside the OpenCV preview window."""
    active_ids = sorted(int(o) for o in active_obj_ids)
    active_id_set = set(active_ids)
    if active_ids:
        active_summary = f"{active_ids[0]}-{active_ids[-1]} ({len(active_ids)} active)"
    else:
        active_summary = "none"
    digit_buf = ""
    message = "Type obj id, Enter apply, Backspace edit, Esc cancel"

    while True:
        vis = preview_bgr.copy()
        h, w = vis.shape[:2]
        box_h = 112
        cv2.rectangle(vis, (0, 0), (w, min(h, box_h)), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"{side_name} frame={frame_idx} correction",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            message,
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"obj_id: {digit_buf}_   active: {active_summary}",
            (12, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(preview_win, vis)

        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(preview_win, cv2.WND_PROP_VISIBLE) < 1:
                return None
            continue
        key = key & 0xFF

        if key == 27:
            return None
        if key in (8, 127):
            digit_buf = digit_buf[:-1]
            message = "Type obj id, Enter apply, Backspace edit, Esc cancel"
            continue
        if key in (13, 10):
            if not digit_buf:
                message = "Enter an object ID first"
                continue
            obj_id = int(digit_buf)
            if obj_id not in active_id_set:
                message = f"obj_id {obj_id} is not active in this frame"
                continue
            return obj_id
        if ord("0") <= key <= ord("9"):
            digit_buf += chr(key)
            message = "Type obj id, Enter apply, Backspace edit, Esc cancel"


def prompt_id_visibility_in_preview(preview_win: str, preview_bgr, total_obj_count: int, hidden_obj_ids):
    """Toggle one object ID's preview visibility inside the OpenCV window."""
    digit_buf = ""
    message = "Type id to toggle, Enter apply, a show all, Backspace edit, Esc cancel"
    hidden_obj_ids = set(int(v) for v in hidden_obj_ids)
    max_id = max(0, int(total_obj_count) - 1)

    while True:
        visible_count = int(total_obj_count) - len(hidden_obj_ids)
        hidden_preview = sorted(hidden_obj_ids)[:12]
        vis = preview_bgr.copy()
        h, w = vis.shape[:2]
        box_h = 112
        cv2.rectangle(vis, (0, 0), (w, min(h, box_h)), (0, 0, 0), -1)
        cv2.putText(
            vis,
            "Toggle ID Visibility",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            message,
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"id: {digit_buf}_   visible={visible_count}/{total_obj_count}   hidden={hidden_preview}{' ...' if len(hidden_obj_ids) > 12 else ''}",
            (12, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(preview_win, vis)

        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(preview_win, cv2.WND_PROP_VISIBLE) < 1:
                return hidden_obj_ids
            continue
        key = key & 0xFF

        if key == 27:
            return hidden_obj_ids
        if key == ord("a"):
            hidden_obj_ids.clear()
            message = "All IDs visible"
            digit_buf = ""
            continue
        if key in (8, 127):
            digit_buf = digit_buf[:-1]
            message = "Type id to toggle, Enter apply, a show all, Backspace edit, Esc cancel"
            continue
        if key in (13, 10):
            if not digit_buf:
                message = "Enter an ID first"
                continue
            obj_id = int(digit_buf)
            if obj_id < 0 or obj_id > max_id:
                message = f"id {obj_id} is outside 0-{max_id}"
                continue
            if obj_id in hidden_obj_ids:
                hidden_obj_ids.remove(obj_id)
                message = f"id {obj_id} visible"
            else:
                hidden_obj_ids.add(obj_id)
                message = f"id {obj_id} hidden"
            digit_buf = ""
            continue
        if ord("0") <= key <= ord("9"):
            digit_buf += chr(key)
            message = "Type id to toggle, Enter apply, a show all, Backspace edit, Esc cancel"


def prompt_frame_in_preview(preview_win: str, preview_bgr, side_name: str, current_frame_idx: int, total_frames: int):
    """Prompt for a correction frame inside the OpenCV preview window."""
    digit_buf = str(int(current_frame_idx))
    max_frame = max(0, int(total_frames) - 1)
    message = "Type correction frame, Enter apply, Backspace edit, Esc cancel"

    while True:
        vis = preview_bgr.copy()
        h, w = vis.shape[:2]
        box_h = 112
        cv2.rectangle(vis, (0, 0), (w, min(h, box_h)), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"{side_name} correction frame",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            message,
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"frame: {digit_buf}_   current={current_frame_idx}   range=0-{max_frame}",
            (12, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(preview_win, vis)

        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(preview_win, cv2.WND_PROP_VISIBLE) < 1:
                return None
            continue
        key = key & 0xFF

        if key == 27:
            return None
        if key in (8, 127):
            digit_buf = digit_buf[:-1]
            message = "Type correction frame, Enter apply, Backspace edit, Esc cancel"
            continue
        if key in (13, 10):
            if not digit_buf:
                message = "Enter a frame number first"
                continue
            frame_idx = int(digit_buf)
            if frame_idx < 0 or frame_idx > max_frame:
                message = f"Frame {frame_idx} is outside 0-{max_frame}"
                continue
            return frame_idx
        if ord("0") <= key <= ord("9"):
            digit_buf += chr(key)
            message = "Type correction frame, Enter apply, Backspace edit, Esc cancel"


def render_preview_frame(frame_bgr,
                         rows_by_obj,
                         scale: float,
                         preview_max_width: int,
                         header_text: str,
                         zoom: float = 1.0,
                         center_xy=None,
                         reserved_top_px: int = 42,
                         visible_obj_ids=None):
    """Render preview labels after optional zoom crop and output resize."""
    h, w = frame_bgr.shape[:2]
    zoom = max(1.0, float(zoom))
    if center_xy is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = float(center_xy[0]), float(center_xy[1])

    view_w = max(1, int(round(w / zoom)))
    view_h = max(1, int(round(h / zoom)))
    x0 = int(round(cx - view_w / 2.0))
    y0 = int(round(cy - view_h / 2.0))
    x0 = max(0, min(w - view_w, x0))
    y0 = max(0, min(h - view_h, y0))
    x1 = min(w, x0 + view_w)
    y1 = min(h, y0 + view_h)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    vis = frame_bgr[y0:y1, x0:x1].copy()
    adjusted_rows = {}
    for obj_id, row in rows_by_obj.items():
        if int(row.get("valid", 0)) != 1:
            continue
        try:
            px = float(row["u_local"]) * scale
            py = float(row["v_local"]) * scale
        except (TypeError, ValueError):
            continue
        if not (x0 <= px < x1 and y0 <= py < y1):
            continue
        adjusted = dict(row)
        adjusted["u_local"] = (px - x0) / float(scale)
        adjusted["v_local"] = (py - y0) / float(scale)
        adjusted_rows[int(obj_id)] = adjusted

    preview_resize_scale = 1.0
    if preview_max_width > 0 and vis.shape[1] > preview_max_width:
        preview_resize_scale = preview_max_width / float(vis.shape[1])
        new_h = max(1, int(round(vis.shape[0] * preview_resize_scale)))
        vis = cv2.resize(vis, (preview_max_width, new_h), interpolation=cv2.INTER_AREA)

    vis = draw_id_labels_on_processed_frame(
        vis,
        adjusted_rows,
        scale * preview_resize_scale,
        font_scale=0.65,
        text_thickness=2,
        marker_radius=4,
        reserved_top_px=reserved_top_px,
        visible_obj_ids=visible_obj_ids,
    )
    cv2.putText(
        vis,
        header_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    return vis, (cx, cy), preview_resize_scale


def run_single_video_tracking(side_name: str,
                              video_path: str,
                              out_dir: Path,
                              points,
                              crop,
                              scale: float,
                              frame_extractor: str,
                              predictor,
                              device: str,
                              offload_video_to_cpu: bool,
                              offload_state_to_cpu: bool,
                              async_loading_frames: bool,
                              save_masks: bool,
                              save_tracks: bool,
                              save_overlay: bool,
                              preview: bool,
                              preview_max_width: int,
                              corrections_payload=None,
                              corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
                              object_ids=None,
                              frames_dir_override=None,
                              shared_state=None,
                              release_state=True,
                              preview_controller=None,
                              correction_restart_mode: str = "full",
                              auto_pause_missing: bool = False,
                              progress_event_q=None):
    object_ids = list(range(len(points))) if object_ids is None else [int(obj_id) for obj_id in object_ids]
    if len(object_ids) != len(points):
        raise ValueError("object_ids must have the same length as points")
    active_obj_ids = list(object_ids)
    visibility_total_count = max(active_obj_ids) + 1 if active_obj_ids else len(points)

    frames_dir = Path(frames_dir_override) if frames_dir_override is not None else out_dir / "frames"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if frames_dir_override is None:
        prepare_frame_cache(
            video_path,
            frames_dir,
            crop=crop,
            scale=scale,
            frame_extractor=frame_extractor,
            label=side_name,
        )
    else:
        print(f"[INFO][{side_name}] Using prepared frame cache: {frames_dir}")
    first_frame = load_first_frame(frames_dir)
    scaled_points = []
    for point in points:
        x, y, label = _coerce_prompt_point(point, prefer_local=False)
        scaled_negatives = [
            [float(nx) * scale, float(ny) * scale]
            for nx, ny in _coerce_negative_points(point)
        ]
        scaled_points.append(_make_prompt_point(float(x) * scale, float(y) * scale, label, scaled_negatives))

    (out_dir / "prompts").mkdir(parents=True, exist_ok=True)
    side_key = str(side_name).lower()
    corrections_payload = corrections_payload if corrections_payload is not None else _empty_corrections_payload()
    side_corrections = [
        corr for corr in corrections_for_side(corrections_payload, side_key)
        if int(corr.get("obj_id", -1)) in set(active_obj_ids)
    ]
    with open(out_dir / "prompts" / "frame0_points.json", "w") as f:
        json.dump(points, f, indent=2)
    with open(out_dir / "prompts" / "frame0_points_scaled.json", "w") as f:
        json.dump(scaled_points, f, indent=2)
    with open(out_dir / "prompts" / "corrections_applied.json", "w") as f:
        json.dump(side_corrections, f, indent=2)

    crop_meta = {
        "side": str(side_name),
        "video": str(video_path),
        "crop_applied": crop is not None,
        "x": int(crop[0]) if crop is not None else 0,
        "y": int(crop[1]) if crop is not None else 0,
        "w": int(crop[2]) if crop is not None else int(first_frame.shape[1]),
        "h": int(crop[3]) if crop is not None else int(first_frame.shape[0]),
        "coordinate_space": "cropped" if crop is not None else "original",
        "processing_scale": float(scale),
        "processed_w": int(first_frame.shape[1]),
        "processed_h": int(first_frame.shape[0]),
        "notes": "Tracks are written in original full-frame coordinates. Scaled prompt points are used internally when --scale is not 1.0.",
    }
    with open(out_dir / "prompts" / "crop_roi.json", "w") as f:
        json.dump(crop_meta, f, indent=2)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    if hasattr(predictor, "add_all_frames_to_correct_as_cond"):
        predictor.add_all_frames_to_correct_as_cond = True
    # Keep this disabled for the bundled predictor: it calls a missing helper
    # named _clear_obj_non_cond_mem_around_input when enabled.
    if hasattr(predictor, "clear_non_cond_mem_around_input"):
        predictor.clear_non_cond_mem_around_input = False

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else torch.cpu.amp.autocast(enabled=False)
    )

    with torch.inference_mode(), autocast_ctx:
        restart_start_frame_idx = 0
        restart_seed_points = None

        def build_reseed_points_from_tracks(frame_idx: int):
            rows = {
                int(r["obj_id"]): r
                for r in track_rows_data
                if int(r["frame"]) == int(frame_idx) and int(r.get("valid", 0)) == 1
            }
            seed_points = {}
            missing = []
            for obj_id in active_obj_ids:
                row = rows.get(int(obj_id))
                if row is None:
                    missing.append(int(obj_id))
                    continue
                seed_points[int(obj_id)] = (
                    float(row["u_local"]) * float(scale),
                    float(row["v_local"]) * float(scale),
                )
            if missing:
                raise RuntimeError(
                    f"[CORRECT][{side_name}] Cannot reseed from frame {frame_idx}: "
                    f"missing valid tracked positions for object IDs {missing[:20]}"
                    f"{'...' if len(missing) > 20 else ''}."
                )
            return seed_points

        def build_state_from_prompts(start_frame_idx: int = 0, seed_points_by_obj=None):
            if shared_state is None:
                state = predictor.init_state(
                    video_path=str(frames_dir),
                    offload_video_to_cpu=offload_video_to_cpu,
                    offload_state_to_cpu=offload_state_to_cpu,
                    async_loading_frames=async_loading_frames,
                )
            else:
                predictor.reset_state(shared_state)
                state = shared_state

            frame_idx = int(start_frame_idx)
            prompt_debug = []
            for point_idx, (obj_id, point) in enumerate(zip(active_obj_ids, scaled_points)):
                if seed_points_by_obj is not None:
                    seed_xy = seed_points_by_obj.get(int(obj_id))
                    if seed_xy is None:
                        raise RuntimeError(
                            f"[CORRECT][{side_name}] Cannot reseed from frame {frame_idx}: "
                            f"missing valid tracked position for obj_id={obj_id}."
                        )
                    pts = np.asarray([[float(seed_xy[0]), float(seed_xy[1])]], dtype=np.float32)
                    lbl = np.asarray([1], dtype=np.int32)
                    box = None
                else:
                    pts, lbl, box = build_setup_prompt(point, scale=1.0)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbl,
                    box=box,
                )

                orig_x, orig_y, orig_label = _coerce_prompt_point(points[point_idx], prefer_local=False)
                scaled_x, scaled_y, scaled_label = _coerce_prompt_point(point, prefer_local=False)
                prompt_debug.append({
                    "obj_id": int(obj_id),
                    "frame_idx": int(frame_idx),
                    "source": "reseed_track_position" if seed_points_by_obj is not None else "setup_prompt",
                    "click_xy": [float(orig_x), float(orig_y)],
                    "label": int(orig_label),
                    "scaled_click_xy": pts.reshape(-1, 2).tolist()[0],
                    "scaled_label": int(scaled_label),
                    "points": pts.tolist(),
                    "labels": lbl.tolist(),
                    "box": box.tolist() if box is not None else None,
                })

            with open(out_dir / "prompts" / "frame0_prompt_details.json", "w") as f:
                json.dump(prompt_debug, f, indent=2)

            current_side_corrections = [
                corr for corr in corrections_for_side(corrections_payload, side_key)
                if int(corr.get("obj_id", -1)) in set(active_obj_ids)
                and int(corr.get("frame", 0)) >= int(start_frame_idx)
            ]
            if current_side_corrections:
                print(f"[INFO][{side_name}] Applying {len(current_side_corrections)} saved correction(s)")
                for correction in sorted(current_side_corrections, key=lambda c: int(c.get("frame", 0))):
                    _apply_correction_to_state(predictor, state, correction, scale)
                with open(out_dir / "prompts" / "corrections_applied.json", "w") as f:
                    json.dump(current_side_corrections, f, indent=2)
            return state

        state = build_state_from_prompts()

        masks_root = out_dir / "masks"
        if save_masks:
            masks_root.mkdir(parents=True, exist_ok=True)
            for obj_id in active_obj_ids:
                (masks_root / f"{obj_id:02d}").mkdir(parents=True, exist_ok=True)

        def delete_outputs_from_frame(start_frame_idx: int):
            if save_masks and masks_root.exists():
                for obj_id in active_obj_ids:
                    obj_mask_dir = masks_root / f"{int(obj_id):02d}"
                    if not obj_mask_dir.exists():
                        continue
                    for mask_path in obj_mask_dir.glob("*.jpg"):
                        try:
                            if int(mask_path.stem) >= int(start_frame_idx):
                                mask_path.unlink()
                        except ValueError:
                            continue

        h0, w0 = first_frame.shape[:2]
        out_mp4 = out_dir / "overlay.mp4"
        video_writer = None
        if save_overlay:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(out_mp4), fourcc, _video_fps(video_path), (w0, h0))
            if not video_writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {out_mp4}")

        tracks_csv = out_dir / "tracks_2d.csv"
        track_rows_data = []
        total_extracted_frames = len(list(frames_dir.glob("*.jpg")))

        preview_win = f"SAM2 Preview [{side_name}] (q or Esc to stop)"
        hidden_preview_obj_ids = set()
        controller_preview = preview_controller is not None
        local_preview = bool(preview and not controller_preview)
        if local_preview:
            _open_cv_window(preview_win, preview_max_width, 900)

        stop_requested = False
        frames_seen = 0
        track_rows = 0
        local_progress = None
        local_progress_task = None
        use_local_progress = progress_event_q is None
        try:
            if use_local_progress:
                local_progress = make_rich_progress()
                local_progress.start()
            while True:
                clean_restart_requested = False
                propagate_kwargs = {}
                if int(restart_start_frame_idx) > 0:
                    propagate_kwargs["start_frame_idx"] = int(restart_start_frame_idx)
                if progress_event_q is not None:
                    progress_event_q.put({
                        "type": "reset",
                        "side": str(side_name),
                        "total": int(total_extracted_frames),
                        "completed": int(restart_start_frame_idx),
                        "description": f"{side_name} propagate",
                    })
                elif local_progress is not None:
                    if local_progress_task is None:
                        local_progress_task = local_progress.add_task(
                            f"{side_name} propagate",
                            total=int(total_extracted_frames),
                        )
                    local_progress.update(
                        local_progress_task,
                        completed=int(restart_start_frame_idx),
                        total=int(total_extracted_frames),
                    )
                for f_idx, obj_ids, masks in predictor.propagate_in_video(state, **propagate_kwargs):
                    if progress_event_q is not None:
                        progress_event_q.put({
                            "type": "update",
                            "side": str(side_name),
                            "total": int(total_extracted_frames),
                            "completed": int(f_idx) + 1,
                            "description": f"{side_name} propagate",
                        })
                    elif local_progress is not None and local_progress_task is not None:
                        local_progress.update(local_progress_task, completed=int(f_idx) + 1)
                    current_save_overlay = video_writer is not None
                    frames_seen = max(frames_seen, int(f_idx) + 1)
                    masks_by_obj = {}
                    frame_rows_by_obj = {}
                    if torch.is_tensor(masks):
                        masks_np = masks.detach().float().cpu().numpy()
                    else:
                        masks_np = np.array(masks)

                    if save_tracks:
                        track_rows_data = [r for r in track_rows_data if int(r["frame"]) != int(f_idx)]

                    for j, oid in enumerate(obj_ids):
                        m_2d = np.squeeze(masks_np[j])
                        track_row = _mask_to_track_row(f_idx, int(oid), m_2d, crop, scale=scale)
                        frame_rows_by_obj[int(oid)] = track_row
                        if save_tracks:
                            track_rows_data.append(track_row)
                        if save_masks:
                            m_bin = (m_2d > 0).astype(np.uint8) * 255
                            mask_filename = masks_root / f"{int(oid):02d}" / f"{f_idx:06d}.jpg"
                            cv2.imwrite(str(mask_filename), m_bin)
                        if current_save_overlay or preview or controller_preview:
                            masks_by_obj[int(oid)] = m_2d

                    if current_save_overlay or preview or controller_preview:
                        frame_path = frames_dir / f"{f_idx:06d}.jpg"
                        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                        if frame is None or frame.size == 0:
                            print(f"[WARN][{side_name}] Could not read frame {f_idx}: {frame_path}")
                            continue
                        if preview or controller_preview:
                            preview_masks_by_obj = {
                                obj_id: mask
                                for obj_id, mask in masks_by_obj.items()
                                if obj_id not in hidden_preview_obj_ids
                            }
                            over = overlay_masks(frame, preview_masks_by_obj)
                        else:
                            over = overlay_masks(frame, masks_by_obj)
                        if current_save_overlay:
                            overlay_labeled = draw_id_labels_on_processed_frame(
                                overlay_masks(frame, masks_by_obj),
                                frame_rows_by_obj,
                                scale,
                            )
                            video_writer.write(overlay_labeled)

                    if controller_preview:
                        while True:
                            visible_preview_obj_ids = [
                                obj_id for obj_id in active_obj_ids
                                if obj_id not in hidden_preview_obj_ids
                            ]
                            vis, _, _ = render_preview_frame(
                                over,
                                frame_rows_by_obj,
                                scale,
                                preview_max_width,
                                f"{side_name} frame={f_idx}  objs={len(obj_ids)}  p pause  v IDs  c correct  q/Esc stop",
                                visible_obj_ids=visible_preview_obj_ids,
                            )
                            preview_controller["event_q"].put({
                                "type": "frame",
                                "side": str(side_name),
                                "frame_idx": int(f_idx),
                                "image": vis,
                                "rows_by_obj": frame_rows_by_obj,
                                "obj_ids": [int(v) for v in obj_ids],
                                "active_obj_ids": [int(v) for v in active_obj_ids],
                                "visibility_total_count": int(visibility_total_count),
                                "frames_dir": str(frames_dir),
                                "total_frames": int(total_extracted_frames),
                                "scale": float(scale),
                                "hidden_obj_ids": sorted(int(v) for v in hidden_preview_obj_ids),
                            })
                            cmd = preview_controller["cmd_q"].get()
                            cmd_type = cmd.get("type")
                            if cmd_type == "continue":
                                break
                            if cmd_type == "stop":
                                print(f"[INFO][{side_name}] Stop requested at frame {f_idx}. Finalizing outputs...")
                                stop_requested = True
                                break
                            if cmd_type == "rerender":
                                hidden_preview_obj_ids = set(int(v) for v in cmd.get("hidden_obj_ids", []))
                                preview_masks_by_obj = {
                                    obj_id: mask
                                    for obj_id, mask in masks_by_obj.items()
                                    if obj_id not in hidden_preview_obj_ids
                                }
                                over = overlay_masks(frame, preview_masks_by_obj)
                                continue
                            if cmd_type == "restart":
                                corrections_payload = cmd.get("corrections_payload", corrections_payload)
                                restart_start_frame_idx = int(cmd.get("start_frame_idx", 0) or 0)
                                restart_seed_points = cmd.get("seed_points_by_obj")
                                print(
                                    f"[CORRECT][{side_name}] Saved correction(s). "
                                    f"Restarting cleanly from frame {restart_start_frame_idx}..."
                                )
                                if video_writer is not None:
                                    video_writer.release()
                                    video_writer = None
                                if save_overlay:
                                    print("[WARN] Correction restart invalidated partial overlay.mp4; disabling overlay for this run.")
                                    if out_mp4.exists():
                                        out_mp4.unlink()
                                if shared_state is None:
                                    _release_sam2_state(state, device)
                                else:
                                    predictor.reset_state(state)
                                if restart_start_frame_idx > 0 and restart_seed_points:
                                    restart_seed_points = {
                                        int(k): tuple(v) for k, v in restart_seed_points.items()
                                    }
                                    track_rows_data = [
                                        r for r in track_rows_data
                                        if int(r["frame"]) < int(restart_start_frame_idx)
                                    ]
                                    delete_outputs_from_frame(restart_start_frame_idx)
                                    frames_seen = int(restart_start_frame_idx)
                                    state = build_state_from_prompts(
                                        start_frame_idx=restart_start_frame_idx,
                                        seed_points_by_obj=restart_seed_points,
                                    )
                                else:
                                    restart_start_frame_idx = 0
                                    restart_seed_points = None
                                    track_rows_data = []
                                    delete_outputs_from_frame(0)
                                    frames_seen = 0
                                    state = build_state_from_prompts()
                                clean_restart_requested = True
                                break
                            print(f"[WARN][{side_name}] Unknown preview command: {cmd_type}")
                        if stop_requested or clean_restart_requested:
                            break

                    if local_preview:
                        correction_frame_idx = int(f_idx)
                        correction_frame_bgr = over
                        correction_rows_by_obj = frame_rows_by_obj
                        visible_preview_obj_ids = [
                            obj_id for obj_id in active_obj_ids
                            if obj_id not in hidden_preview_obj_ids
                        ]
                        vis, pause_center, _ = render_preview_frame(
                            over,
                            frame_rows_by_obj,
                            scale,
                            preview_max_width,
                            f"{side_name} frame={f_idx}  objs={len(obj_ids)}  p pause  v IDs  c correct  q/Esc stop",
                            visible_obj_ids=visible_preview_obj_ids,
                        )
                        cv2.imshow(preview_win, vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            print(f"[INFO][{side_name}] Stop requested at frame {f_idx}. Finalizing outputs...")
                            stop_requested = True
                            break
                        if key == ord("v"):
                            hidden_preview_obj_ids = prompt_id_visibility_in_preview(
                                preview_win,
                                vis,
                                visibility_total_count,
                                hidden_preview_obj_ids,
                            )
                            continue
                        valid_markers, expected_markers, missing_markers = _valid_marker_count(
                            frame_rows_by_obj,
                            active_obj_ids,
                        )
                        auto_pause = bool(auto_pause_missing) and expected_markers > 0 and valid_markers < expected_markers
                        if auto_pause and key != ord("p"):
                            print(
                                f"[INFO][{side_name}] Auto-pausing at frame {f_idx}: "
                                f"valid markers {valid_markers}/{expected_markers}; "
                                f"missing IDs {missing_markers[:20]}"
                                f"{'...' if len(missing_markers) > 20 else ''}."
                            )
                        if key == ord("p") or auto_pause:
                            pause_zoom = 1.0
                            pause_frame_idx = int(f_idx)
                            pause_corrections_added = 0
                            pause_correction_frames = []
                            pause_reason = _pause_reason_text(
                                auto_pause,
                                valid_markers,
                                expected_markers,
                                missing_markers,
                            )

                            def load_pause_frame(frame_idx):
                                if int(frame_idx) == int(f_idx):
                                    return over, dict(frame_rows_by_obj)
                                pause_path = frames_dir / f"{int(frame_idx):06d}.jpg"
                                pause_frame = cv2.imread(str(pause_path), cv2.IMREAD_COLOR)
                                if pause_frame is None or pause_frame.size == 0:
                                    return None, {}
                                pause_rows = {
                                    int(r["obj_id"]): r
                                    for r in track_rows_data
                                    if int(r["frame"]) == int(frame_idx)
                                }
                                return pause_frame, pause_rows

                            paused_frame, paused_rows = load_pause_frame(pause_frame_idx)
                            print(
                                f"[INFO][{side_name}] Paused at frame {f_idx}. "
                                "Left/Right frame, p resume, c correct, +/- zoom, WASD pan."
                            )
                            while True:
                                if paused_frame is None:
                                    paused_frame, paused_rows = over, dict(frame_rows_by_obj)
                                    pause_frame_idx = int(f_idx)
                                paused_vis, pause_center, _ = render_preview_frame(
                                    paused_frame,
                                    paused_rows,
                                    scale,
                                    preview_max_width,
                                    f"PAUSED {side_name} frame={pause_frame_idx}{pause_reason}  Left/Right frame  p resume  v IDs  c correct  +/- zoom  WASD pan",
                                    zoom=pause_zoom,
                                    center_xy=pause_center,
                                    visible_obj_ids=[
                                        obj_id for obj_id in active_obj_ids
                                        if obj_id not in hidden_preview_obj_ids
                                    ],
                                )
                                cv2.imshow(preview_win, paused_vis)
                                raw_pause_key = cv2.waitKeyEx(30)
                                if raw_pause_key == -1:
                                    if cv2.getWindowProperty(preview_win, cv2.WND_PROP_VISIBLE) < 1:
                                        stop_requested = True
                                        break
                                    continue
                                pause_key = raw_pause_key & 0xFF
                                if pause_key in (27, ord("q")):
                                    print(f"[INFO][{side_name}] Stop requested while paused at frame {f_idx}.")
                                    stop_requested = True
                                    break
                                if pause_key == ord("p"):
                                    if pause_corrections_added:
                                        save_corrections(corrections_payload, corrections_path)
                                        with open(out_dir / "prompts" / "corrections_applied.json", "w") as f:
                                            json.dump(corrections_for_side(corrections_payload, side_key), f, indent=2)
                                        print(
                                            f"[CORRECT][{side_name}] Saved {pause_corrections_added} correction(s). "
                                            "Restarting cleanly..."
                                        )
                                        if video_writer is not None:
                                            video_writer.release()
                                            video_writer = None
                                        if save_overlay:
                                            print("[WARN] Correction restart invalidated partial overlay.mp4; disabling overlay for this run.")
                                            if video_writer is not None:
                                                video_writer.release()
                                                video_writer = None
                                            if out_mp4.exists():
                                                out_mp4.unlink()
                                        if shared_state is None:
                                            _release_sam2_state(state, device)
                                        else:
                                            predictor.reset_state(state)
                                        if correction_restart_mode == "reseed-from-correction-frame":
                                            restart_start_frame_idx = min(int(v) for v in pause_correction_frames)
                                            restart_seed_points = build_reseed_points_from_tracks(restart_start_frame_idx)
                                            track_rows_data = [
                                                r for r in track_rows_data
                                                if int(r["frame"]) < int(restart_start_frame_idx)
                                            ]
                                            delete_outputs_from_frame(restart_start_frame_idx)
                                            frames_seen = int(restart_start_frame_idx)
                                            print(
                                                f"[CORRECT][{side_name}] Experimental reseed restart from "
                                                f"frame {restart_start_frame_idx}."
                                            )
                                            state = build_state_from_prompts(
                                                start_frame_idx=restart_start_frame_idx,
                                                seed_points_by_obj=restart_seed_points,
                                            )
                                        else:
                                            restart_start_frame_idx = 0
                                            restart_seed_points = None
                                            track_rows_data = []
                                            delete_outputs_from_frame(0)
                                            frames_seen = 0
                                            state = build_state_from_prompts()
                                        clean_restart_requested = True
                                    else:
                                        print(f"[INFO][{side_name}] Resumed at frame {f_idx}.")
                                    break
                                if pause_key == ord("v"):
                                    hidden_preview_obj_ids = prompt_id_visibility_in_preview(
                                        preview_win,
                                        paused_vis,
                                        visibility_total_count,
                                        hidden_preview_obj_ids,
                                    )
                                    continue
                                if pause_key == ord("c"):
                                    correction_frame_path = frames_dir / f"{int(pause_frame_idx):06d}.jpg"
                                    correction_frame = cv2.imread(str(correction_frame_path), cv2.IMREAD_COLOR)
                                    if correction_frame is None or correction_frame.size == 0:
                                        print(f"[CORRECT] Could not read correction frame: {correction_frame_path}")
                                        continue
                                    correction_vis, _, _ = render_preview_frame(
                                        correction_frame,
                                        paused_rows,
                                        scale,
                                        preview_max_width,
                                        f"{side_name} frame={pause_frame_idx} correction",
                                        reserved_top_px=112,
                                        visible_obj_ids=[
                                            obj_id for obj_id in active_obj_ids
                                            if obj_id not in hidden_preview_obj_ids
                                        ],
                                    )
                                    obj_id = prompt_obj_id_in_preview(
                                        preview_win,
                                        correction_vis,
                                        active_obj_ids,
                                        side_name,
                                        int(pause_frame_idx),
                                    )
                                    if obj_id is None:
                                        print("[CORRECT] Object ID selection canceled.")
                                        continue
                                    clicks = collect_correction_clicks(
                                        correction_frame,
                                        obj_id,
                                        int(pause_frame_idx),
                                        preview_max_width,
                                    )
                                    if not clicks:
                                        print("[CORRECT] No correction applied for this object.")
                                        continue
                                    correction = _make_correction_record(int(pause_frame_idx), obj_id, clicks, scale)
                                    corrections_payload.setdefault("corrections", {}).setdefault(side_key, []).append(correction)
                                    pause_corrections_added += 1
                                    pause_correction_frames.append(int(pause_frame_idx))
                                    print(
                                        f"[CORRECT][{side_name}] Queued correction for obj_id={obj_id} "
                                        f"at frame {pause_frame_idx}. Press p to apply/restart, or keep browsing."
                                    )
                                    continue
                                is_left_arrow = pause_key == 81 or raw_pause_key in (2424832, 65361)
                                is_right_arrow = pause_key == 83 or raw_pause_key in (2555904, 65363)
                                if is_left_arrow or is_right_arrow:
                                    delta = -1 if is_left_arrow else 1
                                    next_frame_idx = max(0, min(int(f_idx), int(pause_frame_idx) + delta))
                                    if next_frame_idx != pause_frame_idx:
                                        loaded_frame, loaded_rows = load_pause_frame(next_frame_idx)
                                        if loaded_frame is not None:
                                            pause_frame_idx = next_frame_idx
                                            paused_frame = loaded_frame
                                            paused_rows = loaded_rows
                                            pause_center = None
                                    continue
                                if pause_key in (ord("+"), ord("=")):
                                    pause_zoom = min(12.0, pause_zoom * INTERACTIVE_ZOOM_STEP)
                                    continue
                                if pause_key in (ord("-"), ord("_")):
                                    pause_zoom = max(1.0, pause_zoom / INTERACTIVE_ZOOM_STEP)
                                    continue
                                step = max(10.0, 60.0 / pause_zoom)
                                cx, cy = pause_center
                                if pause_key == ord("a"):
                                    pause_center = (cx - step, cy)
                                elif pause_key == ord("d"):
                                    pause_center = (cx + step, cy)
                                elif pause_key == ord("w"):
                                    pause_center = (cx, cy - step)
                                elif pause_key == ord("s"):
                                    pause_center = (cx, cy + step)
                            if stop_requested:
                                break
                            if clean_restart_requested:
                                break
                        if key == ord("c"):
                            prompt_vis, _, _ = render_preview_frame(
                                correction_frame_bgr,
                                correction_rows_by_obj,
                                scale,
                                preview_max_width,
                                f"{side_name} frame={correction_frame_idx} correction",
                                reserved_top_px=112,
                                visible_obj_ids=[
                                    obj_id for obj_id in active_obj_ids
                                    if obj_id not in hidden_preview_obj_ids
                                ],
                            )
                            target_frame_idx = prompt_frame_in_preview(
                                preview_win,
                                prompt_vis,
                                side_name,
                                int(correction_frame_idx),
                                total_extracted_frames,
                            )
                            if target_frame_idx is None:
                                print("[CORRECT] Frame selection canceled.")
                                continue
                            target_frame_path = frames_dir / f"{target_frame_idx:06d}.jpg"
                            target_frame = cv2.imread(str(target_frame_path), cv2.IMREAD_COLOR)
                            if target_frame is None or target_frame.size == 0:
                                print(f"[CORRECT] Could not read correction frame: {target_frame_path}")
                                continue
                            target_rows_by_obj = {
                                int(r["obj_id"]): r
                                for r in track_rows_data
                                if int(r["frame"]) == int(target_frame_idx)
                            }
                            if int(target_frame_idx) == int(f_idx):
                                target_rows_by_obj.update(frame_rows_by_obj)
                            prompt_vis, _, _ = render_preview_frame(
                                target_frame,
                                target_rows_by_obj,
                                scale,
                                preview_max_width,
                                f"{side_name} frame={target_frame_idx} correction",
                                reserved_top_px=112,
                                visible_obj_ids=[
                                    obj_id for obj_id in active_obj_ids
                                    if obj_id not in hidden_preview_obj_ids
                                ],
                            )
                            batch_added = 0
                            while True:
                                prompt_vis, _, _ = render_preview_frame(
                                    target_frame,
                                    target_rows_by_obj,
                                    scale,
                                    preview_max_width,
                                    f"{side_name} frame={target_frame_idx} correction  Esc done",
                                    reserved_top_px=112,
                                    visible_obj_ids=[
                                        obj_id for obj_id in active_obj_ids
                                        if obj_id not in hidden_preview_obj_ids
                                    ],
                                )
                                obj_id = prompt_obj_id_in_preview(
                                    preview_win,
                                    prompt_vis,
                                    active_obj_ids,
                                    side_name,
                                    int(target_frame_idx),
                                )
                                if obj_id is None:
                                    if batch_added:
                                        print(f"[CORRECT][{side_name}] Finished correction batch with {batch_added} object(s).")
                                        break
                                    print("[CORRECT] Object ID selection canceled.")
                                    break
                                clicks = collect_correction_clicks(target_frame, obj_id, int(target_frame_idx), preview_max_width)
                                if not clicks:
                                    print("[CORRECT] No correction applied for this object.")
                                    continue
                                correction = _make_correction_record(int(target_frame_idx), obj_id, clicks, scale)
                                corrections_payload.setdefault("corrections", {}).setdefault(side_key, []).append(correction)
                                batch_added += 1
                                print(
                                    f"[CORRECT][{side_name}] Queued correction for obj_id={obj_id} "
                                    f"at frame {target_frame_idx}."
                                )

                            if not batch_added:
                                continue

                            save_corrections(corrections_payload, corrections_path)
                            with open(out_dir / "prompts" / "corrections_applied.json", "w") as f:
                                json.dump(corrections_for_side(corrections_payload, side_key), f, indent=2)
                            print(f"[CORRECT][{side_name}] Saved {batch_added} correction(s). Restarting cleanly from frame 0...")
                            track_rows_data = []
                            frames_seen = 0
                            if video_writer is not None:
                                video_writer.release()
                                video_writer = None
                            if save_overlay:
                                print("[WARN] Correction restart invalidated partial overlay.mp4; disabling overlay for this run.")
                                if video_writer is not None:
                                    video_writer.release()
                                    video_writer = None
                                if out_mp4.exists():
                                    out_mp4.unlink()
                            if shared_state is None:
                                _release_sam2_state(state, device)
                            else:
                                predictor.reset_state(state)
                            state = build_state_from_prompts()
                            clean_restart_requested = True
                            break
                if stop_requested or not clean_restart_requested:
                    break
        finally:
            if progress_event_q is not None:
                progress_event_q.put({
                    "type": "done",
                    "side": str(side_name),
                    "total": int(total_extracted_frames),
                    "completed": int(frames_seen),
                    "description": f"{side_name} propagate",
                })
            if local_progress is not None:
                local_progress.stop()
            if local_preview:
                cv2.destroyWindow(preview_win)
            if video_writer is not None:
                video_writer.release()
            if save_tracks:
                track_rows_data.sort(key=lambda r: (int(r["frame"]), int(r["obj_id"])))
                with open(tracks_csv, "w", newline="") as tracks_f:
                    tracks_writer = csv.DictWriter(tracks_f, fieldnames=TRACK_FIELDNAMES)
                    tracks_writer.writeheader()
                    tracks_writer.writerows(track_rows_data)
                track_rows = len(track_rows_data)
                summary = {
                    "side": side_name,
                    "frames_processed": int(frames_seen),
                    "rows": int(track_rows),
                    "coordinate_space": "original",
                    "method": "sam2_dt_peak_direct",
                    "crop_applied": crop is not None,
                    "crop": list(crop) if crop is not None else None,
                    "processing_scale": float(scale),
                    "corrections_applied": len([
                        corr for corr in corrections_for_side(corrections_payload, side_key)
                        if int(corr.get("obj_id", -1)) in set(active_obj_ids)
                    ]),
                    "object_ids": active_obj_ids,
                }
                with open(out_dir / "tracks_2d_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
            if release_state:
                _release_sam2_state(state, device)
            elif shared_state is not None:
                predictor.reset_state(shared_state)

    if save_overlay and out_mp4.exists():
        print(f"[OK][{side_name}] Wrote {out_mp4}")
    if save_tracks:
        print(f"[OK][{side_name}] Wrote {tracks_csv}")
    return (out_mp4 if save_overlay and out_mp4.exists() else None), stop_requested


def _run_single_video_worker(side_name: str,
                             video_path: str,
                             out_dir_str: str,
                             points,
                             crop,
                             scale: float,
                             frame_extractor: str,
                             device: str,
                             offload_video_to_cpu: bool,
                             offload_state_to_cpu: bool,
                             async_loading_frames: bool,
                             save_masks: bool,
                             save_tracks: bool,
                             save_overlay: bool,
                             preview: bool,
                             preview_max_width: int,
                             corrections_payload=None,
                             corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
                             preview_event_q=None,
                             preview_cmd_q=None,
                             correction_restart_mode: str = "full",
                             object_ids=None,
                             frames_dir_override=None,
                             auto_pause_missing: bool = False,
                             progress_event_q=None):
    """Worker entrypoint for running one side on one explicit device."""
    os.environ["SAM2_DISABLE_FRAME_LOADING_PROGRESS"] = "1"
    try:
        device = str(device)
        if device.startswith("cuda"):
            if ":" in device:
                dev_idx = int(device.split(":", 1)[1])
            else:
                dev_idx = 0
            torch.cuda.set_device(dev_idx)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        from sam2.sam2_video_predictor import SAM2VideoPredictor

        predictor = SAM2VideoPredictor.from_pretrained(MODEL_ID).to(device)
        preview_controller = None
        if preview_event_q is not None and preview_cmd_q is not None:
            preview_controller = {
                "event_q": preview_event_q,
                "cmd_q": preview_cmd_q,
            }
        overlay, stopped = run_single_video_tracking(
            side_name=side_name,
            video_path=video_path,
            out_dir=Path(out_dir_str),
            points=points,
            crop=crop,
            scale=scale,
            frame_extractor=frame_extractor,
            predictor=predictor,
            device=device,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            save_masks=save_masks,
            save_tracks=save_tracks,
            save_overlay=save_overlay,
            preview=preview,
            preview_max_width=preview_max_width,
            corrections_payload=corrections_payload,
            corrections_path=corrections_path,
            object_ids=object_ids,
            frames_dir_override=frames_dir_override,
            preview_controller=preview_controller,
            correction_restart_mode=correction_restart_mode,
            auto_pause_missing=auto_pause_missing,
            progress_event_q=progress_event_q,
        )
        return {
            "side": side_name,
            "ok": True,
            "overlay": str(overlay) if overlay is not None else None,
            "stopped": bool(stopped),
            "error": None,
        }
    except Exception as e:
        return {
            "side": side_name,
            "ok": False,
            "overlay": None,
            "stopped": False,
            "error": str(e),
        }


def _run_single_video_process(result_q, *args, **kwargs):
    """Process target wrapper that returns the worker result through a queue."""
    result_q.put(_run_single_video_worker(*args, **kwargs))


def _dual_send_all(command_queues, command):
    for cmd_q in command_queues.values():
        try:
            cmd_q.put_nowait(dict(command))
        except Exception:
            pass


def _handle_dual_correction(event,
                            preview_win: str,
                            active_obj_ids,
                            track_rows_history,
                            hidden_obj_ids,
                            corrections_payload,
                            corrections_path: Path,
                            preview_max_width: int,
                            command_q):
    side_name = str(event["side"])
    side_key = side_name.lower()
    frame_idx = int(event["frame_idx"])
    scale = float(event["scale"])
    total_frames = int(event["total_frames"])
    frames_dir = Path(event["frames_dir"])
    frame_rows_by_obj = {
        int(k): v for k, v in event.get("rows_by_obj", {}).items()
    }
    prompt_vis = event["image"]

    target_frame_idx = prompt_frame_in_preview(
        preview_win,
        prompt_vis,
        side_name,
        frame_idx,
        total_frames,
    )
    if target_frame_idx is None:
        print("[CORRECT] Frame selection canceled.")
        command_q.put({"type": "continue"})
        return corrections_payload

    target_frame_path = frames_dir / f"{target_frame_idx:06d}.jpg"
    target_frame = cv2.imread(str(target_frame_path), cv2.IMREAD_COLOR)
    if target_frame is None or target_frame.size == 0:
        print(f"[CORRECT] Could not read correction frame: {target_frame_path}")
        command_q.put({"type": "continue"})
        return corrections_payload

    target_rows_by_obj = dict(track_rows_history.get(int(target_frame_idx), {}))
    if int(target_frame_idx) == frame_idx:
        target_rows_by_obj.update(frame_rows_by_obj)

    batch_added = 0
    while True:
        correction_vis, _, _ = render_preview_frame(
            target_frame,
            target_rows_by_obj,
            scale,
            preview_max_width,
            f"{side_name} frame={target_frame_idx} correction  Esc done",
            reserved_top_px=112,
            visible_obj_ids=[
                obj_id for obj_id in active_obj_ids
                if obj_id not in hidden_obj_ids
            ],
        )
        obj_id = prompt_obj_id_in_preview(
            preview_win,
            correction_vis,
            active_obj_ids,
            side_name,
            int(target_frame_idx),
        )
        if obj_id is None:
            if batch_added:
                print(f"[CORRECT][{side_name}] Finished correction batch with {batch_added} object(s).")
                break
            print("[CORRECT] Object ID selection canceled.")
            command_q.put({"type": "continue"})
            return corrections_payload

        clicks = collect_correction_clicks(target_frame, obj_id, int(target_frame_idx), preview_max_width)
        if not clicks:
            print("[CORRECT] No correction applied for this object.")
            continue

        correction = _make_correction_record(int(target_frame_idx), obj_id, clicks, scale)
        corrections_payload.setdefault("corrections", {}).setdefault(side_key, []).append(correction)
        batch_added += 1
        print(
            f"[CORRECT][{side_name}] Queued correction for obj_id={obj_id} "
            f"at frame {target_frame_idx}."
        )

    if batch_added:
        save_corrections(corrections_payload, corrections_path)
        command_q.put({
            "type": "restart",
            "corrections_payload": corrections_payload,
        })
    else:
        command_q.put({"type": "continue"})
    return corrections_payload


def run_dual_gpu_with_parent_preview(left_video,
                                     right_video,
                                     left_out,
                                     right_out,
                                     left_points,
                                     right_points,
                                     crop_left,
                                     crop_right,
                                     scale,
                                     frame_extractor,
                                     offload_video_to_cpu,
                                     offload_state_to_cpu,
                                     async_loading_frames,
                                     save_masks,
                                     save_tracks,
                                     save_overlay,
                                     preview_max_width,
                                     corrections_payload,
                                     corrections_path,
                                     correction_restart_mode: str = "full",
                                     left_object_ids=None,
                                     right_object_ids=None,
                                     left_frames_dir=None,
                                     right_frames_dir=None,
                                     auto_pause_missing: bool = False):
    """Run left/right workers on separate GPUs while the parent owns OpenCV UI."""
    mp_ctx = multiprocessing.get_context("spawn")
    event_q = mp_ctx.Queue(maxsize=2)
    progress_q = mp_ctx.Queue()
    result_q = mp_ctx.Queue()
    command_queues = {
        "LEFT": mp_ctx.Queue(maxsize=1),
        "RIGHT": mp_ctx.Queue(maxsize=1),
    }
    procs = []
    specs = [
        (
            "LEFT", left_video, str(left_out), left_points, crop_left, "cuda:0",
            command_queues["LEFT"], left_object_ids, left_frames_dir,
        ),
        (
            "RIGHT", right_video, str(right_out), right_points, crop_right, "cuda:1",
            command_queues["RIGHT"], right_object_ids, right_frames_dir,
        ),
    ]
    for side_name, video_path, out_dir, points, crop, device, cmd_q, object_ids, frames_dir in specs:
        proc = mp_ctx.Process(
            target=_run_single_video_process,
            args=(
                result_q,
                side_name,
                video_path,
                out_dir,
                points,
                crop,
                scale,
                frame_extractor,
                device,
                offload_video_to_cpu,
                offload_state_to_cpu,
                async_loading_frames,
                save_masks,
                save_tracks,
                save_overlay,
                True,
                preview_max_width,
                corrections_payload,
                corrections_path,
                event_q,
                cmd_q,
                correction_restart_mode,
                object_ids,
                str(frames_dir) if frames_dir is not None else None,
                auto_pause_missing,
                progress_q,
            ),
        )
        proc.start()
        procs.append(proc)

    preview_windows = {
        "LEFT": "SAM2 Preview [LEFT] (q or Esc to stop)",
        "RIGHT": "SAM2 Preview [RIGHT] (q or Esc to stop)",
    }
    for win in preview_windows.values():
        _open_cv_window(win, preview_max_width, 900)

    hidden_by_side = {"LEFT": set(), "RIGHT": set()}
    history_by_side = {"LEFT": {}, "RIGHT": {}}
    latest_event_by_side = {}
    results = {}
    stop_all = False
    pending_key = None
    pending_manual_pause_side = None
    progress = make_rich_progress()
    progress_tasks = {}

    def drain_progress_events():
        while True:
            try:
                progress_event = progress_q.get_nowait()
            except queue.Empty:
                break
            side = str(progress_event.get("side", "")).upper()
            if side not in ("LEFT", "RIGHT"):
                continue
            description = str(progress_event.get("description") or f"{side} propagate")
            total = int(progress_event.get("total", 0) or 0)
            completed = int(progress_event.get("completed", 0) or 0)
            if side not in progress_tasks:
                progress_tasks[side] = progress.add_task(description, total=total)
            progress.update(
                progress_tasks[side],
                description=description,
                total=total,
                completed=completed,
            )

    def prompt_manual_pause_side():
        print("[INFO][DUAL] Pause requested. Press l for LEFT, r for RIGHT, or Esc to cancel.")
        while True:
            for side, event in latest_event_by_side.items():
                img = event["image"].copy()
                cv2.rectangle(img, (0, 0), (min(img.shape[1] - 1, 760), 42), (0, 0, 0), -1)
                cv2.putText(
                    img,
                    "Pause which side?  l LEFT   r RIGHT   Esc cancel",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(preview_windows[side], img)
            choice_key = cv2.waitKeyEx(30)
            if choice_key == -1:
                continue
            choice = choice_key & 0xFF
            if choice == 27:
                print("[INFO][DUAL] Pause canceled.")
                return None
            if choice == ord("l"):
                return "LEFT"
            if choice == ord("r"):
                return "RIGHT"

    try:
        progress.start()
        while len(results) < 2:
            drain_progress_events()
            while True:
                try:
                    result = result_q.get_nowait()
                except queue.Empty:
                    break
                results[str(result.get("side", "")).upper()] = result
                if not result.get("ok", False):
                    stop_all = True
                    _dual_send_all(command_queues, {"type": "stop"})

            if len(results) >= 2:
                break

            try:
                event = event_q.get(timeout=0.1)
            except queue.Empty:
                drain_progress_events()
                if not any(proc.is_alive() for proc in procs) and result_q.empty():
                    break
                idle_key = cv2.waitKeyEx(30)
                if idle_key != -1:
                    pending_key = idle_key
                continue

            side_name = str(event.get("side", "")).upper()
            drain_progress_events()
            if event.get("type") != "frame" or side_name not in command_queues:
                continue

            frame_idx = int(event["frame_idx"])
            rows_by_obj = {
                int(k): v for k, v in event.get("rows_by_obj", {}).items()
            }
            active_obj_ids = [int(v) for v in event.get("active_obj_ids", [])]
            history_by_side[side_name][frame_idx] = rows_by_obj
            latest_event_by_side[side_name] = event

            preview_win = preview_windows[side_name]
            cv2.imshow(preview_win, event["image"])
            raw_key = pending_key
            pending_key = None
            if raw_key is None:
                raw_key = cv2.waitKeyEx(30)
            key = -1 if raw_key == -1 else raw_key & 0xFF

            if key in (27, ord("q")):
                print(f"[INFO][{side_name}] Stop requested from dual preview.")
                stop_all = True
                _dual_send_all(command_queues, {"type": "stop"})
                continue

            if key == ord("v"):
                hidden_by_side[side_name] = prompt_id_visibility_in_preview(
                    preview_win,
                    event["image"],
                    int(event["visibility_total_count"]),
                    hidden_by_side[side_name],
                )
                command_queues[side_name].put({
                    "type": "rerender",
                    "hidden_obj_ids": sorted(hidden_by_side[side_name]),
                })
                continue

            valid_markers, expected_markers, missing_markers = _valid_marker_count(
                rows_by_obj,
                active_obj_ids,
            )
            auto_pause = bool(auto_pause_missing) and expected_markers > 0 and valid_markers < expected_markers
            if auto_pause and key != ord("p"):
                print(
                    f"[INFO][{side_name}] Auto-pausing at frame {frame_idx}: "
                    f"valid markers {valid_markers}/{expected_markers}; "
                    f"missing IDs {missing_markers[:20]}"
                    f"{'...' if len(missing_markers) > 20 else ''}."
                )

            manual_pause = False
            if pending_manual_pause_side == side_name:
                manual_pause = True
                pending_manual_pause_side = None
                print(f"[INFO][DUAL] Pausing selected side {side_name} at frame {frame_idx}.")
            elif key == ord("p") and not auto_pause:
                selected_side = prompt_manual_pause_side()
                if selected_side is None:
                    command_queues[side_name].put({"type": "continue"})
                    continue
                if selected_side != side_name:
                    pending_manual_pause_side = selected_side
                    command_queues[side_name].put({"type": "continue"})
                    print(f"[INFO][DUAL] Waiting for next {selected_side} frame to pause.")
                    continue
                manual_pause = True

            if manual_pause or auto_pause:
                pause_zoom = 1.0
                pause_frame_idx = frame_idx
                pause_center = None
                pause_corrections_added = 0
                pause_correction_frames = []
                pause_reason = _pause_reason_text(
                    auto_pause,
                    valid_markers,
                    expected_markers,
                    missing_markers,
                )
                print(
                    f"[INFO][{side_name}] Paused at frame {frame_idx}. "
                    "Left/Right frame, p resume, c correct, +/- zoom, WASD pan."
                )

                def load_pause_frame(side, idx):
                    side_event = latest_event_by_side.get(side)
                    if side_event is None:
                        return None, {}
                    pause_path = Path(side_event["frames_dir"]) / f"{int(idx):06d}.jpg"
                    pause_frame = cv2.imread(str(pause_path), cv2.IMREAD_COLOR)
                    if pause_frame is None or pause_frame.size == 0:
                        return None, {}
                    return pause_frame, dict(history_by_side[side].get(int(idx), {}))

                def render_paused_side(side, idx, zoom, center_xy, title):
                    side_event = latest_event_by_side.get(side)
                    if side_event is None:
                        return center_xy, None, {}
                    side_frame, side_rows = load_pause_frame(side, idx)
                    if side_frame is None:
                        return center_xy, None, {}
                    side_active_obj_ids = [int(v) for v in side_event.get("active_obj_ids", [])]
                    side_vis, next_center, _ = render_preview_frame(
                        side_frame,
                        side_rows,
                        float(side_event["scale"]),
                        preview_max_width,
                        title,
                        zoom=zoom,
                        center_xy=center_xy,
                        visible_obj_ids=[
                            obj_id for obj_id in side_active_obj_ids
                            if obj_id not in hidden_by_side[side]
                        ],
                    )
                    cv2.imshow(preview_windows[side], side_vis)
                    return next_center, side_vis, side_rows

                paused_frame, paused_rows = load_pause_frame(side_name, pause_frame_idx)
                while True:
                    if paused_frame is None:
                        pause_frame_idx = frame_idx
                        paused_frame, paused_rows = load_pause_frame(side_name, pause_frame_idx)
                    pause_center, paused_vis, paused_rows = render_paused_side(
                        side_name,
                        pause_frame_idx,
                        pause_zoom,
                        pause_center,
                        f"PAUSED {side_name} frame={pause_frame_idx}{pause_reason}  Left/Right frame  p resume  v IDs  c correct  +/- zoom  WASD pan",
                    )
                    peer_side = "RIGHT" if side_name == "LEFT" else "LEFT"
                    render_paused_side(
                        peer_side,
                        pause_frame_idx,
                        pause_zoom,
                        pause_center,
                        f"PAUSED WITH {side_name} frame={pause_frame_idx}  correction side={side_name}",
                    )
                    raw_pause_key = cv2.waitKeyEx(30)
                    if raw_pause_key == -1:
                        if cv2.getWindowProperty(preview_win, cv2.WND_PROP_VISIBLE) < 1:
                            stop_all = True
                            _dual_send_all(command_queues, {"type": "stop"})
                            break
                        continue
                    pause_key = raw_pause_key & 0xFF
                    if pause_key in (27, ord("q")):
                        stop_all = True
                        _dual_send_all(command_queues, {"type": "stop"})
                        break
                    if pause_key == ord("p"):
                        if pause_corrections_added:
                            save_corrections(corrections_payload, corrections_path)
                            print(
                                f"[CORRECT][{side_name}] Saved {pause_corrections_added} correction(s). "
                                "Restarting cleanly..."
                            )
                            restart_cmd = {
                                "type": "restart",
                                "corrections_payload": corrections_payload,
                            }
                            if correction_restart_mode == "reseed-from-correction-frame":
                                restart_frame_idx = min(int(v) for v in pause_correction_frames)
                                seed_rows = dict(history_by_side[side_name].get(int(restart_frame_idx), {}))
                                missing = []
                                seed_points = {}
                                for obj_id in active_obj_ids:
                                    row = seed_rows.get(int(obj_id))
                                    if row is None or int(row.get("valid", 0)) != 1:
                                        missing.append(int(obj_id))
                                        continue
                                    seed_points[int(obj_id)] = [
                                        float(row["u_local"]) * float(event["scale"]),
                                        float(row["v_local"]) * float(event["scale"]),
                                    ]
                                if missing:
                                    print(
                                        f"[CORRECT][{side_name}] Cannot reseed from frame {restart_frame_idx}; "
                                        f"missing valid object IDs {missing[:20]}"
                                        f"{'...' if len(missing) > 20 else ''}. Falling back to full restart."
                                    )
                                else:
                                    print(
                                        f"[CORRECT][{side_name}] Experimental reseed restart from "
                                        f"frame {restart_frame_idx}."
                                    )
                                    restart_cmd["start_frame_idx"] = int(restart_frame_idx)
                                    restart_cmd["seed_points_by_obj"] = seed_points
                            command_queues[side_name].put(restart_cmd)
                        else:
                            command_queues[side_name].put({"type": "continue"})
                        break
                    if pause_key == ord("v"):
                        hidden_by_side[side_name] = prompt_id_visibility_in_preview(
                            preview_win,
                            paused_vis,
                            int(event["visibility_total_count"]),
                            hidden_by_side[side_name],
                        )
                        continue
                    if pause_key == ord("c"):
                        side_key = side_name.lower()
                        correction_frame_path = Path(event["frames_dir"]) / f"{int(pause_frame_idx):06d}.jpg"
                        correction_frame = cv2.imread(str(correction_frame_path), cv2.IMREAD_COLOR)
                        if correction_frame is None or correction_frame.size == 0:
                            print(f"[CORRECT] Could not read correction frame: {correction_frame_path}")
                            continue
                        correction_vis, _, _ = render_preview_frame(
                            correction_frame,
                            paused_rows,
                            float(event["scale"]),
                            preview_max_width,
                            f"{side_name} frame={pause_frame_idx} correction",
                            reserved_top_px=112,
                            visible_obj_ids=[
                                obj_id for obj_id in active_obj_ids
                                if obj_id not in hidden_by_side[side_name]
                            ],
                        )
                        obj_id = prompt_obj_id_in_preview(
                            preview_win,
                            correction_vis,
                            active_obj_ids,
                            side_name,
                            int(pause_frame_idx),
                        )
                        if obj_id is None:
                            print("[CORRECT] Object ID selection canceled.")
                            continue
                        clicks = collect_correction_clicks(
                            correction_frame,
                            obj_id,
                            int(pause_frame_idx),
                            preview_max_width,
                        )
                        if not clicks:
                            print("[CORRECT] No correction applied for this object.")
                            continue
                        correction = _make_correction_record(int(pause_frame_idx), obj_id, clicks, float(event["scale"]))
                        corrections_payload.setdefault("corrections", {}).setdefault(side_key, []).append(correction)
                        pause_corrections_added += 1
                        pause_correction_frames.append(int(pause_frame_idx))
                        print(
                            f"[CORRECT][{side_name}] Queued correction for obj_id={obj_id} "
                            f"at frame {pause_frame_idx}. Press p to apply/restart, or keep browsing."
                        )
                        continue
                    is_left_arrow = pause_key == 81 or raw_pause_key in (2424832, 65361)
                    is_right_arrow = pause_key == 83 or raw_pause_key in (2555904, 65363)
                    if is_left_arrow or is_right_arrow:
                        delta = -1 if is_left_arrow else 1
                        next_frame_idx = max(0, min(frame_idx, int(pause_frame_idx) + delta))
                        if next_frame_idx != pause_frame_idx:
                            loaded_frame, loaded_rows = load_pause_frame(side_name, next_frame_idx)
                            if loaded_frame is not None:
                                pause_frame_idx = next_frame_idx
                                paused_frame = loaded_frame
                                paused_rows = loaded_rows
                                pause_center = None
                        continue
                    if pause_key in (ord("+"), ord("=")):
                        pause_zoom = min(12.0, pause_zoom * INTERACTIVE_ZOOM_STEP)
                        continue
                    if pause_key in (ord("-"), ord("_")):
                        pause_zoom = max(1.0, pause_zoom / INTERACTIVE_ZOOM_STEP)
                        continue
                    if pause_center is None:
                        pause_center = (paused_frame.shape[1] / 2.0, paused_frame.shape[0] / 2.0)
                    step = max(10.0, 60.0 / pause_zoom)
                    cx, cy = pause_center
                    if pause_key == ord("a"):
                        pause_center = (cx - step, cy)
                    elif pause_key == ord("d"):
                        pause_center = (cx + step, cy)
                    elif pause_key == ord("w"):
                        pause_center = (cx, cy - step)
                    elif pause_key == ord("s"):
                        pause_center = (cx, cy + step)
                continue

            if key == ord("c"):
                corrections_payload = _handle_dual_correction(
                    event,
                    preview_win,
                    [int(v) for v in event.get("active_obj_ids", [])],
                    history_by_side[side_name],
                    hidden_by_side[side_name],
                    corrections_payload,
                    corrections_path,
                    preview_max_width,
                    command_queues[side_name],
                )
                continue

            if not stop_all:
                command_queues[side_name].put({"type": "continue"})

        for proc in procs:
            proc.join()
        drain_progress_events()
    finally:
        progress.stop()
        for win in preview_windows.values():
            try:
                cv2.destroyWindow(win)
            except Exception:
                pass
        if stop_all:
            for proc in procs:
                if proc.is_alive():
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.terminate()

    left_result = results.get("LEFT")
    right_result = results.get("RIGHT")
    if left_result is None or right_result is None:
        missing = [side for side in ("LEFT", "RIGHT") if side not in results]
        raise RuntimeError(f"Dual-GPU preview ended without result(s): {missing}")
    return left_result, right_result


def main():
    args = parse_args()

    selected_setup_modes = sum(
        bool(v)
        for v in (
            args.setup,
            args.reuse_setup,
            args.modify_setup,
            args.semi_auto_setup,
            args.semi_auto_setup_sections,
        )
    )
    if selected_setup_modes > 1:
        raise RuntimeError(
            "Use only one of --setup, --semi-auto-setup, --semi-auto-setup-sections, "
            "--reuse-setup, or --modify-setup."
        )

    if (
        args.setup
        or args.semi_auto_setup
        or args.semi_auto_setup_sections
        or args.modify_setup
    ) and Path(args.out) == DEFAULT_OUT_DIR:
        args.out = str(DEFAULT_LOCAL_SETUP_DIR)

    left_video = resolve_video_path(args.left_input)
    right_video = resolve_video_path(args.right_input)
    start_frame = int(args.start_frame)
    if start_frame < 0:
        raise RuntimeError("--start-frame must be >= 0")
    source_counts = []
    for video_path in (left_video, right_video):
        cap = cv2.VideoCapture(video_path)
        source_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1)
        cap.release()
    end_frame = int(args.end_frame) if args.end_frame is not None else min(source_counts)
    if end_frame <= start_frame or any(count > 0 and end_frame > count for count in source_counts):
        raise RuntimeError(
            f"Invalid virtual clip range [{start_frame}, {end_frame}) for source counts {source_counts}"
        )
    out_root = Path(args.out)
    corrections_path = Path(args.corrections_json)

    offload_video_to_cpu = args.offload_video_to_cpu
    offload_state_to_cpu = args.offload_state_to_cpu
    async_loading_frames = args.async_loading_frames
    scale = float(args.scale)
    if scale <= 0:
        raise RuntimeError("--scale must be > 0")
    if scale == 1.0:
        print("")
        print("!" * 88)
        print("WARNING: RUNNING SAM2 AT FULL SCALE (--scale 1.0)")
        print("This is slow and memory-heavy. If this was accidental, stop now and rerun with")
        print("something like: --scale 0.5")
        print("!" * 88)
        print("")
    frame_extractor = args.frame_extractor
    save_masks = args.save_masks
    save_tracks = args.save_tracks
    save_overlay = args.save_overlay
    preview = args.preview
    preview_max_width = args.preview_max_width
    corrections_payload = _empty_corrections_payload()
    semi_auto_debug_payload = None
    if args.reuse_setup:
        corrections_payload = load_corrections(corrections_path)
        total_saved_corrections = sum(
            len(v) for v in corrections_payload.get("corrections", {}).values()
        )
        if total_saved_corrections:
            print(f"[INFO] Loaded saved SAM2 corrections: {corrections_path} ({total_saved_corrections})")

    left_crop_forced_none = is_crop_none_arg(args.left_crop) or not bool(args.select_crop)
    right_crop_forced_none = is_crop_none_arg(args.right_crop) or not bool(args.select_crop)
    crop_left = parse_crop_arg(args.left_crop) if bool(args.select_crop) else None
    crop_right = parse_crop_arg(args.right_crop) if bool(args.select_crop) else None
    loaded_crop_left = None
    loaded_crop_right = None

    saved_setup_json = DEFAULT_LOCAL_SETUP_DIR / "prompts" / "points_left_right.json"
    if args.reuse_setup or args.modify_setup:
        if args.points_json is not None or args.left_points is not None or args.right_points is not None:
            raise RuntimeError("Use saved setup or explicit point files, not both.")
        if not saved_setup_json.exists():
            raise FileNotFoundError(
                f"Saved setup was not found: {saved_setup_json}"
            )
        args.points_json = str(saved_setup_json)
        mode_name = "--modify-setup" if args.modify_setup else "--reuse-setup"
        print(f"[OK] {mode_name} loaded saved crop/point setup: {saved_setup_json}")

    have_shared_points = args.points_json is not None
    have_side_points = args.left_points is not None or args.right_points is not None
    if have_side_points and not (args.left_points and args.right_points):
        raise RuntimeError("--left-points and --right-points must be provided together.")
    if have_shared_points and have_side_points:
        raise RuntimeError("Use either --points-json or --left-points/--right-points, not both.")

    clicked_fresh_setup = False
    if have_shared_points:
        left_points = load_points_file(args.points_json, side="left", prefer_local=False)
        right_points = load_points_file(args.points_json, side="right", prefer_local=False)
        loaded_crop_left, loaded_crop_right = load_crops_from_points_json(args.points_json)
        if crop_left is None and not left_crop_forced_none:
            crop_left = loaded_crop_left
        if crop_right is None and not right_crop_forced_none:
            crop_right = loaded_crop_right
        if args.modify_setup and bool(args.select_crop):
            if loaded_crop_left is not None:
                left_points = offset_points(left_points, loaded_crop_left)
            if loaded_crop_right is not None:
                right_points = offset_points(right_points, loaded_crop_right)

            print("[INFO] Select and review LEFT crop for modified setup...")
            crop_left = select_crop_with_review(left_video, "LEFT")
            print("[INFO] Select and review RIGHT crop for modified setup...")
            crop_right = select_crop_with_review(right_video, "RIGHT")

            if crop_left is not None:
                left_points = translate_points(left_points, -float(crop_left[0]), -float(crop_left[1]))
            if crop_right is not None:
                right_points = translate_points(right_points, -float(crop_right[0]), -float(crop_right[1]))
            loaded_crop_left = None
            loaded_crop_right = None
        if left_crop_forced_none and loaded_crop_left is not None:
            left_points = offset_points(left_points, loaded_crop_left)
            print(f"[INFO] LEFT crop disabled; shifted saved crop-local points by x={loaded_crop_left[0]}, y={loaded_crop_left[1]}")
        if right_crop_forced_none and loaded_crop_right is not None:
            right_points = offset_points(right_points, loaded_crop_right)
            print(f"[INFO] RIGHT crop disabled; shifted saved crop-local points by x={loaded_crop_right[0]}, y={loaded_crop_right[1]}")
        print(f"[OK] Loaded headless points from {args.points_json}")
        if args.modify_setup:
            left_first = load_first_frame_from_video(left_video, crop=crop_left, frame_idx=start_frame)
            right_first = load_first_frame_from_video(right_video, crop=crop_right, frame_idx=start_frame)
            print(
                f"[INFO] Modify setup: loaded {len(left_points)} LEFT and {len(right_points)} RIGHT existing markers."
            )
            print("[INFO] Modify setup: LMB select/place moves points; press n before LMB to add a new point. Press q when done.")
            left_points, right_points = click_points_dual(
                left_first,
                right_first,
                initial_left_points=left_points,
                initial_right_points=right_points,
                move_existing_points=True,
            )
            if len(left_points) != len(right_points):
                raise RuntimeError(f"Point count mismatch: LEFT={len(left_points)}, RIGHT={len(right_points)}")
            if len(left_points) == 0:
                raise RuntimeError("No points selected.")
            print(f"[OK] Modified setup now has {len(left_points)} markers on each side.")
    elif have_side_points:
        left_points = load_points_file(args.left_points, side="left", prefer_local=True)
        right_points = load_points_file(args.right_points, side="right", prefer_local=True)
        if crop_left is None:
            crop_left = load_crop_from_auto_points(args.left_points)
        if crop_right is None:
            crop_right = load_crop_from_auto_points(args.right_points)
        print(f"[OK] Loaded headless LEFT points from {args.left_points}")
        print(f"[OK] Loaded headless RIGHT points from {args.right_points}")
    else:
        clicked_fresh_setup = True
        if bool(args.select_crop) and not (left_crop_forced_none and right_crop_forced_none):
            if left_crop_forced_none:
                crop_left = None
                print("[INFO] LEFT crop disabled by --left-crop none")
            else:
                print("[INFO] Select and review LEFT crop...")
                crop_left = select_crop_with_review(left_video, "LEFT")

            if right_crop_forced_none:
                crop_right = None
                print("[INFO] RIGHT crop disabled by --right-crop none")
            else:
                print("[INFO] Select and review RIGHT crop...")
                crop_right = select_crop_with_review(right_video, "RIGHT")
        else:
            if left_crop_forced_none:
                crop_left = None
                print("[INFO] LEFT crop disabled; using full frame")
            if right_crop_forced_none:
                crop_right = None
                print("[INFO] RIGHT crop disabled; using full frame")

        left_first = load_first_frame_from_video(left_video, crop=crop_left, frame_idx=start_frame)
        right_first = load_first_frame_from_video(right_video, crop=crop_right, frame_idx=start_frame)

        if args.semi_auto_setup_sections:
            left_points, right_points, left_meta, right_meta = _run_sectioned_grid_setup(
                left_first,
                right_first,
                default_cols=args.grid_cols,
                default_rows=args.grid_rows,
                section_layout=_parse_section_layout(args.section_layout),
            )
            semi_auto_debug_payload = (left_first, right_first, left_meta, right_meta)
        elif args.semi_auto_setup:
            cols = args.grid_cols
            rows = args.grid_rows
            if cols is None:
                cols = _prompt_int("Semi-auto grid columns", minimum=2)
            if rows is None:
                rows = _prompt_int("Semi-auto grid rows", minimum=2)
            if cols < 2 or rows < 2:
                raise RuntimeError("--grid-cols and --grid-rows must both be at least 2.")
            expected_points = cols * rows
            try:
                left_corners, left_corner_meta = _estimate_grid_corners_from_detection(
                    "left", left_first, cols, rows
                )
                right_corners, right_corner_meta = _estimate_grid_corners_from_detection(
                    "right", right_first, cols, rows
                )
            except Exception as exc:
                print(f"[WARN] Automatic corner estimation failed: {exc}")
                print(
                    "[INFO] Click exactly 4 anchors in each window, in this order: "
                    "top-left, top-right, bottom-left, bottom-right. Press q when done."
                )
                left_corners, right_corners = click_points_dual(left_first, right_first)
            else:
                print(
                    "[INFO] Review the four automatic corner guesses. Move any incorrect guess, "
                    "then press q to continue."
                )
                left_corners, right_corners = click_points_dual(
                    left_first,
                    right_first,
                    initial_left_points=left_corners,
                    initial_right_points=right_corners,
                    anchor_review=True,
                    initial_left_meta=left_corner_meta,
                    initial_right_meta=right_corner_meta,
                )
            if len(left_corners) != 4 or len(right_corners) != 4:
                raise RuntimeError(
                    f"Semi-auto setup needs exactly 4 anchors per side; got LEFT={len(left_corners)}, RIGHT={len(right_corners)}."
                )
            left_corner_xy = [_coerce_point_pair(p, prefer_local=False) for p in left_corners]
            right_corner_xy = [_coerce_point_pair(p, prefer_local=False) for p in right_corners]

            left_points, left_meta = _generate_grid_prompts_from_detection("left", left_first, left_corner_xy, cols, rows)
            right_points, right_meta = _generate_grid_prompts_from_detection("right", right_first, right_corner_xy, cols, rows)

            print(
                f"[INFO] Review generated {expected_points} prompts per side. "
                "Use two-click move for corrections, then press q to accept."
            )
            left_points, right_points = click_points_dual(
                left_first,
                right_first,
                initial_left_points=left_points,
                initial_right_points=right_points,
                fixed_count_review=True,
                initial_left_meta=left_meta.get("points", []),
                initial_right_meta=right_meta.get("points", []),
            )
            if len(left_points) != expected_points or len(right_points) != expected_points:
                raise RuntimeError(
                    f"Semi-auto setup count changed unexpectedly: LEFT={len(left_points)}, RIGHT={len(right_points)}, expected={expected_points}."
                )
            semi_auto_debug_payload = (left_first, right_first, left_meta, right_meta)
        else:
            print("[INFO] Click points in both windows. Press q when done.")
            left_points, right_points = click_points_dual(left_first, right_first)

        if len(left_points) != len(right_points):
            raise RuntimeError(f"Point count mismatch: LEFT={len(left_points)}, RIGHT={len(right_points)}")
        if len(left_points) == 0:
            raise RuntimeError("No points selected.")

        print(f"[OK] Selected {len(left_points)} markers on each side.")

    if clicked_fresh_setup:
        corrections_payload = _empty_corrections_payload()
        if corrections_path.exists():
            corrections_path.unlink()
            print(f"[OK] Cleared old corrections for freshly clicked setup: {corrections_path}")

    if left_crop_forced_none or right_crop_forced_none:
        if Path(args.corrections_json) == DEFAULT_CORRECTIONS_PATH:
            corrections_path = NO_CROP_CORRECTIONS_PATH
        else:
            corrections_path = corrections_path.with_name(
                f"{corrections_path.stem}_no_crop{corrections_path.suffix or '.json'}"
            )
        if corrections_path.exists():
            corrections_payload = load_corrections(corrections_path)
            print(f"[INFO] Loaded no-crop corrections: {corrections_path}")
        else:
            corrections_payload = offset_corrections_payload(
                corrections_payload,
                {
                    "left": loaded_crop_left if left_crop_forced_none else None,
                    "right": loaded_crop_right if right_crop_forced_none else None,
                },
            )
            print("[INFO] Converted saved cropped corrections into full-frame no-crop coordinates for this run.")
        print(f"[INFO] Crop disabled for at least one side; no-crop corrections will save to {corrections_path}")

    if len(left_points) == 0:
        raise RuntimeError("No LEFT points selected.")
    if len(right_points) == 0:
        raise RuntimeError("No RIGHT points selected.")
    if len(left_points) != len(right_points):
        print(
            f"[WARN] Point count mismatch: LEFT={len(left_points)}, RIGHT={len(right_points)}. "
            "This is OK for independent SAM2 tracking, but not for triangulation by obj_id."
        )
    else:
        print(f"[OK] Using {len(left_points)} matched markers on each side.")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    write_setup_package(
        out_root=out_root,
        left_video=left_video,
        right_video=right_video,
        left_points=left_points,
        right_points=right_points,
        crop_left=crop_left,
        crop_right=crop_right,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if semi_auto_debug_payload is not None:
        left_first, right_first, left_meta, right_meta = semi_auto_debug_payload
        _save_semi_auto_debug(out_root, left_first, right_first, left_points, right_points, left_meta, right_meta)
        print(f"[OK] Saved semi-auto setup debug previews: {out_root / 'prompts' / 'semi_auto_debug'}")

    if args.setup or args.semi_auto_setup or args.semi_auto_setup_sections or args.modify_setup:
        if corrections_path.exists():
            if args.setup or args.semi_auto_setup or args.semi_auto_setup_sections:
                corrections_path.unlink()
                print(f"[OK] Cleared old corrections for fresh setup: {corrections_path}")
            else:
                print(f"[INFO] Kept existing corrections for modified setup: {corrections_path}")
        if args.modify_setup:
            done_flag = "--modify-setup"
        elif args.semi_auto_setup_sections:
            done_flag = "--semi-auto-setup-sections"
        elif args.semi_auto_setup:
            done_flag = "--semi-auto-setup"
        else:
            done_flag = "--setup"
        print(f"[OK] {done_flag} complete. Setup package is ready.")
        return

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    gpu_mode = str(args.gpu_mode).lower()

    device = "cuda:0" if gpu_count > 0 else "cpu"
    run_dual = False

    if gpu_mode == "auto":
        run_dual = gpu_count >= 2
        if not run_dual:
            device = "cuda:0" if gpu_count > 0 else "cpu"
    elif gpu_mode == "dual":
        if gpu_count < 2:
            raise RuntimeError("--gpu-mode=dual requires at least 2 CUDA GPUs.")
        run_dual = True
    elif gpu_mode == "single":
        if gpu_count <= 0:
            device = "cpu"
        else:
            if args.single_gpu_index is None:
                idx = choose_single_gpu_index(gpu_names)
            else:
                idx = int(args.single_gpu_index)
            if idx < 0 or idx >= gpu_count:
                raise RuntimeError(f"--single-gpu-index={idx} is out of range [0, {gpu_count - 1}].")
            device = f"cuda:{idx}"
        run_dual = False
    elif gpu_mode == "4090-only":
        if gpu_count <= 0:
            raise RuntimeError("--gpu-mode=4090-only requested, but no CUDA GPUs were detected.")
        idx_4090 = next((i for i, name in enumerate(gpu_names) if "4090" in str(name).upper()), None)
        if idx_4090 is None:
            raise RuntimeError(f"--gpu-mode=4090-only requested, but no 4090 was detected. GPUs: {gpu_names}")
        device = f"cuda:{idx_4090}"
        run_dual = False

    print("[INFO] gpu_mode:", gpu_mode)
    print("[INFO] device:", device)
    print("[INFO] detected CUDA GPUs:", gpu_count)
    if gpu_names:
        for i, name in enumerate(gpu_names):
            print(f"[INFO] GPU[{i}]: {name}")
    print(
        f"[INFO] init_state options: offload_video_to_cpu={offload_video_to_cpu}, "
        f"offload_state_to_cpu={offload_state_to_cpu}, async_loading_frames={async_loading_frames}"
    )
    print(f"[INFO] processing scale: {scale:g}")
    print(f"[INFO] frame extractor: {frame_extractor}")
    print(f"[INFO] correction restart mode: {args.correction_restart_mode}")
    print(
        f"[INFO] outputs: save_tracks={save_tracks}, save_masks={save_masks}, "
        f"save_overlay={save_overlay}, preview={preview}"
    )

    left_out = out_root / "left"
    right_out = out_root / "right"

    if run_dual:
        print("[INFO] Dual-GPU mode: LEFT->cuda:0, RIGHT->cuda:1")
        if preview:
            print("[INFO] Dual-GPU preview: parent process owns OpenCV windows; workers own GPUs.")
            left_result, right_result = run_dual_gpu_with_parent_preview(
                left_video=left_video,
                right_video=right_video,
                left_out=left_out,
                right_out=right_out,
                left_points=left_points,
                right_points=right_points,
                crop_left=crop_left,
                crop_right=crop_right,
                scale=scale,
                frame_extractor=frame_extractor,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                async_loading_frames=async_loading_frames,
                save_masks=save_masks,
                save_tracks=save_tracks,
                save_overlay=save_overlay,
                preview_max_width=preview_max_width,
                corrections_payload=corrections_payload,
                corrections_path=corrections_path,
                correction_restart_mode=args.correction_restart_mode,
                auto_pause_missing=args.auto_pause_missing,
            )
        else:
            mp_ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=mp_ctx) as ex:
                left_future = ex.submit(
                    _run_single_video_worker,
                    "LEFT",
                    left_video,
                    str(left_out),
                    left_points,
                    crop_left,
                    scale,
                    frame_extractor,
                    "cuda:0",
                    offload_video_to_cpu,
                    offload_state_to_cpu,
                    async_loading_frames,
                    save_masks,
                    save_tracks,
                    save_overlay,
                    False,
                    preview_max_width,
                    corrections_payload,
                    corrections_path,
                    None,
                    None,
                    args.correction_restart_mode,
                    None,
                    None,
                    args.auto_pause_missing,
                )
                right_future = ex.submit(
                    _run_single_video_worker,
                    "RIGHT",
                    right_video,
                    str(right_out),
                    right_points,
                    crop_right,
                    scale,
                    frame_extractor,
                    "cuda:1",
                    offload_video_to_cpu,
                    offload_state_to_cpu,
                    async_loading_frames,
                    save_masks,
                    save_tracks,
                    save_overlay,
                    False,
                    preview_max_width,
                    corrections_payload,
                    corrections_path,
                    None,
                    None,
                    args.correction_restart_mode,
                    None,
                    None,
                    args.auto_pause_missing,
                )
                left_result = left_future.result()
                right_result = right_future.result()

        for result in (left_result, right_result):
            if not result["ok"]:
                raise RuntimeError(f"{result['side']} failed in dual-GPU worker: {result['error']}")

        left_overlay = Path(left_result["overlay"]) if left_result["overlay"] else None
        right_overlay = Path(right_result["overlay"]) if right_result["overlay"] else None
        left_stopped = bool(left_result["stopped"])
        right_stopped = bool(right_result["stopped"])
    else:
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        if device.startswith("cuda"):
            dev_idx = int(device.split(":", 1)[1]) if ":" in device else 0
            torch.cuda.set_device(dev_idx)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        predictor = SAM2VideoPredictor.from_pretrained(MODEL_ID).to(device)
        print(f"[INFO] Single-device mode: both sides on {device}")

        left_overlay, left_stopped = run_single_video_tracking(
            side_name="LEFT",
            video_path=left_video,
            out_dir=left_out,
            points=left_points,
            crop=crop_left,
            scale=scale,
            frame_extractor=frame_extractor,
            predictor=predictor,
            device=device,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            save_masks=save_masks,
            save_tracks=save_tracks,
            save_overlay=save_overlay,
            preview=preview,
            preview_max_width=preview_max_width,
            corrections_payload=corrections_payload,
            corrections_path=corrections_path,
            correction_restart_mode=args.correction_restart_mode,
            auto_pause_missing=args.auto_pause_missing,
        )

        right_overlay, right_stopped = run_single_video_tracking(
            side_name="RIGHT",
            video_path=right_video,
            out_dir=right_out,
            points=right_points,
            crop=crop_right,
            scale=scale,
            frame_extractor=frame_extractor,
            predictor=predictor,
            device=device,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            save_masks=save_masks,
            save_tracks=save_tracks,
            save_overlay=save_overlay,
            preview=preview,
            preview_max_width=preview_max_width,
            corrections_payload=corrections_payload,
            corrections_path=corrections_path,
            correction_restart_mode=args.correction_restart_mode,
            auto_pause_missing=args.auto_pause_missing,
        )

    if left_stopped:
        print("[INFO] LEFT processing stopped early by user.")
    if right_stopped:
        print("[INFO] RIGHT processing stopped early by user.")

    for overlay in (left_overlay, right_overlay):
        if overlay is None:
            continue
        try:
            os.startfile(str(overlay))
        except Exception as e:
            print(f"[WARN] Could not open {overlay}: {e}")

if __name__ == "__main__":
    main()
