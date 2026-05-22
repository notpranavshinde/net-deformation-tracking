"""
Automatically detect candidate SAM2 prompt points for painted net nodes.

Run this from sam2\\sam2. The default command processes LEFT and RIGHT together,
uses any existing SAM2 crop metadata in out\\left/right\\prompts\\crop_roi.json,
and writes outputs under work\\auto_prompts. If you use --select-crop, it uses
the same select + review crop workflow as run_sam2_markers.py and saves crop
metadata in the same crop_roi.json format.

Paste-ready examples:
    python .\\auto_detect_sam2_prompts.py --left-input .\\in\\left.mp4 --right-input .\\in\\right.mp4 --frame 150

    python .\\auto_detect_sam2_prompts.py --frame 150 --tune

    python .\\auto_detect_sam2_prompts.py --left-input .\\in\\left.mp4 --right-input .\\in\\right.mp4 --frame 150 --select-crop

    python .\\auto_detect_sam2_prompts.py --left-input .\\in\\left.mp4 --right-input .\\in\\right.mp4 --frame 150 --left-crop 420,0,2500,2160 --right-crop 390,0,2500,2160 --debug

Common tuning:
    --min-area 35           Raise if small specks are being detected.
    --max-area 1600         Lower if large net/rope blobs are being detected.
    --min-distance 35       Raise if one paint mark gets multiple detections.
    --min-score 0.45        Raise if weak false positives remain.
    --method all            Also run gray/blob diagnostics if color misses nodes.
    --max-candidates 80     Optional review/debug cap; default keeps all.

Text tuning:
    --tune opens a terminal tuning loop. Type exact values, then type apply to
    recompute previews only when you are ready. Type save to write the chosen
    parameters to work\\auto_prompts\\detection_params.json and continue detection.
    Future runs load that JSON automatically unless --ignore-saved-params is set.

Outputs per side:
    work\\auto_prompts\\left_candidates.csv
    work\\auto_prompts\\left_candidates.json
    work\\auto_prompts\\left_candidates_preview.png
    work\\auto_prompts\\left_crop_roi.json
    work\\auto_prompts\\left\\prompts\\crop_roi.json
    work\\auto_prompts\\debug_left\\...     only with --debug or --debug-dir

When you use --select-crop, the newly selected crop is automatically written to:
    out\\left\\prompts\\crop_roi.json
    out\\right\\prompts\\crop_roi.json

For non-interactive/manual crop runs, add --save-crops-to-out if you also want
to update:
    out\\left\\prompts\\crop_roi.json
    out\\right\\prompts\\crop_roi.json

CSV columns keep x,y in full-frame pixel coordinates. local_x/local_y are also
written for the cropped SAM2 coordinate space.
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_WORK_DIR = Path("work") / "auto_prompts"
DEFAULT_OUT_ROOT = Path("out")
METHOD_NAMES = ["color", "hsv", "lab", "gray", "blob", "all"]
TUNABLE_KEYS = ["method", "min_area", "max_area", "min_distance", "min_score", "max_candidates"]


@dataclass
class Candidate:
    x: float
    y: float
    area: float
    method: str
    score: float
    local_x: float
    local_y: float
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    methods: set = field(default_factory=set)

    def __post_init__(self):
        if not self.methods:
            self.methods = {self.method}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_crop_arg(crop_arg: Optional[str], flag_name: str = "--crop") -> Optional[Tuple[int, int, int, int]]:
    if crop_arg is None:
        return None
    parts = [p.strip() for p in str(crop_arg).split(",")]
    if len(parts) != 4:
        raise ValueError(f"{flag_name} must be in format x,y,w,h")
    try:
        x, y, w, h = [int(v) for v in parts]
    except ValueError as e:
        raise ValueError(f"{flag_name} values must be integers") from e
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"{flag_name} requires x>=0, y>=0, w>0, h>0")
    return x, y, w, h


def clip_crop(crop: Optional[Tuple[int, int, int, int]], frame_shape: Tuple[int, int, int]):
    h, w = frame_shape[:2]
    if crop is None:
        return 0, 0, w, h
    x, y, cw, ch = crop
    x2 = min(w, x + cw)
    y2 = min(h, y + ch)
    if x >= w or y >= h or x2 <= x or y2 <= y:
        raise ValueError(f"Crop {crop} is outside frame size {w}x{h}")
    return x, y, x2 - x, y2 - y


def crop_to_dict(crop):
    if crop is None:
        return None
    return {"x": int(crop[0]), "y": int(crop[1]), "w": int(crop[2]), "h": int(crop[3])}


def build_crop_meta(side: str, video_path: Path, crop, crop_applied: bool, frame_shape):
    h, w = frame_shape[:2]
    if crop is None:
        x, y, cw, ch = 0, 0, int(w), int(h)
    else:
        x, y, cw, ch = [int(v) for v in crop]
    return {
        "side": str(side),
        "video": str(video_path),
        "crop_applied": bool(crop_applied),
        "x": x,
        "y": y,
        "w": cw,
        "h": ch,
        "coordinate_space": "cropped" if crop_applied else "original",
        "notes": "Add x/y offsets back to tracked points to convert cropped coordinates to original video coordinates.",
    }


def save_crop_meta(path: Path, side: str, video_path: Path, crop, crop_applied: bool, frame_shape):
    ensure_parent(path)
    payload = build_crop_meta(side, video_path, crop, crop_applied, frame_shape)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def read_existing_crop(side: str, out_root: Path) -> Optional[Tuple[int, int, int, int]]:
    path = out_root / side / "prompts" / "crop_roi.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if not payload.get("crop_applied", False):
            return None
        crop = (
            int(payload.get("x", 0)),
            int(payload.get("y", 0)),
            int(payload.get("w", 0)),
            int(payload.get("h", 0)),
        )
        if crop[2] > 0 and crop[3] > 0:
            print(f"[INFO][{side}] Using existing crop from {path}: {crop}")
            return crop
    except Exception as e:
        print(f"[WARN][{side}] Could not read existing crop {path}: {e}")
    return None


def downscale(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame.copy()
    h, w = frame.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (out_w, out_h), interpolation=interp)


def robust_read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_idx < 0:
        cap.release()
        raise ValueError("--frame must be >= 0")
    if total_frames > 0 and frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"--frame {frame_idx} is outside video length {total_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if ok and frame is not None:
        cap.release()
        return frame, frame_idx, {"frame_count": total_frames, "fps": fps, "width": width, "height": height}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = None
    for _ in range(frame_idx + 1):
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame, frame_idx, {"frame_count": total_frames, "fps": fps, "width": width, "height": height}


def select_crop_roi(video_path: str):
    """
    Same crop picker used by run_sam2_markers.py.
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
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
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

    # Clamp to frame bounds, exactly like run_sam2_markers.py.
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    rw = max(1, min(w - x, rw))
    rh = max(1, min(h - y, rh))

    print(f"[INFO] Selected crop ROI x={x}, y={y}, w={rw}, h={rh}")
    return x, y, rw, rh


def _draw_crop_hud(display, frame_idx, total_frames, fps, crop, g_mode, digit_buf):
    x, y, w, h = crop
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 255), 2)

    cv2.putText(display, f"Frame: {frame_idx}/{max(0, total_frames - 1)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(display, f"Time: {frame_idx / max(fps, 1e-6):.2f}s", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(display, "a/d frame  A/D ~1s  Ctrl+A/Ctrl+D ~10s  g goto", (20, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "q accept crop  r reselect crop  Esc cancel", (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if g_mode:
        cv2.putText(display, f"Go to frame: {digit_buf}_", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)


def review_crop_over_video(video_path: str, crop, label: str):
    """Same crop review loop used by run_sam2_markers.py."""
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
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
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

        disp = frame.copy()
        _draw_crop_hud(disp, current_frame, total_frames, fps, crop, g_mode, digit_buf)
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
    """Same select/review/reselect loop used by run_sam2_markers.py."""
    while True:
        crop = select_crop_roi(video_path)
        action = review_crop_over_video(video_path, crop, label)
        if action == "accept":
            return crop


def odd_kernel(size: int, minimum: int = 3) -> int:
    size = max(minimum, int(size))
    return size if size % 2 == 1 else size + 1


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo = float(np.percentile(img, 1))
    hi = float(np.percentile(img, 99))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    out = (img - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_mask(mask: np.ndarray, close_iters: int = 1) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    if mask.size == 0:
        return mask
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=close_iters)
    return mask


def save_debug(debug_dir: Optional[Path], name: str, image: np.ndarray):
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / name), image)


def contour_metrics(component_mask: np.ndarray):
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    compactness = float(4.0 * math.pi * area / (perimeter * perimeter))
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = float(area / hull_area)
    return compactness, solidity


def component_candidates(
    mask: np.ndarray,
    score_img: np.ndarray,
    method: str,
    scale: float,
    crop_xy: Tuple[int, int],
    min_area: float,
    max_area: float,
    min_component_score: float,
) -> List[Candidate]:
    mask = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Candidate] = []
    crop_x, crop_y = crop_xy
    scale_area = scale * scale

    for label in range(1, n_labels):
        area_proc = float(stats[label, cv2.CC_STAT_AREA])
        if area_proc <= 0:
            continue
        area_full = area_proc / max(scale_area, 1e-9)
        if area_full < min_area or area_full > max_area:
            continue

        x, y, w, h = [float(v) for v in stats[label, :4]]
        aspect = max(w / max(h, 1.0), h / max(w, 1.0))
        extent = area_proc / max(w * h, 1.0)
        if aspect > 4.5 or extent < 0.16:
            continue

        component = (labels == label).astype(np.uint8) * 255
        compactness, solidity = contour_metrics(component)
        if compactness < 0.08 or solidity < 0.22:
            continue

        label_mask = labels == label
        score = float(np.percentile(score_img[label_mask], 80)) if np.any(label_mask) else 0.0
        shape_score = float(np.clip(0.55 * compactness + 0.45 * solidity, 0.0, 1.0))
        score = float(np.clip(0.82 * score + 0.18 * shape_score, 0.0, 1.0))
        if score < min_component_score:
            continue

        cx, cy = centroids[label]
        local_x = float(cx) / scale
        local_y = float(cy) / scale
        out.append(
            Candidate(
                x=crop_x + local_x,
                y=crop_y + local_y,
                local_x=local_x,
                local_y=local_y,
                area=area_full,
                method=method,
                score=score,
                bbox=(crop_x + x / scale, crop_y + y / scale, w / scale, h / scale),
            )
        )

    return out


def detect_hsv(frame: np.ndarray, scale: float, crop_xy, args, debug_dir: Optional[Path]) -> List[Candidate]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sf = s.astype(np.float32)
    vf = v.astype(np.float32)

    # Painted marks should be chromatic relative to underwater green/gray net.
    sat_thr = max(55.0, float(np.percentile(sf, 91)))
    val_floor = max(45.0, float(np.percentile(vf, 38)))
    yellow_green = (h >= 14) & (h <= 92)
    red_wrap = (h <= 8) | (h >= 165)
    blue = (h >= 95) & (h <= 135)
    hue_ok = yellow_green | red_wrap | blue
    mask = (hue_ok & (sf >= sat_thr) & (vf >= val_floor)).astype(np.uint8) * 255
    mask = clean_mask(mask, close_iters=1)

    score_img = np.clip((sf - sat_thr) / max(255.0 - sat_thr, 1.0), 0.0, 1.0)
    score_img = np.maximum(score_img, np.clip((vf - val_floor) / max(255.0 - val_floor, 1.0), 0.0, 1.0) * 0.35)

    save_debug(debug_dir, "hsv_mask.png", mask)
    save_debug(debug_dir, "hsv_saturation.png", s)
    save_debug(debug_dir, "hsv_value.png", v)
    return component_candidates(mask, score_img, "hsv", scale, crop_xy, args.min_area, args.max_area, args.min_score)


def detect_lab(frame: np.ndarray, scale: float, crop_xy, args, debug_dir: Optional[Path]) -> List[Candidate]:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)
    med_a = float(np.median(a_chan))
    med_b = float(np.median(b_chan))
    chroma = np.sqrt((a_chan - 128.0) ** 2 + (b_chan - 128.0) ** 2)
    bg_dist = np.sqrt((a_chan - med_a) ** 2 + (b_chan - med_b) ** 2)

    chroma_thr = max(12.0, float(np.percentile(chroma, 90)))
    dist_thr = max(8.0, float(np.percentile(bg_dist, 93)))
    light_floor = max(20.0, float(np.percentile(l_chan, 25)))
    mask = ((chroma >= chroma_thr) & (bg_dist >= dist_thr) & (l_chan >= light_floor)).astype(np.uint8) * 255
    mask = clean_mask(mask, close_iters=1)

    score_img = np.clip((bg_dist - dist_thr) / max(float(np.percentile(bg_dist, 99)) - dist_thr, 1.0), 0.0, 1.0)
    save_debug(debug_dir, "lab_mask.png", mask)
    save_debug(debug_dir, "lab_color_distance.png", normalize_to_u8(bg_dist))
    save_debug(debug_dir, "lab_chroma.png", normalize_to_u8(chroma))
    return component_candidates(mask, score_img, "lab", scale, crop_xy, args.min_area, args.max_area, args.min_score)


def detect_gray(frame: np.ndarray, scale: float, crop_xy, args, debug_dir: Optional[Path]) -> List[Candidate]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur_size = odd_kernel(max(31, min(gray.shape[:2]) // 18))
    background = cv2.GaussianBlur(eq, (blur_size, blur_size), 0)
    response = cv2.absdiff(eq, background)
    thr = max(18.0, float(np.percentile(response, 99.3)))
    mask = (response >= thr).astype(np.uint8) * 255
    mask = clean_mask(mask, close_iters=1)
    score_img = np.clip((response.astype(np.float32) - thr) / max(float(np.percentile(response, 99.9)) - thr, 1.0), 0.0, 1.0)

    save_debug(debug_dir, "gray_mask.png", mask)
    save_debug(debug_dir, "gray_local_contrast.png", normalize_to_u8(response))
    return component_candidates(mask, score_img, "gray", scale, crop_xy, args.min_area, args.max_area, max(args.min_score, 0.5))


def detect_blob(frame: np.ndarray, scale: float, crop_xy, args, debug_dir: Optional[Path]) -> List[Candidate]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 20
    params.maxThreshold = 245
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = max(3.0, args.min_area * scale * scale)
    params.maxArea = max(params.minArea + 1.0, args.max_area * scale * scale)
    params.minDistBetweenBlobs = max(2.0, args.min_distance * scale)
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.2
    params.filterByInertia = True
    params.minInertiaRatio = 0.08
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(eq)
    save_debug(
        debug_dir,
        "blob_keypoints.png",
        cv2.drawKeypoints(frame, keypoints, None, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
    )

    crop_x, crop_y = crop_xy
    out: List[Candidate] = []
    for kp in keypoints:
        px, py = kp.pt
        radius = max(1.0, float(kp.size) * 0.5)
        area_full = math.pi * (radius / max(scale, 1e-9)) ** 2
        if area_full < args.min_area or area_full > args.max_area:
            continue
        local_x = px / scale
        local_y = py / scale
        score = float(eq[int(round(py)) % eq.shape[0], int(round(px)) % eq.shape[1]]) / 255.0
        if score < max(args.min_score, 0.45):
            continue
        out.append(
            Candidate(
                x=crop_x + local_x,
                y=crop_y + local_y,
                local_x=local_x,
                local_y=local_y,
                area=area_full,
                method="blob",
                score=score,
                bbox=(crop_x + (px - radius) / scale, crop_y + (py - radius) / scale, kp.size / scale, kp.size / scale),
            )
        )
    return out


def merge_candidates(candidates: Sequence[Candidate], min_distance: float, max_candidates: int) -> List[Candidate]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: (-c.score, -c.area))
    kept: List[Candidate] = []
    clusters: List[List[Candidate]] = []

    for cand in ordered:
        target = None
        for i, cluster in enumerate(clusters):
            cx = sum(c.x * max(c.score, 0.05) for c in cluster) / sum(max(c.score, 0.05) for c in cluster)
            cy = sum(c.y * max(c.score, 0.05) for c in cluster) / sum(max(c.score, 0.05) for c in cluster)
            if math.hypot(cand.x - cx, cand.y - cy) <= min_distance:
                target = i
                break
        if target is None:
            clusters.append([cand])
        else:
            clusters[target].append(cand)

    for cluster in clusters:
        weights = [max(c.score, 0.05) for c in cluster]
        wsum = sum(weights)
        x = sum(c.x * w for c, w in zip(cluster, weights)) / wsum
        y = sum(c.y * w for c, w in zip(cluster, weights)) / wsum
        local_x = sum(c.local_x * w for c, w in zip(cluster, weights)) / wsum
        local_y = sum(c.local_y * w for c, w in zip(cluster, weights)) / wsum
        methods = sorted({m for c in cluster for m in c.methods})
        confidence = 1.0 - np.prod([1.0 - min(0.96, max(0.0, c.score)) for c in cluster])
        if len(methods) > 1:
            confidence = min(1.0, confidence + 0.08)
        best = max(cluster, key=lambda c: c.score)
        kept.append(
            Candidate(
                x=x,
                y=y,
                local_x=local_x,
                local_y=local_y,
                area=max(c.area for c in cluster),
                method="+".join(methods),
                score=float(np.clip(confidence, 0.0, 1.0)),
                bbox=best.bbox,
                methods=set(methods),
            )
        )

    kept = sorted(kept, key=lambda c: (-c.score, c.y, c.x))
    if max_candidates > 0:
        kept = kept[:max_candidates]
    return sorted(kept, key=lambda c: (c.y, c.x))


def draw_preview(frame_full: np.ndarray, crop, candidates: Sequence[Candidate], scale: float) -> np.ndarray:
    crop_x, crop_y, crop_w, crop_h = crop
    crop_frame = frame_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    preview = downscale(crop_frame, scale)
    text_scale = max(0.45, min(1.0, preview.shape[1] / 1600.0))
    thickness = max(1, int(round(text_scale * 2)))
    radius = max(4, int(round(7 * scale)))

    for idx, cand in enumerate(candidates):
        px = int(round(cand.local_x * scale))
        py = int(round(cand.local_y * scale))
        cv2.circle(preview, (px, py), radius + 2, (0, 0, 0), -1)
        cv2.circle(preview, (px, py), radius, (0, 255, 255), -1)
        label = str(idx)
        org = (px + radius + 4, py - radius - 4)
        cv2.putText(preview, label, org, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(preview, label, org, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return preview


def write_csv(path: Path, candidates: Sequence[Candidate]):
    ensure_parent(path)
    fields = ["candidate_id", "x", "y", "area", "method", "score", "local_x", "local_y"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, c in enumerate(candidates):
            writer.writerow(
                {
                    "candidate_id": idx,
                    "x": f"{c.x:.3f}",
                    "y": f"{c.y:.3f}",
                    "area": f"{c.area:.3f}",
                    "method": c.method,
                    "score": f"{c.score:.4f}",
                    "local_x": f"{c.local_x:.3f}",
                    "local_y": f"{c.local_y:.3f}",
                }
            )


def write_json(path: Path, side: str, input_path: Path, frame_idx: int, crop, args, video_info, method_counts, candidates):
    ensure_parent(path)
    payload = {
        "side": side,
        "input": str(input_path),
        "frame": int(frame_idx),
        "scale": float(args.scale),
        "crop": crop_to_dict(crop),
        "coordinate_note": "x/y are full-frame pixels; local_x/local_y are cropped SAM2 prompt pixels.",
        "method": args.method,
        "min_area": float(args.min_area),
        "max_area": float(args.max_area),
        "min_distance": float(args.min_distance),
        "min_score": float(args.min_score),
        "max_candidates": int(args.max_candidates),
        "video": video_info,
        "method_counts_before_merge": method_counts,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "candidate_id": idx,
                "x": round(float(c.x), 3),
                "y": round(float(c.y), 3),
                "local_x": round(float(c.local_x), 3),
                "local_y": round(float(c.local_y), 3),
                "area": round(float(c.area), 3),
                "method": c.method,
                "score": round(float(c.score), 4),
            }
            for idx, c in enumerate(candidates)
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def open_image(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"[WARN] Could not open preview {path}: {e}")


def detection_methods(method_name: str):
    return {
        "color": ["hsv", "lab"],
        "hsv": ["hsv"],
        "lab": ["lab"],
        "gray": ["gray"],
        "blob": ["blob"],
        "all": ["hsv", "lab", "gray", "blob"],
    }[method_name]


def detect_candidates_on_processed_frame(side: str, frame_proc: np.ndarray, crop_xy, args, debug_dir: Optional[Path] = None):
    all_candidates: List[Candidate] = []
    method_counts: Dict[str, int] = {}
    for method in detection_methods(args.method):
        if method == "hsv":
            found = detect_hsv(frame_proc, args.scale, crop_xy, args, debug_dir)
        elif method == "lab":
            found = detect_lab(frame_proc, args.scale, crop_xy, args, debug_dir)
        elif method == "gray":
            found = detect_gray(frame_proc, args.scale, crop_xy, args, debug_dir)
        elif method == "blob":
            found = detect_blob(frame_proc, args.scale, crop_xy, args, debug_dir)
        else:
            found = []
        method_counts[method] = len(found)
        all_candidates.extend(found)
        if side:
            print(f"[INFO][{side}] {method} candidates: {len(found)}")

    merged = merge_candidates(all_candidates, args.min_distance, args.max_candidates)
    return all_candidates, method_counts, merged


def save_detection_params(path: Path, args):
    ensure_parent(path)
    payload = {
        "method": str(args.method),
        "min_area": float(args.min_area),
        "max_area": float(args.max_area),
        "min_distance": float(args.min_distance),
        "min_score": float(args.min_score),
        "max_candidates": int(args.max_candidates),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Saved detection parameters: {path}")


def load_detection_params(path: Path):
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    params = {}
    for key in TUNABLE_KEYS:
        if key in payload:
            params[key] = payload[key]
    return params


def apply_saved_detection_params(args, parser):
    if args.ignore_saved_params:
        return
    params = load_detection_params(args.params_file)
    if not params:
        return

    applied = []
    for key, value in params.items():
        default_value = parser.get_default(key)
        if getattr(args, key) == default_value:
            if key in ("min_area", "max_area", "min_distance", "min_score"):
                value = float(value)
            elif key == "max_candidates":
                value = int(value)
            elif key == "method":
                value = str(value)
                if value not in METHOD_NAMES:
                    continue
            setattr(args, key, value)
            applied.append(key)

    if applied:
        print(f"[INFO] Loaded saved detection params from {args.params_file}: {', '.join(applied)}")


def clone_args_with_params(args, method, min_area, max_area, min_distance, min_score, max_candidates):
    tuned = argparse.Namespace(**vars(args))
    tuned.method = method
    tuned.min_area = float(min_area)
    tuned.max_area = float(max_area)
    tuned.min_distance = float(min_distance)
    tuned.min_score = float(min_score)
    tuned.max_candidates = int(max_candidates)
    return tuned


def make_tuning_preview(side_frames, candidates_by_side, counts_by_side, args, display_side):
    panels = []
    sides = ["left", "right"] if display_side == "both" else [display_side]
    for side in sides:
        frame_full, crop = side_frames[side]["frame_full"], side_frames[side]["crop"]
        preview = draw_preview(frame_full, crop, candidates_by_side.get(side, []), args.scale)
        preview = downscale(preview, min(1.0, 1500.0 / max(preview.shape[1], 1)))
        text = (
            f"{side.upper()} count={counts_by_side.get(side, 0)}  "
            f"method={args.method}  min_area={args.min_area:.0f}  max_area={args.max_area:.0f}  "
            f"min_dist={args.min_distance:.0f}  min_score={args.min_score:.2f}"
        )
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(preview, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(preview)

    if len(panels) == 1:
        canvas = panels[0]
    else:
        h = min(p.shape[0] for p in panels)
        resized = [cv2.resize(p, (int(round(p.shape[1] * h / p.shape[0])), h), interpolation=cv2.INTER_AREA) for p in panels]
        canvas = cv2.hconcat(resized)

    help_lines = [
        "Text tune preview. Use terminal commands: set min_area 50, apply, save, cancel.",
        "Saved params become defaults for future auto-detect runs.",
    ]
    footer_h = 58
    footer = np.zeros((footer_h, canvas.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(help_lines):
        cv2.putText(footer, line, (10, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
    return cv2.vconcat([canvas, footer])


def print_tune_status(args):
    print(
        "\nCurrent detection params:\n"
        f"  method         = {args.method}\n"
        f"  min_area       = {args.min_area:g}\n"
        f"  max_area       = {args.max_area:g}\n"
        f"  min_distance   = {args.min_distance:g}\n"
        f"  min_score      = {args.min_score:g}\n"
        f"  max_candidates = {args.max_candidates}  (0 keeps all)\n"
    )


def write_tuning_previews(side_frames, candidates_by_side, args):
    preview_paths = {}
    for side, data in side_frames.items():
        path = args.work_dir / f"tune_{side}_preview.png"
        preview = draw_preview(data["frame_full"], data["crop"], candidates_by_side.get(side, []), args.scale)
        ensure_parent(path)
        cv2.imwrite(str(path), preview)
        preview_paths[side] = path

    combined_path = args.work_dir / "tune_combined_preview.png"
    combined = make_tuning_preview(
        side_frames,
        candidates_by_side,
        {side: len(candidates_by_side.get(side, [])) for side in side_frames},
        args,
        "both",
    )
    ensure_parent(combined_path)
    cv2.imwrite(str(combined_path), combined)
    preview_paths["combined"] = combined_path
    return preview_paths


def compute_tuning_preview(side_frames, args, open_previews=True):
    print("[INFO][tune] Computing preview with current parameters...")
    candidates_by_side = {}
    method_counts_by_side = {}
    for side, data in side_frames.items():
        raw, counts, merged = detect_candidates_on_processed_frame(
            "",
            data["frame_proc"],
            (data["crop"][0], data["crop"][1]),
            args,
            debug_dir=None,
        )
        candidates_by_side[side] = merged
        method_counts_by_side[side] = counts
        print(f"[INFO][tune][{side}] raw={len(raw)} merged={len(merged)} methods={counts}")

    preview_paths = write_tuning_previews(side_frames, candidates_by_side, args)
    print(f"[OK][tune] Wrote combined preview: {preview_paths['combined']}")
    print(f"[OK][tune] Wrote left preview: {preview_paths['left']}")
    print(f"[OK][tune] Wrote right preview: {preview_paths['right']}")
    if open_previews:
        open_image(preview_paths["combined"])
    return preview_paths


def update_tune_param(args, key, value):
    key = key.strip().replace("-", "_")
    if key == "method":
        value = value.strip().lower()
        if value not in METHOD_NAMES:
            raise ValueError(f"method must be one of: {', '.join(METHOD_NAMES)}")
        args.method = value
    elif key in ("min_area", "max_area", "min_distance", "min_score"):
        setattr(args, key, float(value))
    elif key == "max_candidates":
        setattr(args, key, int(float(value)))
    else:
        raise ValueError(f"Unknown parameter: {key}")

    if args.min_area <= 0:
        raise ValueError("min_area must be > 0")
    if args.max_area <= args.min_area:
        raise ValueError("max_area must be greater than min_area")
    if args.min_distance < 0:
        raise ValueError("min_distance must be >= 0")
    if args.min_score < 0 or args.min_score > 1:
        raise ValueError("min_score must be between 0 and 1")
    if args.max_candidates < 0:
        raise ValueError("max_candidates must be >= 0")


def run_tuning_window(side_frames, args):
    print(
        "\n[INFO] Text tuning mode.\n"
        "Commands:\n"
        "  show                         print current values\n"
        "  set min_area 50              set exact value\n"
        "  min_distance 35              shorthand for set min_distance 35\n"
        "  method color                 method: color, hsv, lab, gray, blob, all\n"
        "  apply                        recompute previews only when ready\n"
        "  open                         reopen last combined preview\n"
        "  save                         save params and continue detection\n"
        "  cancel                       exit without saving\n"
    )
    args = argparse.Namespace(**vars(args))
    print_tune_status(args)
    last_preview_paths = compute_tuning_preview(side_frames, args, open_previews=not args.no_open_preview)

    while True:
        try:
            command = input("tune> ").strip()
        except EOFError:
            command = "save"

        if not command:
            continue

        lower = command.lower()
        if lower in ("show", "status", "?"):
            print_tune_status(args)
            continue

        if lower in ("apply", "a", "run", "recompute"):
            last_preview_paths = compute_tuning_preview(side_frames, args, open_previews=not args.no_open_preview)
            continue

        if lower in ("open", "preview"):
            if last_preview_paths:
                open_image(last_preview_paths["combined"])
            else:
                print("[WARN] No preview has been computed yet. Type apply first.")
            continue

        if lower in ("save", "s", "done", "q"):
            save_detection_params(args.params_file, args)
            return args

        if lower in ("cancel", "exit", "esc"):
            raise RuntimeError("Tuning canceled by user.")

        parts = command.split()
        try:
            if len(parts) >= 3 and parts[0].lower() == "set":
                update_tune_param(args, parts[1], " ".join(parts[2:]))
            elif len(parts) >= 2:
                update_tune_param(args, parts[0], " ".join(parts[1:]))
            else:
                print("[WARN] Could not parse command. Example: set min_area 50")
                continue
            print_tune_status(args)
            print("[INFO] Value changed. Type apply to recompute previews, or save to store without another preview.")
        except Exception as e:
            print(f"[WARN] {e}")


def prepare_tuning_frame(side: str, input_path: Path, crop_arg, args):
    frame_full, frame_idx, video_info = robust_read_frame(input_path, args.frame)
    crop = clip_crop(crop_arg, frame_full.shape)
    if crop_arg is not None and tuple(crop_arg) != tuple(crop):
        print(f"[WARN][{side}] Crop {crop_arg} was clipped to frame bounds: {crop}")
    crop_x, crop_y, crop_w, crop_h = crop
    frame_crop = frame_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    frame_proc = downscale(frame_crop, args.scale)
    return {
        "frame_full": frame_full,
        "frame_idx": frame_idx,
        "video_info": video_info,
        "crop": crop,
        "frame_proc": frame_proc,
    }


def process_side(side: str, input_path: Path, crop_arg, args):
    print(f"\n[INFO][{side}] Loading frame {args.frame}: {input_path}")
    frame_full, frame_idx, video_info = robust_read_frame(input_path, args.frame)
    crop_applied = crop_arg is not None
    crop = clip_crop(crop_arg, frame_full.shape)
    if crop_arg is not None and tuple(crop_arg) != tuple(crop):
        print(f"[WARN][{side}] Crop {crop_arg} was clipped to frame bounds: {crop}")
    crop_x, crop_y, crop_w, crop_h = crop
    frame_crop = frame_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    frame_proc = downscale(frame_crop, args.scale)

    debug_dir = None
    if args.debug or args.debug_dir:
        debug_root = args.debug_dir if args.debug_dir else (args.work_dir / f"debug_{side}")
        debug_dir = debug_root / side if args.debug_dir else debug_root
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_debug(debug_dir, "input_crop_processed.png", frame_proc)

    print(f"[INFO][{side}] Video size: {video_info['width']}x{video_info['height']}  fps: {video_info['fps']:.3f}")
    print(f"[INFO][{side}] Crop: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    print(f"[INFO][{side}] Processing size: {frame_proc.shape[1]}x{frame_proc.shape[0]}  scale={args.scale}")

    methods_to_run = {
        "color": ["hsv", "lab"],
        "hsv": ["hsv"],
        "lab": ["lab"],
        "gray": ["gray"],
        "blob": ["blob"],
        "all": ["hsv", "lab", "gray", "blob"],
    }[args.method]

    all_candidates: List[Candidate] = []
    method_counts: Dict[str, int] = {}
    for method in methods_to_run:
        if method == "hsv":
            found = detect_hsv(frame_proc, args.scale, (crop_x, crop_y), args, debug_dir)
        elif method == "lab":
            found = detect_lab(frame_proc, args.scale, (crop_x, crop_y), args, debug_dir)
        elif method == "gray":
            found = detect_gray(frame_proc, args.scale, (crop_x, crop_y), args, debug_dir)
        elif method == "blob":
            found = detect_blob(frame_proc, args.scale, (crop_x, crop_y), args, debug_dir)
        else:
            found = []
        method_counts[method] = len(found)
        all_candidates.extend(found)
        print(f"[INFO][{side}] {method} candidates: {len(found)}")

    merged = merge_candidates(all_candidates, args.min_distance, args.max_candidates)
    print(f"[INFO][{side}] Raw candidates after filters: {len(all_candidates)}")
    if args.max_candidates > 0:
        print(f"[INFO][{side}] Candidates after merge/cap: {len(merged)}")
    else:
        print(f"[INFO][{side}] Candidates after merge: {len(merged)}")

    csv_path = args.work_dir / f"{side}_candidates.csv"
    json_path = args.work_dir / f"{side}_candidates.json"
    preview_path = args.work_dir / f"{side}_candidates_preview.png"
    work_crop_path = args.work_dir / f"{side}_crop_roi.json"
    work_sam2_crop_path = args.work_dir / side / "prompts" / "crop_roi.json"
    out_crop_path = args.out_root / side / "prompts" / "crop_roi.json"
    write_csv(csv_path, merged)
    write_json(json_path, side, input_path, frame_idx, crop, args, video_info, method_counts, merged)
    save_crop_meta(work_crop_path, side, input_path, crop, crop_applied, frame_full.shape)
    save_crop_meta(work_sam2_crop_path, side, input_path, crop, crop_applied, frame_full.shape)
    should_save_crop_to_out = args.select_crop or args.save_crops_to_out
    if should_save_crop_to_out:
        save_crop_meta(out_crop_path, side, input_path, crop, crop_applied, frame_full.shape)
    preview = draw_preview(frame_full, crop, merged, args.scale)
    ensure_parent(preview_path)
    cv2.imwrite(str(preview_path), preview)

    print(f"[OK][{side}] Saved CSV: {csv_path}")
    print(f"[OK][{side}] Saved JSON: {json_path}")
    print(f"[OK][{side}] Saved crop: {work_crop_path}")
    print(f"[OK][{side}] Saved preview: {preview_path}")
    if should_save_crop_to_out:
        print(f"[OK][{side}] Updated SAM2 crop metadata: {out_crop_path}")
    if debug_dir:
        print(f"[OK][{side}] Saved debug images: {debug_dir}")
    return len(merged)


def build_argparser():
    parser = argparse.ArgumentParser(description="Detect candidate painted net nodes for LEFT/RIGHT SAM2 prompts.")
    parser.add_argument("--left-input", default=Path("in") / "left.mp4", type=Path, help="LEFT input video.")
    parser.add_argument("--right-input", default=Path("in") / "right.mp4", type=Path, help="RIGHT input video.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to detect on.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Output directory for CSV/JSON/previews.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="SAM2 output root used to find existing crop metadata.")
    parser.add_argument("--crop", default=None, help="Optional crop for both sides: x,y,w,h.")
    parser.add_argument("--left-crop", default=None, help="Optional LEFT crop: x,y,w,h.")
    parser.add_argument("--right-crop", default=None, help="Optional RIGHT crop: x,y,w,h.")
    parser.add_argument("--select-crop", action="store_true", help="Interactively select LEFT and RIGHT crops.")
    parser.add_argument("--ignore-existing-crops", action="store_true", help="Do not load crop_roi.json from existing SAM2 outputs.")
    parser.add_argument("--scale", type=float, default=1.0, help="Processing and preview scale.")
    parser.add_argument("--tune", action="store_true", help="Open text tuning loop, save chosen detection parameters, then run detection.")
    parser.add_argument("--params-file", type=Path, default=None, help="Detection parameter JSON. Default: work-dir/detection_params.json.")
    parser.add_argument("--ignore-saved-params", action="store_true", help="Do not load saved detection params before running.")
    parser.add_argument("--method", choices=METHOD_NAMES, default="color", help="Detection method set.")
    parser.add_argument("--min-area", type=float, default=35.0, help="Minimum component area in full-frame pixels.")
    parser.add_argument("--max-area", type=float, default=1600.0, help="Maximum component area in full-frame pixels.")
    parser.add_argument("--min-distance", type=float, default=28.0, help="Merge/suppression distance in full-frame pixels.")
    parser.add_argument("--min-score", type=float, default=0.38, help="Minimum per-component score.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional review cap per side; 0 keeps all candidates.")
    parser.add_argument("--debug", action="store_true", help="Write debug images under work-dir.")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional debug root directory.")
    parser.add_argument("--save-crops-to-out", action="store_true", help="Also update out/left/right/prompts/crop_roi.json for non-interactive crop runs. --select-crop always saves there.")
    parser.add_argument("--no-open-preview", action="store_true", help="Do not automatically open preview PNGs after writing them.")
    return parser


def validate_args(args):
    if args.scale <= 0:
        raise ValueError("--scale must be > 0")
    if args.frame < 0:
        raise ValueError("--frame must be >= 0")
    if args.min_area <= 0:
        raise ValueError("--min-area must be > 0")
    if args.max_area <= args.min_area:
        raise ValueError("--max-area must be greater than --min-area")
    if args.min_distance < 0:
        raise ValueError("--min-distance must be >= 0")
    if args.max_candidates < 0:
        raise ValueError("--max-candidates must be >= 0")
    if not args.left_input.exists():
        raise FileNotFoundError(f"LEFT input video not found: {args.left_input}")
    if not args.right_input.exists():
        raise FileNotFoundError(f"RIGHT input video not found: {args.right_input}")


def resolve_crops(args):
    both_crop = parse_crop_arg(args.crop, "--crop")
    left_crop = parse_crop_arg(args.left_crop, "--left-crop") or both_crop
    right_crop = parse_crop_arg(args.right_crop, "--right-crop") or both_crop

    if args.select_crop:
        print("[INFO] Select and review LEFT crop...")
        left_crop = select_crop_with_review(str(args.left_input), "LEFT")
        print(f"[INFO][left] Selected crop: {left_crop}")
        print("[INFO] Select and review RIGHT crop...")
        right_crop = select_crop_with_review(str(args.right_input), "RIGHT")
        print(f"[INFO][right] Selected crop: {right_crop}")

    if not args.ignore_existing_crops:
        if left_crop is None:
            left_crop = read_existing_crop("left", args.out_root)
        if right_crop is None:
            right_crop = read_existing_crop("right", args.out_root)

    return left_crop, right_crop


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.params_file is None:
        args.params_file = args.work_dir / "detection_params.json"
    apply_saved_detection_params(args, parser)
    validate_args(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    left_crop, right_crop = resolve_crops(args)

    if args.tune:
        side_frames = {
            "left": prepare_tuning_frame("left", args.left_input, left_crop, args),
            "right": prepare_tuning_frame("right", args.right_input, right_crop, args),
        }
        args = run_tuning_window(side_frames, args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_side, "left", args.left_input, left_crop, args): "left",
            executor.submit(process_side, "right", args.right_input, right_crop, args): "right",
        }
        counts = {}
        for future in concurrent.futures.as_completed(futures):
            side = futures[future]
            counts[side] = future.result()

    left_count = counts["left"]
    right_count = counts["right"]

    summary_path = args.work_dir / "summary.json"
    summary = {
        "frame": int(args.frame),
        "work_dir": str(args.work_dir),
        "left_candidates": int(left_count),
        "right_candidates": int(right_count),
        "left_input": str(args.left_input),
        "right_input": str(args.right_input),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] Saved summary: {summary_path}")

    if not args.no_open_preview:
        open_image(args.work_dir / "left_candidates_preview.png")
        open_image(args.work_dir / "right_candidates_preview.png")


if __name__ == "__main__":
    main()
