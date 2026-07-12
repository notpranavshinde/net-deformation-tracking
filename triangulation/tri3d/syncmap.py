"""Synchronize left/right videos and translate saved trim metadata to frame drops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibration.stereo_checker_debug import (
    AUDIO_MIN_SEPARATION_FRAMES_DEFAULT,
    AUDIO_TOPK_DEFAULT,
    brightness_curve,
    extract_audio_envelope,
    find_audio_peak_candidates,
    pick_flash_frame,
    prompt_audio_choice,
    prompt_yes_no,
    render_audio_peak_preview,
    show,
)
from .viz_encode import open_video

SYNC_FLASH_STEP_DEFAULT = 5


def extract_sync_curves(
    video_path: str, scale: float, max_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns frame indices, brightness curve, and motion curve.
    - brightness: mean gray value
    - motion: mean absolute diff(gray_t - gray_{t-1})
    Uses sequential reads to avoid random-seek HEVC instability.
    """
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_scan = total if max_frames < 0 else min(total, max_frames)

    idxs = []
    brightness = []
    motion = []

    prev_gray = None
    idx = 0
    while idx < max_scan:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b = float(np.mean(gray))
        if prev_gray is None:
            m = 0.0
        else:
            m = float(np.mean(cv2.absdiff(gray, prev_gray)))

        idxs.append(idx)
        brightness.append(b)
        motion.append(m)

        prev_gray = gray
        idx += 1

    cap.release()
    return np.asarray(idxs, dtype=np.int32), np.asarray(brightness, dtype=np.float32), np.asarray(motion, dtype=np.float32)


def pick_event_frame(
    idxs: np.ndarray, brightness: np.ndarray, motion: np.ndarray, mode: str
) -> tuple[int, str]:
    if len(idxs) == 0:
        raise RuntimeError("No frames available for sync detection.")

    if mode == "flash":
        if len(brightness) < 5:
            return int(idxs[int(np.argmax(brightness))]), "flash-peak"
        diff = np.diff(brightness)
        j = int(np.argmax(diff))
        return int(idxs[j + 1]), "flash-rise"

    # hybrid: combine normalized flash-rise and motion impulse for robustness.
    if len(brightness) >= 2:
        bdiff = np.zeros_like(brightness)
        bdiff[1:] = np.diff(brightness)
    else:
        bdiff = np.zeros_like(brightness)

    b_std = float(np.std(bdiff))
    m_std = float(np.std(motion))
    b_norm = bdiff / (b_std + 1e-6)
    m_norm = motion / (m_std + 1e-6)
    score = b_norm + m_norm
    j = int(np.argmax(score))
    return int(idxs[j]), "hybrid"


def detect_sync_from_videos(left_video: str,
                            right_video: str,
                            mode: str,
                            scale: float,
                            max_frames: int,
                            preview_out_path: str | None = None) -> tuple[int, int, dict[str, Any]]:
    if mode == "audio":
        left_audio = extract_audio_envelope(left_video, max_frames=max_frames)
        right_audio = extract_audio_envelope(right_video, max_frames=max_frames)

        candL = find_audio_peak_candidates(
            left_audio["env"],
            left_audio["audio_rate"],
            left_audio["video_fps"],
            left_audio["max_scan_frames"],
            top_k=AUDIO_TOPK_DEFAULT,
            min_separation_frames=AUDIO_MIN_SEPARATION_FRAMES_DEFAULT,
        )
        candR = find_audio_peak_candidates(
            right_audio["env"],
            right_audio["audio_rate"],
            right_audio["video_fps"],
            right_audio["max_scan_frames"],
            top_k=AUDIO_TOPK_DEFAULT,
            min_separation_frames=AUDIO_MIN_SEPARATION_FRAMES_DEFAULT,
        )

        if len(candL) == 0 or len(candR) == 0:
            raise RuntimeError("[SYNC] No audio peak candidates found.")

        if preview_out_path:
            audio_preview_png = str(Path(preview_out_path).with_suffix("")) + "_audio_peaks.png"
        else:
            audio_preview_png = str(Path("triang_sync").with_suffix("")) + "_audio_peaks.png"

        try:
            audio_preview = render_audio_peak_preview(
                left_audio["env"], right_audio["env"], candL, candR, audio_preview_png
            )
            print(f"[SYNC] Saved audio peaks preview: {audio_preview_png}")
            print("[SYNC] Close audio peaks window to continue.")
            show("SYNC AUDIO PEAKS", audio_preview, wait=0)
        except Exception as e:
            print(f"[SYNC] WARNING: audio peak preview failed ({e}). Continuing.")

        chooseL = chooseR = 0
        print(
            f"[SYNC] First guess from strongest peaks: "
            f"LEFT frame={candL[0]['frame_idx']}, RIGHT frame={candR[0]['frame_idx']}"
        )
        accept_guess = prompt_yes_no("[SYNC] Accept this audio sync guess?", default_yes=True)
        if not accept_guess:
            chooseL = prompt_audio_choice("LEFT", candL, default_rank=1)
            chooseR = prompt_audio_choice("RIGHT", candR, default_rank=1)

        fL = int(candL[chooseL]["frame_idx"])
        fR = int(candR[chooseR]["frame_idx"])
        if accept_guess:
            reasonL, reasonR = "audio-peak-auto", "audio-peak-auto"
        else:
            reasonL = f"audio-peak-candidate-{chooseL + 1}"
            reasonR = f"audio-peak-candidate-{chooseR + 1}"

        offset = fL - fR
        if offset > 0:
            left_drop, right_drop = offset, 0
        else:
            left_drop, right_drop = 0, -offset

        info = {
            "mode": mode,
            "left_event_frame": int(fL),
            "right_event_frame": int(fR),
            "left_reason": reasonL,
            "right_reason": reasonR,
            "offset_left_minus_right": int(offset),
            "left_drop": int(left_drop),
            "right_drop": int(right_drop),
            "sync_max_frames": int(max_frames),
        }
        return left_drop, right_drop, info

    if mode == "flash":
        idxL, meanL, _ = brightness_curve(left_video, scale, SYNC_FLASH_STEP_DEFAULT, max_frames, use_cuda=False)
        idxR, meanR, _ = brightness_curve(right_video, scale, SYNC_FLASH_STEP_DEFAULT, max_frames, use_cuda=False)

        fL = pick_flash_frame(idxL, meanL)
        fR = pick_flash_frame(idxR, meanR)
        reasonL, reasonR = "flash", "flash"

        offset = fL - fR
        if offset > 0:
            left_drop, right_drop = offset, 0
        else:
            left_drop, right_drop = 0, -offset

        info = {
            "mode": mode,
            "left_event_frame": int(fL),
            "right_event_frame": int(fR),
            "left_reason": reasonL,
            "right_reason": reasonR,
            "offset_left_minus_right": int(offset),
            "left_drop": int(left_drop),
            "right_drop": int(right_drop),
            "sync_scale": float(scale),
            "sync_max_frames": int(max_frames),
            "sync_step": int(SYNC_FLASH_STEP_DEFAULT),
        }
        return left_drop, right_drop, info

    idxL, bL, mL = extract_sync_curves(left_video, scale=scale, max_frames=max_frames)
    idxR, bR, mR = extract_sync_curves(right_video, scale=scale, max_frames=max_frames)

    fL, reasonL = pick_event_frame(idxL, bL, mL, mode)
    fR, reasonR = pick_event_frame(idxR, bR, mR, mode)

    offset = fL - fR
    if offset > 0:
        left_drop, right_drop = offset, 0
    else:
        left_drop, right_drop = 0, -offset

    info = {
        "mode": mode,
        "left_event_frame": int(fL),
        "right_event_frame": int(fR),
        "left_reason": reasonL,
        "right_reason": reasonR,
        "offset_left_minus_right": int(offset),
        "left_drop": int(left_drop),
        "right_drop": int(right_drop),
        "sync_scale": float(scale),
        "sync_max_frames": int(max_frames),
    }
    return left_drop, right_drop, info


def load_calibration_sync(sync_json_path: str) -> tuple[int, int, dict[str, Any]]:
    """Load calibration sync.json and convert trim values to triangulation drops.

    Calibration maps frames with: global = frame - trim.
    This script maps frames with: global = frame + drop.
    Therefore drop = -trim.
    """
    path = Path(sync_json_path)
    data = json.loads(path.read_text())
    if "trim_left" not in data or "trim_right" not in data:
        raise ValueError(f"{path} is not a calibration sync JSON with trim_left/trim_right.")

    trim_left = int(data["trim_left"])
    trim_right = int(data["trim_right"])
    left_drop = -trim_left
    right_drop = -trim_right
    info = {
        "mode": "calibration_sync_json",
        "source": str(path),
        "trim_left": trim_left,
        "trim_right": trim_right,
        "left_drop": left_drop,
        "right_drop": right_drop,
    }
    for key in ("left_flash", "right_flash", "left_reason", "right_reason", "scale"):
        if key in data:
            info[key] = data[key]
    return left_drop, right_drop, info
