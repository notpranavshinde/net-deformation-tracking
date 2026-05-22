r"""
Triangulate SAM2 2D tracks into 3D points using stereo calibration.

Pipeline context:
1) Calibrate stereo once using calibration script -> produces stereo.npz
2) Run SAM2 for each camera -> produces tracks_2d.csv files
3) Run this script to:
     - detect per-video sync (audio/flash/hybrid) OR use manual drops
     - match left/right 2D observations by (global frame, obj_id)
     - triangulate 3D points
     - compute reprojection error and validity flags
     - optionally generate side-by-side 3D verification video

-------------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------------

Required inputs:
- none, if you use the standard repo layout:
  calibration/work/stereo.npz
  calibration/work/sync.json
  sam2/sam2/out/left/tracks_2d.csv
  sam2/sam2/out/right/tracks_2d.csv

Sync options (choose ONE approach):
A) Reuse calibration sync JSON (recommended when experiment clips were split
   from the same synchronized full videos)
     - --sync-json calibration/work/sync.json
     - This is the default.

B) Auto sync from videos
     - --left-video / --right-video
     - --sync-mode audio|flash|hybrid
     - Explicit --sync-mode automatically disables the default calibration sync.

C) Manual sync override
     - --left-drop N --right-drop M
     (skips auto sync detection)

-------------------------------------------------------------------------------
COMMON COMMANDS
-------------------------------------------------------------------------------

1) Recommended standard repo run
python .\triangulation\points_to_3d.py

2) Reuse calibration sync with custom paths
python points_to_3d.py \
    --stereo calibration/work/stereo.npz \
    --left sam2/sam2/out/left/tracks_2d.csv \
    --right sam2/sam2/out/right/tracks_2d.csv \
    --sync-json calibration/work/sync.json \
    --out-csv triangulation/out/triangulated_3d.csv \
    --out-summary triangulation/out/summary.json

3) Audio sync + outputs
python points_to_3d.py \
    --stereo work/stereo.npz \
    --left sam2/tracks_L.csv \
    --right sam2/tracks_R.csv \
    --left-video sam2/in/left.MP4 \
    --right-video sam2/in/right.MP4 \
    --sync-mode audio \
    --sync-max-frames 300 \
    --out-sync work/detected_sync.json \
    --out-csv work/tracks_3d.csv \
    --out-summary work/triang_summary.json

4) Audio sync + visualization video for quick QA
python points_to_3d.py \
    --stereo work/stereo.npz \
    --left sam2/tracks_L.csv \
    --right sam2/tracks_R.csv \
    --left-video sam2/in/left.MP4 \
    --right-video sam2/in/right.MP4 \
    --sync-mode audio \
    --visualize \
    --viz-out work/triangulation_3d_verify.mp4 \
    --viz-max-frames 500 \
    --out-csv work/tracks_3d.csv \
    --out-summary work/triang_summary.json

5) Visualization only from an existing triangulated CSV
python points_to_3d.py \
    --viz-only \
    --visualize \
    --out-csv triangulation/out/triangulated_3d.csv \
    --left-video sam2/sam2/in/left.mp4 \
    --viz-out triangulation/out/triangulated_3d_viz.mp4

    python .\triangulation\points_to_3d.py --stereo .\calibration\work\stereo.npz --left .\sam2\sam2\out\left\trals_2d_left.csv --right .\sam2\sam2\out\right\tracks_2d_right.csv --left-video .\sam2\sam2\in\left.MP4 --right-video .\sam2\sam2\in\right.MP4 --sync-mode audio --visualize --viz-out .\triangulation\out\3d_out.mp4 --viz-max-frames 10000 --out-csv .\triangulation\out\triangulated_3d.csv --out-summary .\triangulation\out\summary.json

6) Manual sync (if you already know drops)
python points_to_3d.py \
    --stereo work/stereo.npz \
    --left sam2/tracks_L.csv \
    --right sam2/tracks_R.csv \
    --left-drop 5 --right-drop 0 \
    --out-csv work/tracks_3d.csv \
    --out-summary work/triang_summary.json

6) Stricter filtering by 2D quality and reprojection error
python points_to_3d.py \
    --stereo work/stereo.npz \
    --left sam2/tracks_L.csv \
    --right sam2/tracks_R.csv \
    --left-video sam2/in/left.MP4 \
    --right-video sam2/in/right.MP4 \
    --sync-mode audio \
    --quality-min 0.5 \
    --max-reproj 8.0 \
    --out-csv work/tracks_3d.csv \
    --out-summary work/triang_summary.json

-------------------------------------------------------------------------------
KEY FLAGS
-------------------------------------------------------------------------------

--sync-mode:
    audio  -> uses audio peak (requires ffmpeg on PATH)
    flash  -> uses visual brightness rise
    hybrid -> combines visual brightness rise + motion impulse

--sync-max-frames:
    -1 scans entire video for sync event.
    Use smaller values (e.g., 200-500) if the sync signal happens near start.

--quality-min:
    Filter low-confidence 2D points before triangulation.

--max-reproj:
    Reject 3D points with mean reprojection error above threshold (pixels).

--visualize / --viz-out / --viz-max-frames:
    Generate side-by-side left/right verification video with observed points,
    reprojected points, per-object depth (Z), and reprojection error.

--viz-only:
    Skip triangulation and render visualization from an existing --out-csv.

--workers:
    Worker processes for scene visualization iso/topdown rendering. 0=auto.

-------------------------------------------------------------------------------
OUTPUTS
-------------------------------------------------------------------------------

--out-csv:
    Per matched observation with frame mapping, XYZ, reprojection errors, and
    valid_3d flag. Includes observed/reprojected 2D points for debugging.

--out-summary:
    Aggregate counts and reprojection stats.

--out-sync (optional):
    Detected sync metadata (event frames, offset, frame drops).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.stereo_checker_debug import (  # noqa: E402
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


SYNC_FLASH_STEP_DEFAULT = 5


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def extract_sync_curves(video_path: str, scale: float, max_frames: int):
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


def pick_event_frame(idxs: np.ndarray, brightness: np.ndarray, motion: np.ndarray, mode: str):
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
                            preview_out_path: str = None):
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


def load_calibration_sync(sync_json_path: str):
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


def undist_norm_points(K, D, pts_uv: np.ndarray) -> np.ndarray:
    """
    pts_uv: (N,2) float in pixel coords
    returns: (N,2) float normalized coords (x,y) on z=1 plane
    """
    pts = pts_uv.reshape(-1, 1, 2).astype(np.float64)
    # undistortPoints returns normalized coords if P=None
    und = cv2.undistortPoints(pts, K, D, P=None)
    return und.reshape(-1, 2)


def project_points(K, D, pts_xyz: np.ndarray) -> np.ndarray:
    """
    pts_xyz: (N,3) in camera coordinates
    returns: (N,2) pixel coords
    """
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    img, _ = cv2.projectPoints(pts_xyz.astype(np.float64), rvec, tvec, K, D)
    return img.reshape(-1, 2)


def color_for_obj(obj_id: int):
    rng = np.random.RandomState(int(obj_id))
    return tuple(int(v) for v in rng.randint(50, 255, size=3))


def depth_to_bgr(z: float, z_min: float, z_max: float):
    """Map a depth value to a BGR color using a viridis-like ramp."""
    if z_max <= z_min:
        t = 0.5
    else:
        t = (z - z_min) / (z_max - z_min)
    t = float(np.clip(t, 0.0, 1.0))
    stops = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 144, 141],
        [94, 201, 98],
        [253, 231, 37],
    ], dtype=np.float32)  # RGB
    pos = t * (len(stops) - 1)
    i0 = int(np.floor(pos))
    i1 = min(i0 + 1, len(stops) - 1)
    f = pos - i0
    rgb = (1 - f) * stops[i0] + f * stops[i1]
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def draw_value_colorbar(img: np.ndarray, value_min: float, value_max: float,
                        x: int, y: int, w: int = 18, h: int = 180,
                        label: str = "value"):
    for i in range(h):
        t = 1.0 - (i / max(h - 1, 1))
        value = value_min + t * (value_max - value_min)
        col = depth_to_bgr(value, value_min, value_max)
        cv2.rectangle(img, (x, y + i), (x + w, y + i + 1), col, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(img, f"{value_max:.3f}", (x + w + 6, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"{value_min:.3f}", (x + w + 6, y + h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, label, (x - 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_colorbar(img: np.ndarray, z_min: float, z_max: float,
                  x: int, y: int, w: int = 18, h: int = 180):
    draw_value_colorbar(img, z_min, z_max, x, y, w=w, h=h, label="depth")


def draw_cross(img: np.ndarray, x: int, y: int, color, size: int = 6, thickness: int = 2):
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def create_3d_visualization_video(out_rows: list,
                                  left_video: str,
                                  out_path: str,
                                  max_frames: int = -1):
    if len(out_rows) == 0:
        print("[VIS] No rows to visualize. Skipping.")
        return

    capL = open_video(left_video)
    fpsL = float(capL.get(cv2.CAP_PROP_FPS))
    fps = fpsL if fpsL > 0 else 30.0

    grouped = {}
    for row in out_rows:
        f = int(row["frame_L"])
        grouped.setdefault(f, []).append(row)

    target_frames = sorted(grouped.keys())
    if max_frames > 0:
        target_frames = target_frames[:max_frames]
    if len(target_frames) == 0:
        capL.release()
        print("[VIS] No frames selected for visualization. Skipping.")
        return

    ok, first_frame = capL.read()
    if not ok:
        capL.release()
        raise RuntimeError("[VIS] Could not read first frame from left video.")

    h, w_single = first_frame.shape[:2]
    w = w_single * 2
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    current_idx = 0
    frame = first_frame
    target_ptr = 0
    next_target = int(target_frames[target_ptr])

    while True:
        if current_idx == next_target:
            rows = grouped[next_target]
            vis_orig = frame.copy()
            vis_csv = np.zeros_like(vis_orig)

            valid_count = 0
            for r in rows:
                obj_id = int(r["obj_id"])
                col = color_for_obj(obj_id)
                valid = int(r["valid_3d"]) == 1
                if valid:
                    valid_count += 1

                u = int(round(float(r["uL"])))
                v = int(round(float(r["vL"])))
                if u < 0 or v < 0 or u >= w_single or v >= h:
                    continue

                draw_col = col if valid else (120, 120, 120)
                r_size = 5 if valid else 3
                cv2.circle(vis_csv, (u, v), r_size, draw_col, -1)
                cv2.circle(vis_orig, (u, v), 3, draw_col, -1)

            cv2.putText(vis_orig, "Original (Left)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_csv, "CSV Reconstruction", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            both = np.hstack([vis_orig, vis_csv])
            cv2.putText(both, f"frame={next_target}  active_nodes={len(rows)}  valid_3d={valid_count}",
                        (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            writer.write(both)

            target_ptr += 1
            if target_ptr >= len(target_frames):
                break
            next_target = int(target_frames[target_ptr])

            if target_ptr % max(1, len(target_frames) // 10) == 0:
                print(f"[VIS] rendered {target_ptr}/{len(target_frames)} frames")

        ok, frame = capL.read()
        if not ok:
            break
        current_idx += 1

        if current_idx > int(target_frames[-1]):
            break

    writer.release()
    capL.release()
    print(f"[OK] Wrote visualization video: {out_path}")


_WORKER = {}
VIZ_CMAP = "viridis"


def _worker_init(kind, w, h, x_lim, y_lim, z_lim, z_min, z_max, disp_max, iso_azim):
    """Initialize a reusable matplotlib figure inside each worker process."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dpi = 100
    disp_max = max(float(disp_max), 1e-9)
    norm = colors.Normalize(vmin=0.0, vmax=disp_max)
    cmap = plt.get_cmap(VIZ_CMAP)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="white")
    if kind == "iso":
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        x_lo, x_hi = x_lim
        y_lo, y_hi = y_lim
        z_lo, z_hi = z_lim
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_zlim(z_lo, z_hi)
        ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
        ax.set_xlabel("x (m)", color="black", fontsize=22, labelpad=18)
        ax.set_ylabel("y (m)", color="black", fontsize=22, labelpad=18)
        ax.set_zlabel("z (m)", color="black", fontsize=22, labelpad=18)
        ax.tick_params(colors="black", labelsize=18)
        ax.view_init(elev=30, azim=float(iso_azim))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 1))
            axis.pane.set_edgecolor((0.75, 0.75, 0.75, 1))
            axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 1)
            axis._axinfo["grid"]["linewidth"] = 1.0
        ax.grid(True)
        yy, zz = np.meshgrid(
            np.linspace(y_lo, y_hi, 2),
            np.linspace(z_lo, z_hi, 2),
        )
        xx = np.zeros_like(yy)
        ax.plot_surface(
            xx, yy, zz,
            color=(0.86, 0.86, 0.86, 0.25),
            edgecolor=(0.50, 0.50, 0.50, 0.80),
            linewidth=1.0,
            shade=False,
            alpha=0.25,
            zorder=0,
        )
        scatter = ax.scatter([], [], [], s=88, c=[], cmap=cmap, norm=norm,
                             depthshade=False, edgecolors="black", linewidths=0.25)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.05)
        cbar.set_label("|Δpos from t0| (m)", color="black", fontsize=18)
        cbar.ax.tick_params(colors="black", labelsize=14)
        fig.subplots_adjust(left=0.02, right=0.94, top=0.98, bottom=0.04)
    else:
        ax = fig.add_subplot(111, facecolor="white")
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("")
        ax.set_ylabel("z (m)", color="black", fontsize=24, labelpad=14)
        ax.xaxis.tick_top()
        ax.text(0.0, 1.08, "x (m)", transform=ax.transAxes,
                color="black", fontsize=26, ha="left", va="bottom")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(colors="black", labelsize=20, width=1.5, length=7)
        ax.grid(True, color=(0.78, 0.78, 0.78), linestyle="-", linewidth=1.0)
        ax.axvline(0.0, color=(0.45, 0.45, 0.45), linestyle="--", linewidth=1.5)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)
        scatter = ax.scatter([], [], s=88, c=[], cmap=cmap, norm=norm,
                             edgecolors="black", linewidths=0.25)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08)
        cbar.set_label("|Δpos from t0| (m)", color="black", fontsize=18)
        cbar.ax.tick_params(colors="black", labelsize=14)
        fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)

    _WORKER.update(dict(
        kind=kind, fig=fig, ax=ax, scatter=scatter,
        w=w, h=h, cmap=cmap, norm=norm,
    ))


def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _open_writer(path, fps, w, h, encoder):
    """Returns either a cv2.VideoWriter or a dict wrapping an ffmpeg pipe."""
    path = str(path)
    if encoder == "nvenc" and _has_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
            "-i", "-",
            "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
            path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return {"proc": proc, "kind": "ffmpeg"}
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return {"writer": vw, "kind": "cv2"}


def _writer_write(w, img):
    if w["kind"] == "ffmpeg":
        w["proc"].stdin.write(img.tobytes())
    else:
        w["writer"].write(img)


def _writer_close(w):
    if w["kind"] == "ffmpeg":
        w["proc"].stdin.close()
        w["proc"].wait()
    else:
        w["writer"].release()


def _draw_chunk(args):
    kind, frame_ids, chunk_xyz, fps, segment_path, encoder = args
    fig = _WORKER["fig"]
    scatter = _WORKER["scatter"]
    w = _WORKER["w"]
    h = _WORKER["h"]
    cmap = _WORKER["cmap"]
    norm = _WORKER["norm"]

    writer = _open_writer(segment_path, fps, w, h, encoder)

    for fid, (Xs, Ys, Zs, Ds) in zip(frame_ids, chunk_xyz):
        colors = cmap(norm(Ds)) if len(Ds) > 0 else np.zeros((0, 4), dtype=np.float32)
        if kind == "iso":
            scatter._offsets3d = (Xs, Ys, Zs)
            scatter.set_array(np.asarray(Ds, dtype=np.float64))
            scatter.set_facecolor(colors)
            scatter.set_edgecolor(colors)
        else:
            if len(Xs) > 0:
                scatter.set_offsets(np.column_stack([Xs, Zs]))
            else:
                scatter.set_offsets(np.zeros((0, 2)))
            scatter.set_array(np.asarray(Ds, dtype=np.float64))
            scatter.set_facecolor(colors)
            scatter.set_edgecolor(colors)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        cw, ch = fig.canvas.get_width_height()
        img = buf.reshape(ch, cw, 4)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))

        title = "Local net frame: out-of-plane x" if kind == "iso" else "Top-down: x deformation vs z"
        cv2.putText(img, f"{title}  frame={fid}", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        _writer_write(writer, img)

    _writer_close(writer)
    return segment_path


def _reference_positions(valid_rows):
    refs = {}
    for r in sorted(valid_rows, key=lambda row: (int(row["frame_L"]), int(row["obj_id"]))):
        obj_id = int(r["obj_id"])
        if obj_id not in refs:
            refs[obj_id] = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])], dtype=np.float64)
    return refs


def _attach_displacements(valid_rows):
    refs = _reference_positions(valid_rows)
    by_obj = {}
    disp_max = 0.0
    for r in valid_rows:
        obj_id = int(r["obj_id"])
        xyz = np.array([float(r["X"]), float(r["Y"]), float(r["Z"])], dtype=np.float64)
        disp = float(np.linalg.norm(xyz - refs[obj_id]))
        r["_disp_m"] = disp
        by_obj.setdefault(obj_id, []).append(disp)
        disp_max = max(disp_max, disp)

    motion = {obj_id: float(np.percentile(vals, 95)) for obj_id, vals in by_obj.items()}
    if not motion:
        return refs, set(), max(disp_max, 1e-9)
    cutoff = float(np.percentile(list(motion.values()), 15))
    fixed_ids = {obj_id for obj_id, value in motion.items() if value <= cutoff}
    return refs, fixed_ids, max(disp_max, 1e-9)


def _build_net_reference_frame(refs, fixed_ids):
    ref_items = sorted(refs.items())
    ref_xyz = np.array([xyz for _obj_id, xyz in ref_items], dtype=np.float64)
    origin = ref_xyz.mean(axis=0)
    centered = ref_xyz - origin
    if len(ref_xyz) >= 3:
        _vals, vecs = np.linalg.eigh(np.cov(centered.T))
        normal = vecs[:, 0]
        plane_y = vecs[:, 2]
        plane_z = vecs[:, 1]
    else:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        plane_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        plane_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    fixed_xyz = np.array(
        [refs[obj_id] for obj_id in fixed_ids if obj_id in refs],
        dtype=np.float64,
    )
    if fixed_xyz.size == 0:
        fixed_xyz = ref_xyz

    ref_local = np.column_stack([
        centered @ normal,
        centered @ plane_y,
        centered @ plane_z,
    ])
    fixed_centered = fixed_xyz - origin
    fixed_local = np.column_stack([
        fixed_centered @ normal,
        fixed_centered @ plane_y,
        fixed_centered @ plane_z,
    ])

    # Put the low-motion fixed end toward small y and high z in the local net plane.
    if float(np.mean(fixed_local[:, 1])) > float(np.mean(ref_local[:, 1])):
        plane_y = -plane_y
        ref_local[:, 1] *= -1.0
        fixed_local[:, 1] *= -1.0
    if float(np.mean(fixed_local[:, 2])) < float(np.mean(ref_local[:, 2])):
        plane_z = -plane_z
        ref_local[:, 2] *= -1.0
        fixed_local[:, 2] *= -1.0

    basis = np.vstack([normal, plane_y, plane_z])
    return origin, basis, ref_local, fixed_local


def _to_local_xyz(xyz, origin, basis):
    return (np.asarray(xyz, dtype=np.float64) - origin) @ basis.T


def _choose_iso_azim(ref_local, fixed_local):
    """Pick an azimuth that projects the low-motion fixed end to top-left."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    candidates = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    x_lim = (float(ref_local[:, 0].min()), float(ref_local[:, 0].max()))
    y_lim = (float(ref_local[:, 1].min()), float(ref_local[:, 1].max()))
    z_lim = (float(ref_local[:, 2].min()), float(ref_local[:, 2].max()))
    best = (-1e18, -120)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
    all_center = ref_local.mean(axis=0)
    fixed_center = fixed_local.mean(axis=0)
    for azim in candidates:
        ax.view_init(elev=30, azim=azim)
        fig.canvas.draw()
        fx, fy, _ = proj3d.proj_transform(*fixed_center, ax.get_proj())
        axc, ayc, _ = proj3d.proj_transform(*all_center, ax.get_proj())
        # Match the rendered image: fixed end should project left and high, while
        # keeping an oblique view so out-of-plane deformation remains visible.
        oblique_preference = -0.006 * abs(azim - 60) / 60.0
        score = (fx - axc) + (fy - ayc) + oblique_preference
        if score > best[0]:
            best = (float(score), int(azim))
    plt.close(fig)
    return best[1]


def _concat_segments(segments, out_path):
    """Lossless concat of mp4 segments via ffmpeg."""
    if len(segments) == 1:
        shutil.move(segments[0], out_path)
        return
    list_file = Path(out_path).with_suffix(".list.txt")
    list_file.write_text("\n".join(f"file '{Path(s).resolve()}'" for s in segments))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)
    for s in segments:
        Path(s).unlink(missing_ok=True)


def _render_panel_parallel(kind, frame_ids, per_frame_xyz, fps, out_path,
                           w, h, x_lim, y_lim, z_lim, z_min, z_max,
                           workers, encoder, disp_max, iso_azim):
    n = len(frame_ids)
    if n == 0:
        return
    workers = max(1, min(workers, n))
    chunk_size = (n + workers - 1) // workers
    chunks = []
    tmpdir = Path(tempfile.mkdtemp(prefix=f"viz_{kind}_"))
    for i in range(workers):
        s = i * chunk_size
        e = min(s + chunk_size, n)
        if s >= e:
            break
        seg = tmpdir / f"seg_{i:04d}.mp4"
        chunks.append((kind, frame_ids[s:e], per_frame_xyz[s:e], fps, str(seg), encoder))

    if workers == 1:
        _worker_init(kind, w, h, x_lim, y_lim, z_lim, z_min, z_max, disp_max, iso_azim)
        segments = [_draw_chunk(c) for c in chunks]
    else:
        with Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(kind, w, h, x_lim, y_lim, z_lim, z_min, z_max, disp_max, iso_azim),
        ) as pool:
            segments = pool.map(_draw_chunk, chunks)

    if not _has_ffmpeg() and len(segments) > 1:
        print("[VIS] ffmpeg not found; concatenating via cv2 (slower).")
        first = cv2.VideoCapture(segments[0])
        fps_v = first.get(cv2.CAP_PROP_FPS) or fps
        first.release()
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))
        for seg in segments:
            cap = cv2.VideoCapture(seg)
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                vw.write(frm)
            cap.release()
            Path(seg).unlink(missing_ok=True)
        vw.release()
    else:
        _concat_segments(segments, str(out_path))

    shutil.rmtree(tmpdir, ignore_errors=True)


def create_3d_scene_visualization(out_rows: list,
                                  left_video: str,
                                  out_path: str,
                                  max_frames: int = -1,
                                  trail_len: int = 60,
                                  workers: int = 0,
                                  encoder: str = "auto"):
    """Render left overlay, isometric 3D, and top-down videos as separate files."""
    _ = trail_len
    if len(out_rows) == 0:
        print("[VIS] No rows to visualize. Skipping.")
        return

    valid_rows = [r for r in out_rows if int(r["valid_3d"]) == 1]
    if not valid_rows:
        print("[VIS] No valid 3D rows. Falling back to side-by-side viz.")
        create_3d_visualization_video(out_rows, left_video, out_path, max_frames)
        return

    refs, fixed_ids, disp_max = _attach_displacements(valid_rows)
    origin, basis, ref_local, fixed_local = _build_net_reference_frame(refs, fixed_ids)
    iso_azim = _choose_iso_azim(ref_local, fixed_local)
    print(
        f"[VIS] displacement color scale: 0-{disp_max:.4f} m; "
        f"fixed_ids={len(fixed_ids)}; iso_azim={iso_azim}"
    )

    local_all = np.array([
        _to_local_xyz([float(r["X"]), float(r["Y"]), float(r["Z"])], origin, basis)
        for r in valid_rows
    ], dtype=np.float64)
    Xs_all = local_all[:, 0]
    Ys_all = local_all[:, 1]
    Zs_all = local_all[:, 2]
    z_min, z_max = float(np.percentile(Zs_all, 2)), float(np.percentile(Zs_all, 98))
    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0

    def pad_lim(arr):
        lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
        pad = 0.1 * (hi - lo + 1e-6)
        return (lo - pad, hi + pad)

    def include_zero(lim):
        lo, hi = lim
        return (min(lo, 0.0), max(hi, 0.0))

    x_lim = include_zero(pad_lim(Xs_all))
    y_lim = pad_lim(Ys_all)
    z_lim = pad_lim(Zs_all)

    capL = open_video(left_video)
    fpsL = float(capL.get(cv2.CAP_PROP_FPS))
    fps = fpsL if fpsL > 0 else 30.0

    grouped = {}
    for row in out_rows:
        grouped.setdefault(int(row["frame_L"]), []).append(row)

    target_frames = sorted(grouped.keys())
    if max_frames > 0:
        target_frames = target_frames[:max_frames]
    if not target_frames:
        capL.release()
        print("[VIS] No frames selected. Skipping.")
        return

    ok, first_frame = capL.read()
    if not ok:
        capL.release()
        raise RuntimeError("[VIS] Could not read first frame from left video.")

    h, w_single = first_frame.shape[:2]

    base = Path(out_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.with_suffix("")
    left_path = Path(f"{stem}_left.mp4")
    iso_path = Path(f"{stem}_iso.mp4")
    top_path = Path(f"{stem}_topdown.mp4")

    if encoder == "auto":
        enc = "nvenc" if _has_ffmpeg() else "mp4v"
    else:
        enc = encoder
    if enc == "nvenc" and not _has_ffmpeg():
        print("[VIS] nvenc requested but ffmpeg not found; falling back to mp4v.")
        enc = "mp4v"
    print(f"[VIS] encoder={enc}")

    if workers <= 0:
        workers = min(os.cpu_count() or 8, len(target_frames))
    print(f"[VIS] workers={workers}, frames={len(target_frames)}")

    writer_left = _open_writer(left_path, fps, w_single, h, enc)
    per_frame_xyz = []

    current_idx = 0
    frame = first_frame
    target_ptr = 0
    next_target = int(target_frames[target_ptr])

    while True:
        if current_idx == next_target:
            rows = grouped[next_target]
            valid_here = [r for r in rows if int(r["valid_3d"]) == 1]

            vis_left = frame.copy()
            for r in rows:
                u = int(round(float(r["uL"])))
                v = int(round(float(r["vL"])))
                if u < 0 or v < 0 or u >= w_single or v >= h:
                    continue
                valid = int(r["valid_3d"]) == 1
                if valid:
                    disp = float(r.get("_disp_m", 0.0))
                    col = depth_to_bgr(disp, 0.0, disp_max)
                    cv2.circle(vis_left, (u, v), 10, col, -1)
                    cv2.circle(vis_left, (u, v), 11, (255, 255, 255), 2)
                    cv2.putText(vis_left, f"{disp:.3f}", (u + 12, v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                else:
                    cv2.circle(vis_left, (u, v), 4, (120, 120, 120), -1)

            cv2.putText(vis_left, "Left view (displacement-colored)", (24, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis_left,
                        f"frame={next_target}  nodes={len(rows)}  valid={len(valid_here)}",
                        (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            draw_value_colorbar(vis_left, 0.0, disp_max, x=w_single - 130, y=110, w=28, h=260, label="|d| m")
            _writer_write(writer_left, vis_left)

            if valid_here:
                local = np.array([
                    _to_local_xyz([float(r["X"]), float(r["Y"]), float(r["Z"])], origin, basis)
                    for r in valid_here
                ], dtype=np.float64)
                Xs = local[:, 0]
                Ys = local[:, 1]
                Zs = local[:, 2]
                Ds = np.array([float(r.get("_disp_m", 0.0)) for r in valid_here], dtype=np.float64)
            else:
                Xs = np.zeros(0)
                Ys = np.zeros(0)
                Zs = np.zeros(0)
                Ds = np.zeros(0)
            per_frame_xyz.append((Xs, Ys, Zs, Ds))

            target_ptr += 1
            if target_ptr >= len(target_frames):
                break
            next_target = int(target_frames[target_ptr])
            if target_ptr % max(1, len(target_frames) // 10) == 0:
                print(f"[VIS] left pass: {target_ptr}/{len(target_frames)}")

        ok, frame = capL.read()
        if not ok:
            break
        current_idx += 1
        if current_idx > int(target_frames[-1]):
            break

    _writer_close(writer_left)
    capL.release()
    print(f"[OK] Wrote {left_path}")

    print("[VIS] rendering iso panel (parallel)...")
    _render_panel_parallel("iso", target_frames, per_frame_xyz, fps, iso_path,
                           w_single, h, x_lim, y_lim, z_lim, z_min, z_max,
                           workers=workers, encoder=enc, disp_max=disp_max, iso_azim=iso_azim)
    print(f"[OK] Wrote {iso_path}")

    print("[VIS] rendering topdown panel (parallel)...")
    _render_panel_parallel("topdown", target_frames, per_frame_xyz, fps, top_path,
                           w_single, h, x_lim, y_lim, z_lim, z_min, z_max,
                           workers=workers, encoder=enc, disp_max=disp_max, iso_azim=iso_azim)
    print(f"[OK] Wrote {top_path}")


def visualize_existing_rows(args, out_rows):
    if not args.left_video:
        print("[VIS] --visualize requested but --left-video not provided. Skipping visualization.")
        return
    if args.viz_mode == "scene":
        create_3d_scene_visualization(
            out_rows=out_rows,
            left_video=args.left_video,
            out_path=args.viz_out,
            max_frames=args.viz_max_frames,
            trail_len=args.viz_trail,
            workers=args.workers,
            encoder=args.viz_encoder,
        )
    else:
        create_3d_visualization_video(
            out_rows=out_rows,
            left_video=args.left_video,
            out_path=args.viz_out,
            max_frames=args.viz_max_frames,
        )


def load_existing_triangulation_csv(path: str):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"--viz-only requested but CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ["frame_L", "obj_id", "uL", "vL", "X", "Y", "Z", "valid_3d"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required visualization columns: {missing}")
    rows = df.to_dict(orient="records")
    print(f"[VIS] Loaded existing triangulation CSV: {csv_path} rows={len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stereo", default="calibration/work/stereo.npz", help="Path to stereo.npz")
    ap.add_argument("--left", default="sam2/sam2/out/left/tracks_2d.csv", help="Left tracks_2d.csv")
    ap.add_argument("--right", default="sam2/sam2/out/right/tracks_2d.csv", help="Right tracks_2d.csv")
    ap.add_argument("--left-video", default="sam2/sam2/in/left.mp4", help="Left video path for per-run sync detection or visualization")
    ap.add_argument("--right-video", default="sam2/sam2/in/right.mp4", help="Right video path for per-run sync detection")
    ap.add_argument("--sync-mode", choices=["flash", "audio", "hybrid"], default=None,
                    help="Sync detection mode: audio peak, flash, or hybrid(visual)")
    ap.add_argument("--sync-scale", type=float, default=0.25, help="Downscale for sync detection")
    ap.add_argument("--sync-max-frames", type=int, default=-1, help="Frames to scan for sync (-1=all)")
    ap.add_argument("--sync-json", default="calibration/work/sync.json",
                    help="Calibration sync.json with trim_left/trim_right; use 'none' to disable")
    ap.add_argument("--left-drop", type=int, default=None, help="Manual left frame drop override")
    ap.add_argument("--right-drop", type=int, default=None, help="Manual right frame drop override")
    ap.add_argument("--out-sync", default=None, help="Optional path to save detected sync JSON")
    ap.add_argument("--out-csv", default="triangulation/out/triangulated_3d.csv")
    ap.add_argument("--out-summary", default="triangulation/out/summary.json")
    ap.add_argument("--visualize", action="store_true", help="Generate side-by-side original + CSV reconstruction video")
    ap.add_argument("--viz-only", action="store_true",
                    help="Skip triangulation and render visualization from an existing --out-csv")
    ap.add_argument("--viz-out", default="triangulation/out/triangulated_3d_viz.mp4", help="Output path for visualization video")
    ap.add_argument("--viz-max-frames", type=int, default=-1, help="Max rendered frames with tracked nodes (-1=all)")
    ap.add_argument("--viz-mode", choices=["scene", "sidebyside"], default="scene",
                    help="scene: depth-colored overlay + isometric + top-down. sidebyside: original legacy view.")
    ap.add_argument("--viz-trail", type=int, default=60, help="Recent-frames trail length in 3D panel (scene mode; currently unused)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers for iso/topdown rendering. 0=auto (cpu_count).")
    ap.add_argument("--viz-encoder", choices=["auto", "nvenc", "mp4v"], default="auto",
                    help="Video encoder. auto=nvenc if ffmpeg present, else mp4v.")
    ap.add_argument("--quality-min", type=float, default=0.0, help="Filter 2D points by quality >= this")
    ap.add_argument("--max-reproj", type=float, default=20.0, help="Reject if mean reproj err > this (px)")
    args = ap.parse_args()

    if args.viz_only:
        if not args.visualize:
            raise ValueError("--viz-only requires --visualize.")
        out_rows = load_existing_triangulation_csv(args.out_csv)
        visualize_existing_rows(args, out_rows)
        return

    stereo = np.load(args.stereo, allow_pickle=True)
    K1 = stereo["K1"].astype(np.float64)
    D1 = stereo["D1"].astype(np.float64).reshape(-1, 1)
    K2 = stereo["K2"].astype(np.float64)
    D2 = stereo["D2"].astype(np.float64).reshape(-1, 1)
    R = stereo["R"].astype(np.float64)
    T = stereo["T"].astype(np.float64).reshape(3, 1)

    image_size = tuple(int(x) for x in stereo["image_size"].tolist()) if "image_size" in stereo else None
    scale = float(stereo["scale"][0]) if "scale" in stereo else None
    calib_scale = float(scale) if scale is not None else 1.0

    print(f"[INFO] Loaded stereo: scale={scale}, image_size={image_size}")
    if not np.isclose(calib_scale, 1.0):
        print(
            f"[INFO] Scaling input 2D tracks by stereo scale={calib_scale:g} for triangulation; "
            "output/reprojection columns stay in original full-frame pixels."
        )

    sync_info = None
    explicit_sync_mode = args.sync_mode is not None
    sync_mode = args.sync_mode or "audio"
    sync_json = None if str(args.sync_json).strip().lower() in {"", "none", "null", "false"} else args.sync_json
    if explicit_sync_mode and args.sync_json == "calibration/work/sync.json":
        sync_json = None

    has_manual_sync = args.left_drop is not None or args.right_drop is not None
    if explicit_sync_mode and sync_json:
        raise ValueError("Use either --sync-mode for new sync or --sync-json for saved sync, not both.")
    if sync_json and has_manual_sync:
        raise ValueError("Use either --sync-json or --left-drop/--right-drop, not both.")

    if sync_json:
        left_drop, right_drop, sync_info = load_calibration_sync(sync_json)
        print(
            "[INFO] Sync mapping (calibration sync JSON): "
            f"trim_left={sync_info['trim_left']}, trim_right={sync_info['trim_right']} -> "
            f"left_drop={left_drop}, right_drop={right_drop}"
        )
        if args.out_sync:
            out_sync = Path(args.out_sync)
            out_sync.parent.mkdir(parents=True, exist_ok=True)
            out_sync.write_text(json.dumps(sync_info, indent=2))
            print(f"[OK] Wrote {out_sync}")
    elif has_manual_sync:
        left_drop = int(args.left_drop or 0)
        right_drop = int(args.right_drop or 0)
        sync_info = {
            "mode": "manual",
            "left_drop": left_drop,
            "right_drop": right_drop,
        }
        print(f"[INFO] Sync mapping (manual): left_drop={left_drop}, right_drop={right_drop}")
    else:
        if not args.left_video or not args.right_video:
            raise ValueError("Provide --left-video and --right-video for auto sync, or set --left-drop/--right-drop manually.")
        left_drop, right_drop, sync_info = detect_sync_from_videos(
            args.left_video,
            args.right_video,
            mode=sync_mode,
            scale=args.sync_scale,
            max_frames=args.sync_max_frames,
            preview_out_path=args.out_sync,
        )
        print(
            f"[INFO] Sync mapping ({sync_info['mode']}): "
            f"left_drop={left_drop}, right_drop={right_drop}, "
            f"left_event={sync_info['left_event_frame']}, right_event={sync_info['right_event_frame']}"
        )
        if args.out_sync:
            out_sync = Path(args.out_sync)
            out_sync.parent.mkdir(parents=True, exist_ok=True)
            out_sync.write_text(json.dumps(sync_info, indent=2))
            print(f"[OK] Wrote {out_sync}")

    L = pd.read_csv(args.left)
    Rdf = pd.read_csv(args.right)

    # Basic cleaning
    for df in (L, Rdf):
        # enforce expected columns
        for col in ["frame", "obj_id", "u", "v", "quality", "valid"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {df}")
        df["frame"] = df["frame"].astype(int)
        df["obj_id"] = df["obj_id"].astype(int)
        df["valid"] = df["valid"].astype(int)
        df["quality"] = df["quality"].astype(float)

    # Filter invalid / low-quality
    L = L[(L["valid"] == 1) & (L["quality"] >= args.quality_min)].copy()
    Rdf = Rdf[(Rdf["valid"] == 1) & (Rdf["quality"] >= args.quality_min)].copy()

    # Map to a common timeline so frame indices line up
    L["gframe"] = L["frame"] + left_drop
    Rdf["gframe"] = Rdf["frame"] + right_drop

    # Build dicts for fast matching
    L_keyed = {(int(r.gframe), int(r.obj_id)): (float(r.u), float(r.v), float(r.quality), int(r.frame))
               for r in L.itertuples(index=False)}
    R_keyed = {(int(r.gframe), int(r.obj_id)): (float(r.u), float(r.v), float(r.quality), int(r.frame))
               for r in Rdf.itertuples(index=False)}

    keys = sorted(set(L_keyed.keys()) & set(R_keyed.keys()))
    print(f"[INFO] Matched observations: {len(keys)}")

    # Projection matrices in normalized-coordinates space:
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
    P2 = np.hstack([R, T]).astype(np.float64)

    out_rows = []
    reproj_errs = []

    for (gframe, obj_id) in keys:
        u1, v1, q1, lframe = L_keyed[(gframe, obj_id)]
        u2, v2, q2, rframe = R_keyed[(gframe, obj_id)]

        pts1 = np.array([[u1 * calib_scale, v1 * calib_scale]], dtype=np.float64)
        pts2 = np.array([[u2 * calib_scale, v2 * calib_scale]], dtype=np.float64)

        # Undistort to normalized coords
        x1 = undist_norm_points(K1, D1, pts1)  # (1,2)
        x2 = undist_norm_points(K2, D2, pts2)  # (1,2)

        # Triangulate (expects 2xN)
        Xh = cv2.triangulatePoints(P1, P2, x1.T, x2.T)  # (4,1)
        X = (Xh[:3] / Xh[3]).reshape(3)  # in LEFT camera coords

        # Depth checks
        ZL = float(X[2])
        XR = (R @ X.reshape(3, 1) + T).reshape(3)
        ZR = float(XR[2])

        # Reproject
        p1_hat = project_points(K1, D1, X.reshape(1, 3))[0]
        p2_hat = project_points(K2, D2, XR.reshape(1, 3))[0]
        p1_hat_out = p1_hat / calib_scale
        p2_hat_out = p2_hat / calib_scale

        errL = float(np.linalg.norm(np.array([u1, v1]) - p1_hat_out))
        errR = float(np.linalg.norm(np.array([u2, v2]) - p2_hat_out))
        err = 0.5 * (errL + errR)

        ok = 1
        if ZL <= 0 or ZR <= 0:
            ok = 0
        if err > args.max_reproj:
            ok = 0

        if ok:
            reproj_errs.append(err)

        out_rows.append({
            "gframe": gframe,
            "frame_L": lframe,
            "frame_R": rframe,
            "obj_id": obj_id,
            "uL": float(u1),
            "vL": float(v1),
            "uR": float(u2),
            "vR": float(v2),
            "uL_hat": float(p1_hat_out[0]),
            "vL_hat": float(p1_hat_out[1]),
            "uR_hat": float(p2_hat_out[0]),
            "vR_hat": float(p2_hat_out[1]),
            "X": float(X[0]),
            "Y": float(X[1]),
            "Z": float(X[2]),
            "Z_right": ZR,
            "errL_px": errL,
            "errR_px": errR,
            "err_px": err,
            "qL": q1,
            "qR": q2,
            "valid_3d": ok,
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")

    # Summary
    valid_count = sum(1 for r in out_rows if r["valid_3d"] == 1)
    total_count = len(out_rows)
    reproj_errs = np.array(reproj_errs, dtype=np.float64) if reproj_errs else np.array([], dtype=np.float64)

    summary = {
        "total_matched_obs": total_count,
        "valid_3d_obs": valid_count,
        "valid_pct": (100.0 * valid_count / max(total_count, 1)),
        "sync": sync_info,
        "quality_min": args.quality_min,
        "max_reproj_px": args.max_reproj,
        "stereo_scale": scale,
        "stereo_image_size": list(image_size) if image_size else None,
        "input_track_coordinate_space": "original_full_frame_pixels",
        "reprojection_error_coordinate_space": "original_full_frame_pixels",
        "reproj_err_px": {
            "mean": float(reproj_errs.mean()) if reproj_errs.size else None,
            "median": float(np.median(reproj_errs)) if reproj_errs.size else None,
            "p95": float(np.percentile(reproj_errs, 95)) if reproj_errs.size else None,
        },
    }

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"[OK] Wrote {out_summary}")

    if args.visualize:
        visualize_existing_rows(args, out_rows)


if __name__ == "__main__":
    main()
