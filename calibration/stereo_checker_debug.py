import argparse
import json
import os
import queue
import shutil
import subprocess
import tempfile
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Manager
from typing import List, Tuple, Optional

import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


_CUDA_STATUS: Optional[bool] = None
_CUDA_FALLBACK_WARNED = False

AUDIO_TOPK_DEFAULT = 5
AUDIO_MIN_SEPARATION_FRAMES_DEFAULT = 20
DEFAULT_LEFT_VIDEO = "in/left.mp4"
DEFAULT_RIGHT_VIDEO = "in/right.mp4"
DEFAULT_SYNC_JSON = "work/sync.json"
DEFAULT_STATS_DIR = "work/stats"
DEFAULT_MONO_NPZ = "work/mono.npz"
DEFAULT_STEREO_NPZ = "work/stereo.npz"
DEFAULT_RECTIFY_DIR = "work/rectify"


class RichProgressBar:
    def __init__(self, total=None, desc="", unit="it", disable=False, **_kwargs):
        self._total = total
        self.disable = bool(disable)
        self.desc = desc
        self.unit = unit
        self.progress = None
        self.task = None
        if not self.disable:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self.progress.start()
            self.task = self.progress.add_task(desc, total=total)

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        self._total = value
        if self.progress is not None and self.task is not None:
            self.progress.update(self.task, total=value)

    def update(self, n=1):
        if self.progress is not None and self.task is not None:
            self.progress.update(self.task, advance=int(n))

    def refresh(self):
        if self.progress is not None:
            self.progress.refresh()

    def write(self, msg):
        if self.progress is not None:
            self.progress.console.print(msg)
        else:
            print(msg)

    def close(self):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.task = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def progress_bar(*args, **kwargs):
    return RichProgressBar(*args, **kwargs)


# -----------------------------
# UI helpers
# -----------------------------
def fit_to_screen(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s

def show(win, img, wait=1, max_w=1280, max_h=720):
    vis, _ = fit_to_screen(img, max_w=max_w, max_h=max_h)
    cv2.imshow(win, vis)
    return cv2.waitKey(wait) & 0xFF

def ensure_dir(p: str):
    if not p:
        return
    p_str = str(p)
    if not os.path.exists(p_str):
        os.makedirs(p_str, exist_ok=True)

def write_json(path: str, payload: dict):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def validate_args(args):
    if hasattr(args, "scale") and args.scale <= 0:
        raise ValueError("--scale must be > 0")
    if hasattr(args, "step") and args.step < 1:
        raise ValueError("--step must be >= 1")
    if hasattr(args, "cols") and args.cols < 2:
        raise ValueError("--cols must be >= 2")
    if hasattr(args, "rows") and args.rows < 2:
        raise ValueError("--rows must be >= 2")
    if hasattr(args, "max_scan") and args.max_scan < 1:
        raise ValueError("--max-scan must be >= 1")
    if hasattr(args, "max_frames") and args.max_frames == 0:
        raise ValueError("--max-frames cannot be 0 (use -1 for all frames)")
    if hasattr(args, "max_pairs") and args.max_pairs < 1:
        raise ValueError("--max-pairs must be >= 1")
    if hasattr(args, "workers") and args.workers < 1:
        raise ValueError("--workers must be >= 1")

def downscale(frame, scale: float):
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def cuda_available(use_cuda: bool) -> bool:
    global _CUDA_STATUS
    if not use_cuda:
        return False

    if _CUDA_STATUS is not None:
        return _CUDA_STATUS

    if not hasattr(cv2, "cuda") or not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
        print("[CUDA] OpenCV CUDA API not found in this build. Falling back to CPU.")
        _CUDA_STATUS = False
        return _CUDA_STATUS

    try:
        count = int(cv2.cuda.getCudaEnabledDeviceCount())
        if count <= 0:
            print("[CUDA] No CUDA devices found. Falling back to CPU.")
            _CUDA_STATUS = False
            return _CUDA_STATUS
        cv2.cuda.setDevice(0)
        print("[CUDA] Enabled: using device 0 for supported operations.")
        _CUDA_STATUS = True
    except Exception as e:
        print(f"[CUDA] Initialization failed ({e}). Falling back to CPU.")
        _CUDA_STATUS = False

    return _CUDA_STATUS

def warn_cuda_fallback(msg: str):
    global _CUDA_FALLBACK_WARNED
    if not _CUDA_FALLBACK_WARNED:
        print(msg)
        _CUDA_FALLBACK_WARNED = True

def prepare_frame_and_gray(frame: np.ndarray, scale: float):
    # Checkerboard detection runs on CPU, so keep preprocess on CPU to avoid GPU transfer overhead.
    frame_ds = downscale(frame, scale)
    gray = cv2.cvtColor(frame_ds, cv2.COLOR_BGR2GRAY)
    return frame_ds, gray

def remap_pair_with_backend(fL: np.ndarray, fR: np.ndarray,
                            map1x: np.ndarray, map1y: np.ndarray,
                            map2x: np.ndarray, map2y: np.ndarray,
                            use_cuda: bool):
    if cuda_available(use_cuda):
        try:
            gpuL = cv2.cuda_GpuMat(); gpuR = cv2.cuda_GpuMat()
            gpuL.upload(fL); gpuR.upload(fR)

            map1x_gpu = cv2.cuda_GpuMat(); map1y_gpu = cv2.cuda_GpuMat()
            map2x_gpu = cv2.cuda_GpuMat(); map2y_gpu = cv2.cuda_GpuMat()
            map1x_gpu.upload(map1x); map1y_gpu.upload(map1y)
            map2x_gpu.upload(map2x); map2y_gpu.upload(map2y)

            rectL = cv2.cuda.remap(gpuL, map1x_gpu, map1y_gpu, cv2.INTER_LINEAR).download()
            rectR = cv2.cuda.remap(gpuR, map2x_gpu, map2y_gpu, cv2.INTER_LINEAR).download()
            return rectL, rectR
        except Exception as e:
            warn_cuda_fallback(f"[CUDA] GPU remap failed ({e}). Using CPU path.")

    rectL = cv2.remap(fL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(fR, map2x, map2y, cv2.INTER_LINEAR)
    return rectL, rectR


# -----------------------------
# Video helpers
# -----------------------------
def open_video(path: str):
    requested = Path(path)
    candidates = [requested]
    suffix = requested.suffix
    if suffix:
        candidates.extend([
            requested.with_suffix(suffix.lower()),
            requested.with_suffix(suffix.upper()),
        ])
    # Preserve order while removing duplicates.
    candidates = list(dict.fromkeys(candidates))

    for candidate in candidates:
        cap = cv2.VideoCapture(str(candidate))
        if cap.isOpened():
            return cap
        cap.release()

    path_hint = ", ".join(str(p) for p in candidates)
    cwd = os.getcwd()
    raise RuntimeError(f"Could not open video: {path} (cwd={cwd}; tried: {path_hint})")

def read_frame_at(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return ok, frame

def read_frame_progressive(cap: cv2.VideoCapture, target_idx: int, next_idx: Optional[int]):
    """Read target frame by advancing with grab when possible, falling back to seek as needed.

    next_idx tracks the next frame index expected from the decoder after the previous read.
    """
    target_idx = int(target_idx)

    if next_idx is None or target_idx < int(next_idx):
        ok, frame = read_frame_at(cap, target_idx)
        return ok, frame, ((target_idx + 1) if ok else next_idx)

    skip_count = target_idx - int(next_idx)
    for _ in range(max(0, skip_count)):
        if not cap.grab():
            ok, frame = read_frame_at(cap, target_idx)
            return ok, frame, ((target_idx + 1) if ok else next_idx)

    ok, frame = cap.read()
    if ok:
        return ok, frame, (target_idx + 1)

    ok2, frame2 = read_frame_at(cap, target_idx)
    return ok2, frame2, ((target_idx + 1) if ok2 else next_idx)

def read_frame_sequential(video_path: str, idx: int):
    """Fallback for codecs that don't seek reliably (e.g., some HEVC files)."""
    cap = open_video(video_path)
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ok:
        cap.release()
        return False, None

    frame = None
    for _ in range(max(0, idx) + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return False, None
    cap.release()
    return True, frame

def robust_read_frame(video_path: str, idx: int, window: int = 6):
    """Try random seek first, then nearby frames, then sequential decode fallback."""
    cap = open_video(video_path)
    try:
        ok, frame = read_frame_at(cap, idx)
        if ok and frame is not None:
            return True, frame, idx

        for delta in range(1, window + 1):
            for candidate in (idx - delta, idx + delta):
                if candidate < 0:
                    continue
                ok, frame = read_frame_at(cap, candidate)
                if ok and frame is not None:
                    return True, frame, candidate
    finally:
        cap.release()

    ok, frame = read_frame_sequential(video_path, idx)
    if ok and frame is not None:
        return True, frame, idx

    for delta in range(1, window + 1):
        for candidate in (idx - delta, idx + delta):
            if candidate < 0:
                continue
            ok, frame = read_frame_sequential(video_path, candidate)
            if ok and frame is not None:
                return True, frame, candidate

    return False, None, idx

def get_video_info(cap: cv2.VideoCapture):
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return n, fps, (w, h)

def stepped_iteration_count(total_frames: int, start_idx: int, step: int, max_count: Optional[int] = None) -> int:
    """Return effective loop iterations for idx=start_idx; idx<total_frames; idx+=step."""
    if step <= 0:
        return 0
    if total_frames <= start_idx:
        return 0
    possible = (int(total_frames) - int(start_idx) + int(step) - 1) // int(step)
    if max_count is None or max_count < 0:
        return int(possible)
    return int(min(int(max_count), int(possible)))

def adaptive_halving_steps(base_step: int, levels: int = 4) -> List[int]:
    s = max(1, int(base_step))
    out: List[int] = []
    for _ in range(max(1, int(levels))):
        if s not in out:
            out.append(s)
        if s == 1:
            break
        s = max(1, s // 2)
    return out

def build_scan_indices(intervals: List[Tuple[int, int]],
                       step: int,
                       min_idx: int,
                       max_idx: int) -> List[int]:
    idxs: List[int] = []
    step = max(1, int(step))
    min_idx = int(min_idx)
    max_idx = int(max_idx)
    for a, b in intervals:
        lo = max(min_idx, int(a))
        hi = min(max_idx, int(b))
        if hi < lo:
            continue
        i = lo
        while i <= hi:
            idxs.append(int(i))
            i += step
    if not idxs:
        return []
    return sorted(set(idxs))

def build_roi_intervals(found_indices: List[int],
                        margin: int,
                        min_idx: int,
                        max_idx: int) -> List[Tuple[int, int]]:
    if not found_indices:
        return []

    margin = max(1, int(margin))
    min_idx = int(min_idx)
    max_idx = int(max_idx)

    spans: List[Tuple[int, int]] = []
    for f in sorted(set(int(x) for x in found_indices)):
        lo = max(min_idx, f - margin)
        hi = min(max_idx, f + margin)
        if hi >= lo:
            spans.append((lo, hi))

    if not spans:
        return []

    merged: List[Tuple[int, int]] = [spans[0]]
    for lo, hi in spans[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + 1:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged

def get_video_fps_and_frames(video_path: str):
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read FPS from video: {video_path}")
    return fps, total

def extract_audio_envelope(video_path: str, max_frames: int, sample_rate: int = 16000):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg for --sync-mode audio.")

    fps, total_frames = get_video_fps_and_frames(video_path)
    max_scan_frames = total_frames if max_frames < 0 else min(total_frames, max_frames)
    max_sec = max_scan_frames / fps

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-t", f"{max_sec:.6f}",
            wav_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed for {video_path}: {proc.stderr.strip()}")

        with wave.open(wav_path, "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            fr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if nch != 1:
            raise RuntimeError(f"Expected mono WAV after extraction, got {nch} channels.")
        if sw != 2:
            raise RuntimeError(f"Expected 16-bit PCM WAV, got sample width={sw} bytes.")
        if fr <= 0:
            raise RuntimeError("Invalid extracted WAV sample rate.")

        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if signal.size == 0:
            raise RuntimeError(f"No audio samples found in video: {video_path}")

        env = np.abs(signal)
        win = max(3, int(0.01 * fr))
        kernel = np.ones(win, dtype=np.float32) / float(win)
        env_smooth = np.convolve(env, kernel, mode="same")

        return {
            "video_fps": float(fps),
            "audio_rate": int(fr),
            "max_scan_frames": int(max_scan_frames),
            "env": env_smooth.astype(np.float32),
        }
    finally:
        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass

def find_audio_peak_candidates(env_smooth: np.ndarray,
                               audio_rate: int,
                               video_fps: float,
                               max_scan_frames: int,
                               top_k: int = 5,
                               min_separation_frames: int = 20):
    if env_smooth.size == 0:
        return []

    if env_smooth.size < 3:
        j = int(np.argmax(env_smooth))
        frame_idx = int(np.clip(round((j / float(audio_rate)) * video_fps), 0, max_scan_frames - 1))
        return [{
            "sample_idx": j,
            "frame_idx": frame_idx,
            "score": float(env_smooth[j]),
            "time_sec": j / float(audio_rate),
        }]

    local_max = np.where((env_smooth[1:-1] > env_smooth[:-2]) & (env_smooth[1:-1] >= env_smooth[2:]))[0] + 1
    if local_max.size == 0:
        local_max = np.arange(env_smooth.size)

    scores = env_smooth[local_max]
    order = np.argsort(scores)[::-1]

    min_sep_samples = max(1, int(round((min_separation_frames / max(video_fps, 1e-6)) * audio_rate)))

    selected = []
    for oi in order:
        j = int(local_max[oi])
        if all(abs(j - s["sample_idx"]) >= min_sep_samples for s in selected):
            frame_idx = int(np.clip(round((j / float(audio_rate)) * video_fps), 0, max_scan_frames - 1))
            selected.append({
                "sample_idx": j,
                "frame_idx": frame_idx,
                "score": float(env_smooth[j]),
                "time_sec": j / float(audio_rate),
            })
            if len(selected) >= top_k:
                break

    selected.sort(key=lambda d: d["sample_idx"])
    return selected

def render_audio_peak_preview(left_env: np.ndarray,
                              right_env: np.ndarray,
                              left_candidates: List[dict],
                              right_candidates: List[dict],
                              out_png: str):
    h, w = 900, 1600
    img = np.full((h, w, 3), 20, dtype=np.uint8)

    def draw_track(y0, y1, env, candidates, title, color):
        cv2.rectangle(img, (50, y0), (w - 50, y1), (70, 70, 70), 1)
        cv2.putText(img, title, (60, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

        if env.size == 0:
            return

        xs = np.linspace(0, env.size - 1, max(2, w - 100)).astype(np.int32)
        ys = env[xs]
        vmax = float(np.max(ys)) if ys.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        ys_norm = ys / vmax

        pts = []
        for i, yn in enumerate(ys_norm):
            x = 50 + i
            y = int(round(y1 - 20 - yn * ((y1 - y0) - 60)))
            pts.append((x, y))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 1)

        for rank, c in enumerate(candidates, start=1):
            x = int(round(50 + (c["sample_idx"] / max(env.size - 1, 1)) * (w - 100)))
            cv2.line(img, (x, y0 + 40), (x, y1 - 20), (0, 255, 255), 1)
            label = f"#{rank} f={c['frame_idx']}"
            cv2.putText(img, label, (x + 4, y0 + 60 + 22 * ((rank - 1) % 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    draw_track(40, 430, left_env, left_candidates, "LEFT audio envelope", (255, 140, 100))
    draw_track(470, 860, right_env, right_candidates, "RIGHT audio envelope", (100, 180, 255))

    cv2.putText(img, "Peak labels are selectable candidates", (50, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    cv2.imwrite(str(out_png), img)
    return img

def prompt_audio_choice(side: str, candidates: List[dict], default_rank: int = 1) -> int:
    print(f"[SYNC] {side} audio peak candidates:")
    for i, c in enumerate(candidates, start=1):
        print(f"  {i}) frame={c['frame_idx']}  t={c['time_sec']:.3f}s  score={c['score']:.2f}")

    while True:
        raw = input(f"[SYNC] Choose {side} peak index [1-{len(candidates)}] (Enter={default_rank}): ").strip()
        if raw == "":
            return max(1, min(default_rank, len(candidates))) - 1
        try:
            v = int(raw)
            if 1 <= v <= len(candidates):
                return v - 1
        except Exception:
            pass
        print("[SYNC] Invalid selection. Try again.")

def prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        raw = input(f"{question} {suffix}: ").strip().lower()
        if raw == "":
            return bool(default_yes)
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("[SYNC] Please answer y or n.")


# -----------------------------
# Sync by flash (same logic as your script)
# -----------------------------
def brightness_curve(video_path: str, scale: float, step: int, max_frames: int, use_cuda: bool = False):
    cap = open_video(video_path)
    total, fps, _ = get_video_info(cap)
    max_scan = min(total, max_frames if max_frames > 0 else total)
    effective_total = stepped_iteration_count(max_scan, 0, step)

    means, idxs = [], []
    pbar = progress_bar(total=effective_total, desc="[Brightness scan]", unit="frame")
    i = 0
    next_idx = None
    while i < max_scan:
        ok, frame, next_idx = read_frame_progressive(cap, i, next_idx)
        if not ok:
            break
        _, gray = prepare_frame_and_gray(frame, scale)
        means.append(float(np.mean(gray)))
        idxs.append(i)
        i += step
        pbar.update(1)
    pbar.close()

    cap.release()
    return np.array(idxs, np.int32), np.array(means, np.float32), fps

def pick_flash_frame(idxs: np.ndarray, means: np.ndarray):
    if len(means) < 5:
        return int(idxs[int(np.argmax(means))])
    diff = np.diff(means)
    j_jump = int(np.argmax(diff))
    frame_jump = int(idxs[j_jump + 1])
    j_max = int(np.argmax(means))
    frame_max = int(idxs[j_max])
    if diff[j_jump] < 5.0:
        return frame_max
    return frame_jump

def save_sync_preview(left_path, right_path, left_idx, right_idx, out_png, scale):
    okL, fL, used_left_idx = robust_read_frame(left_path, left_idx)
    okR, fR, used_right_idx = robust_read_frame(right_path, right_idx)
    if not okL or not okR:
        raise RuntimeError("Could not read sync frames for preview.")

    fL = downscale(fL, scale)
    fR = downscale(fR, scale)
    h = max(fL.shape[0], fR.shape[0])

    def pad_to_h(img, h):
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

    fL = pad_to_h(fL, h)
    fR = pad_to_h(fR, h)

    both = np.hstack([fL, fR])
    cv2.putText(both, f"L frame {left_idx} (used {used_left_idx})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(both, f"R frame {right_idx} (used {used_right_idx})", (fL.shape[1]+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imwrite(str(out_png), both)
    return both

def canonicalize_corners(corners, pattern):
    cols, rows = pattern
    C = corners.reshape(rows, cols, 2).copy()

    # If first row is decreasing in x, flip horizontally
    if C[0,0,0] > C[0,-1,0]:
        C = C[:, ::-1, :]

    # If first column is decreasing in y, flip vertically
    if C[0,0,1] > C[-1,0,1]:
        C = C[::-1, :, :]

    return C.reshape(-1,1,2)

# -----------------------------
# Checkerboard detection
# -----------------------------
def detect_checkerboard(gray: np.ndarray, pattern: Tuple[int,int]) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Returns (found, corners) where corners is (N,1,2) float32
    Uses ChessboardCornersSB if available (more robust).
    """
    cols, rows = pattern
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

    if not found or corners is None:
        return False, None

    # Subpixel refine (important)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (7,7), (-1,-1), term)
    corners = canonicalize_corners(corners, pattern)
    return True, corners.astype(np.float32)


@dataclass
class Sample:
    frame_idx: int
    img: np.ndarray  # (N,2)
    obj: np.ndarray  # (N,3)


def make_object_points(pattern: Tuple[int,int], square_m: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((rows*cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] = grid * square_m
    return obj


# -----------------------------
# Collect samples
# -----------------------------
def collect_samples_checker(video_path: str,
                            sync_start: int,
                            pattern: Tuple[int,int],
                            square_m: float,
                            scale: float,
                            step: int,
                            max_scan: int,
                            debug_every: int,
                            label: str,
                            out_dir: str,
                            use_cuda: bool = False,
                            frame_indices: Optional[List[int]] = None,
                            no_adaptive: bool = False,
                            progress_position: int = 0,
                            show_progress: bool = True,
                            progress_queue=None,
                            quiet: bool = False):
    ensure_dir(out_dir)
    cap = open_video(video_path)
    total, fps, (W, H) = get_video_info(cap)
    image_size = (int(W*scale), int(H*scale))

    obj_template = make_object_points(pattern, square_m)

    scanned = 0
    found_count = 0
    used: List[Sample] = []

    if frame_indices is not None:
        # Keep first occurrence order, clip to valid range, and optionally cap by max_scan.
        seen = set()
        idx_list = []
        for i in frame_indices:
            ii = int(i)
            if ii < 0 or ii >= total or ii in seen:
                continue
            seen.add(ii)
            idx_list.append(ii)
        if max_scan > 0:
            idx_list = idx_list[:max_scan]

        if progress_queue is not None:
            progress_queue.put(("total", len(idx_list)))
        pbar = progress_bar(
            total=len(idx_list),
            desc=f"[{label} detection]",
            unit="frame",
            position=progress_position,
            disable=not show_progress,
        )
        next_idx = None
        for idx in idx_list:
            ok, frame, next_idx = read_frame_progressive(cap, idx, next_idx)
            pbar.update(1)
            if progress_queue is not None:
                progress_queue.put(("update", 1))
            if not ok:
                continue

            frame, gray = prepare_frame_and_gray(frame, scale)
            scanned += 1
            found, corners = detect_checkerboard(gray, pattern)

            if found:
                found_count += 1
                pts = corners.reshape(-1, 2)
                used.append(Sample(frame_idx=idx, img=pts, obj=obj_template.copy()))

            if debug_every > 0 and (scanned % debug_every == 0):
                vis = frame.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(vis, pattern, corners, found)
                    cv2.putText(vis, f"{label} FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(vis, f"{label} NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                cv2.putText(vis, f"scan={scanned} used={len(used)} idx={idx}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imwrite(os.path.join(out_dir, f"{label}_debug_{scanned:05d}.png"), vis)
        pbar.close()
        scan_source = "provided_indices"
    else:
        scan_lo = max(0, int(sync_start))
        scan_hi = total - 1
        intervals = [(scan_lo, scan_hi)]
        steps = [int(step)] if no_adaptive else adaptive_halving_steps(step, levels=4)
        visited = set()
        next_idx = None
        for pass_i, pass_step in enumerate(steps, start=1):
            remaining = max_scan - scanned
            if remaining <= 0 or scan_hi < scan_lo or len(intervals) == 0:
                break

            idx_list = build_scan_indices(intervals, pass_step, scan_lo, scan_hi)
            idx_list = [i for i in idx_list if i not in visited]
            if len(idx_list) == 0:
                continue
            if len(idx_list) > remaining:
                idx_list = idx_list[:remaining]

            if not quiet:
                print(
                    f"[{label}] {'simple' if no_adaptive else 'adaptive'} pass {pass_i}/{len(steps)}: "
                    f"step={pass_step}, intervals={len(intervals)}, frames={len(idx_list)}"
                )
            if progress_queue is not None:
                progress_queue.put(("total", len(idx_list)))
            pbar = progress_bar(
                total=len(idx_list),
                desc=f"[{label} detection s={pass_step}]",
                unit="frame",
                position=progress_position,
                disable=not show_progress,
            )

            for idx in idx_list:
                ok, frame, next_idx = read_frame_progressive(cap, idx, next_idx)
                visited.add(int(idx))
                pbar.update(1)
                if progress_queue is not None:
                    progress_queue.put(("update", 1))
                if not ok:
                    continue

                frame, gray = prepare_frame_and_gray(frame, scale)
                scanned += 1
                found, corners = detect_checkerboard(gray, pattern)

                if found:
                    found_count += 1
                    pts = corners.reshape(-1, 2)
                    used.append(Sample(frame_idx=idx, img=pts, obj=obj_template.copy()))

                if debug_every > 0 and (scanned % debug_every == 0):
                    vis = frame.copy()
                    if found and corners is not None:
                        cv2.drawChessboardCorners(vis, pattern, corners, found)
                        cv2.putText(vis, f"{label} FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(vis, f"{label} NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    cv2.putText(vis, f"scan={scanned} used={len(used)} idx={idx}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    cv2.imwrite(os.path.join(out_dir, f"{label}_debug_{scanned:05d}.png"), vis)

            pbar.close()

            if pass_i < len(steps) and len(used) > 0:
                found_idx = [s.frame_idx for s in used]
                intervals = build_roi_intervals(found_idx, margin=pass_step * 2, min_idx=scan_lo, max_idx=scan_hi)

            if scanned >= max_scan:
                break

        scan_source = "adaptive_halving_scan"

    cap.release()

    stats = {
        "video": video_path,
        "sync_start": sync_start,
        "scan_source": scan_source,
        "scale": scale,
        "step": step,
        "max_scan": max_scan,
        "scanned": scanned,
        "found": found_count,
        "found_pct": 0.0 if scanned == 0 else 100.0 * found_count / scanned,
        "used_frames": len(used),
        "pattern_inner_corners": list(pattern),
        "square_m": square_m,
        "image_size_scaled": list(image_size),
    }
    return used, stats, image_size


# -----------------------------
# Calibration
# -----------------------------
def reprojection_stats(samples: List[Sample], K, dist, rvecs, tvecs):
    errs = []
    for i, s in enumerate(samples):
        img_proj, _ = cv2.projectPoints(s.obj, rvecs[i], tvecs[i], K, dist)
        img_proj = img_proj.reshape(-1, 2)
        e = np.linalg.norm(img_proj - s.img, axis=1)
        errs.append(float(np.mean(e)))
    errs = np.array(errs, np.float32)
    return {
        "mean_px": float(errs.mean()) if len(errs) else 0.0,
        "median_px": float(np.median(errs)) if len(errs) else 0.0,
        "max_px": float(errs.max()) if len(errs) else 0.0,
    }

def calibrate_mono(samples: List[Sample], image_size: Tuple[int,int]):
    objpoints = [s.obj.reshape(-1,1,3).astype(np.float32) for s in samples]
    imgpoints = [s.img.reshape(-1,1,2).astype(np.float32) for s in samples]

    # Good default for GoPro: allow more distortion terms.
    flags = cv2.CALIB_RATIONAL_MODEL
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
        flags=flags, criteria=criteria
    )
    return rms, K, dist, rvecs, tvecs

# def stereo_calibrate(samplesL: List[Sample], samplesR: List[Sample], image_size, K1, D1, K2, D2):
#     assert len(samplesL) == len(samplesR)
#     objpoints, img1, img2 = [], [], []
#     for a, b in zip(samplesL, samplesR):
#         objpoints.append(a.obj.reshape(-1,1,3).astype(np.float32))
#         img1.append(a.img.reshape(-1,1,2).astype(np.float32))
#         img2.append(b.img.reshape(-1,1,2).astype(np.float32))

#     flags = cv2.CALIB_FIX_INTRINSIC
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-7)

#     rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
#         objpoints, img1, img2,
#         K1, D1, K2, D2,
#         image_size,
#         flags=flags, criteria=criteria
#     )
#     return rms, R, T, E, F

def stereo_calibrate(samplesL, samplesR, image_size, K1, D1, K2, D2):
    assert len(samplesL) == len(samplesR)
    objpoints, img1, img2 = [], [], []
    for a, b in zip(samplesL, samplesR):
        objpoints.append(a.obj.reshape(-1,1,3).astype(np.float32))
        img1.append(a.img.reshape(-1,1,2).astype(np.float32))
        img2.append(b.img.reshape(-1,1,2).astype(np.float32))

    # ADD THE RATIONAL MODEL FLAG HERE TO MATCH YOUR MONO SETTINGS
    flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-7)

    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, img1, img2,
        K1, D1, K2, D2,
        image_size,
        flags=flags, criteria=criteria
    )
    return rms, R, T, E, F

def rectify_pair(K1, D1, K2, D2, R, T, image_size, alpha: float):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    return (R1, R2, P1, P2, Q), (map1x, map1y, map2x, map2y)


# -----------------------------
# Commands
# -----------------------------
def cmd_sync(args):
    ensure_dir(os.path.dirname(args.out) or ".")
    if args.sync_mode == "audio":
        print("\n[SYNC] Scanning audio peaks...")
        left_audio = extract_audio_envelope(args.left, max_frames=args.max_frames, sample_rate=args.audio_sample_rate)
        right_audio = extract_audio_envelope(args.right, max_frames=args.max_frames, sample_rate=args.audio_sample_rate)

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
            raise RuntimeError("[SYNC] No audio peak candidates found. Try increasing --max-frames.")

        audio_preview_png = str(Path(args.out).with_suffix("")) + "_audio_peaks.png"
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
    else:
        print("\n[SYNC] Scanning brightness for flash...")
        idxL, meanL, _ = brightness_curve(args.left, args.scale, args.step, args.max_frames, use_cuda=args.use_cuda)
        idxR, meanR, _ = brightness_curve(args.right, args.scale, args.step, args.max_frames, use_cuda=args.use_cuda)
        fL = pick_flash_frame(idxL, meanL)
        fR = pick_flash_frame(idxR, meanR)
        reasonL, reasonR = "flash", "flash"

    offset = fL - fR
    if offset > 0:
        startL, startR = offset, 0
    else:
        startL, startR = 0, -offset

    print(f"[SYNC] left_flash={fL}, right_flash={fR}, offset(left-right)={offset}")
    print(f"[SYNC] Proposed trim: drop {startL} frames from LEFT, drop {startR} frames from RIGHT")

    preview_png = str(Path(args.out).with_suffix("")) + "_preview.png"
    try:
        both = save_sync_preview(args.left, args.right, fL, fR, preview_png, args.scale)
        print(f"[SYNC] Saved sync preview: {preview_png}")
        print("[SYNC] Close preview window to continue.")
        show("SYNC PREVIEW (flash frames)", both, wait=0)
    except Exception as e:
        print(f"[SYNC] WARNING: preview generation failed ({e}). Continuing without preview.")

    write_json(args.out, {
        "mode": str(args.sync_mode),
        "trim_left": int(startL),
        "trim_right": int(startR),
        "left_flash": int(fL),
        "right_flash": int(fR),
        "left_reason": str(reasonL),
        "right_reason": str(reasonR),
        "scale": float(args.scale),
    })
    print(f"[SYNC] Wrote: {args.out}")

def load_sync(path: str):
    with open(path, "r") as f:
        d = json.load(f)
    return int(d["trim_left"]), int(d["trim_right"])

def load_positive_indices(stats_json_path: str) -> List[int]:
    with open(stats_json_path, "r") as f:
        d = json.load(f)
    idxs = d.get("detected_frame_indices", None)
    if idxs is None:
        raise RuntimeError(
            f"[MONO] Missing 'detected_frame_indices' in {stats_json_path}. "
            "Re-run stats with this updated script."
        )
    return [int(i) for i in idxs]

def build_paired_indices_from_stats(trimL: int,
                                    trimR: int,
                                    left_indices: List[int],
                                    right_indices: List[int]) -> List[Tuple[int, int]]:
    """Build synchronized (idxL, idxR) pairs from per-camera positive detections."""
    left_g = {}
    right_g = {}

    for idx in left_indices:
        g = int(idx) - int(trimL)
        if g >= 0 and g not in left_g:
            left_g[g] = int(idx)

    for idx in right_indices:
        g = int(idx) - int(trimR)
        if g >= 0 and g not in right_g:
            right_g[g] = int(idx)

    common_g = sorted(set(left_g.keys()) & set(right_g.keys()))
    return [(left_g[g], right_g[g]) for g in common_g]


def collect_stats_worker(params: dict):
    samples, stats, image_size = collect_samples_checker(
        params["video_path"],
        params["trim"],
        params["pattern"],
        params["square_m"],
        scale=params["scale"],
        step=params["step"],
        max_scan=params["max_scan"],
        debug_every=params["debug_every"],
        label=params["label"],
        out_dir=params["out_dir"],
        use_cuda=params["use_cuda"],
        no_adaptive=params["no_adaptive"],
        show_progress=params["show_progress"],
        progress_queue=params["progress_queue"],
        quiet=params["quiet"],
    )
    stats["detected_frame_indices"] = [int(s.frame_idx) for s in samples]
    return stats, image_size


def chunk_indices(indices: List[int], n_chunks: int) -> List[List[int]]:
    if not indices:
        return []
    n_chunks = max(1, min(int(n_chunks), len(indices)))
    chunk_size = (len(indices) + n_chunks - 1) // n_chunks
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def detect_stats_chunk_worker(params: dict):
    video_path = params["video_path"]
    idx_list = [int(i) for i in params["idx_list"]]
    pattern = tuple(params["pattern"])
    scale = float(params["scale"])
    label = params["label"]
    debug_every = int(params["debug_every"])
    out_dir = params["out_dir"]
    progress_queue = params.get("progress_queue")

    cap = open_video(video_path)
    found_indices = []
    scanned = 0
    next_idx = None
    try:
        for local_i, idx in enumerate(idx_list, start=1):
            ok, frame, next_idx = read_frame_progressive(cap, idx, next_idx)
            if progress_queue is not None:
                progress_queue.put(("update", 1))
            if not ok:
                continue

            frame, gray = prepare_frame_and_gray(frame, scale)
            scanned += 1
            found, corners = detect_checkerboard(gray, pattern)
            if found:
                found_indices.append(int(idx))

            if debug_every > 0 and (local_i % debug_every == 0):
                vis = frame.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(vis, pattern, corners, found)
                    cv2.putText(vis, f"{label} FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(vis, f"{label} NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(vis, f"idx={idx}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imwrite(os.path.join(out_dir, f"{label}_debug_idx_{idx:08d}.png"), vis)
    finally:
        cap.release()

    return {
        "scanned": scanned,
        "found_indices": found_indices,
    }


def collect_sample_chunk_worker(params: dict):
    video_path = params["video_path"]
    idx_list = [int(i) for i in params["idx_list"]]
    pattern = tuple(params["pattern"])
    square_m = float(params["square_m"])
    scale = float(params["scale"])
    label = params["label"]
    debug_every = int(params["debug_every"])
    out_dir = params["out_dir"]
    progress_queue = params.get("progress_queue")

    cap = open_video(video_path)
    obj_template = make_object_points(pattern, square_m)
    samples = []
    scanned = 0
    next_idx = None
    try:
        for local_i, idx in enumerate(idx_list, start=1):
            ok, frame, next_idx = read_frame_progressive(cap, idx, next_idx)
            if progress_queue is not None:
                progress_queue.put(("update", 1))
            if not ok:
                continue

            frame, gray = prepare_frame_and_gray(frame, scale)
            scanned += 1
            found, corners = detect_checkerboard(gray, pattern)

            if found:
                pts = corners.reshape(-1, 2)
                samples.append(Sample(frame_idx=idx, img=pts, obj=obj_template.copy()))

            if debug_every > 0 and (local_i % debug_every == 0):
                vis = frame.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(vis, pattern, corners, found)
                    cv2.putText(vis, f"{label} FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(vis, f"{label} NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(vis, f"idx={idx}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imwrite(os.path.join(out_dir, f"{label}_debug_idx_{idx:08d}.png"), vis)
    finally:
        cap.release()

    return {
        "scanned": scanned,
        "samples": samples,
    }


def collect_stereo_pair_chunk_worker(params: dict):
    left_video = params["left_video"]
    right_video = params["right_video"]
    pair_list = [(int(a), int(b)) for a, b in params["pair_list"]]
    pattern = tuple(params["pattern"])
    square_m = float(params["square_m"])
    scale = float(params["scale"])
    debug_every = int(params["debug_every"])
    out_dir = params["out_dir"]
    progress_queue = params.get("progress_queue")

    capL = open_video(left_video)
    capR = open_video(right_video)
    obj_template = make_object_points(pattern, square_m)
    usedL = []
    usedR = []
    scanned = 0
    nextL = None
    nextR = None
    try:
        for local_i, (idxL, idxR) in enumerate(pair_list, start=1):
            okL, fL, nextL = read_frame_progressive(capL, idxL, nextL)
            okR, fR, nextR = read_frame_progressive(capR, idxR, nextR)
            if progress_queue is not None:
                progress_queue.put(("update", 1))
            if not okL or not okR:
                continue

            fL, gL = prepare_frame_and_gray(fL, scale)
            fR, gR = prepare_frame_and_gray(fR, scale)
            scanned += 1
            foundL, cL = detect_checkerboard(gL, pattern)
            foundR, cR = detect_checkerboard(gR, pattern)

            if foundL and foundR:
                usedL.append(Sample(idxL, cL.reshape(-1, 2), obj_template.copy()))
                usedR.append(Sample(idxR, cR.reshape(-1, 2), obj_template.copy()))

            if debug_every > 0 and (local_i % debug_every == 0):
                visL = fL.copy()
                visR = fR.copy()
                if foundL:
                    cv2.drawChessboardCorners(visL, pattern, cL, True)
                if foundR:
                    cv2.drawChessboardCorners(visR, pattern, cR, True)
                both = np.hstack([visL, visR])
                cv2.putText(both, f"idxL={idxL} idxR={idxR} paired={len(usedL)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.imwrite(os.path.join(out_dir, f"pair_idx_{idxL:08d}_{idxR:08d}.png"), both)
    finally:
        capL.release()
        capR.release()

    return {
        "scanned": scanned,
        "usedL": usedL,
        "usedR": usedR,
    }


def drain_progress_queue(progress_queue, pbar):
    while True:
        try:
            event, value = progress_queue.get_nowait()
        except queue.Empty:
            break
        if event == "total":
            pbar.total = int(pbar.total or 0) + int(value)
            pbar.refresh()
        elif event == "update":
            pbar.update(int(value))


def drain_progress_queue_to_task(progress_queue, progress, task_id, totals):
    while True:
        try:
            event, value = progress_queue.get_nowait()
        except queue.Empty:
            break
        if event == "total":
            totals[task_id] = int(totals.get(task_id) or 0) + int(value)
            progress.update(task_id, total=totals[task_id])
        elif event == "update":
            progress.update(task_id, advance=int(value))


def collect_stats_parallel_side(params: dict, workers: int, progress_queue=None, pbar=None):
    ensure_dir(params["out_dir"])
    cap = open_video(params["video_path"])
    try:
        total, _fps, (W, H) = get_video_info(cap)
    finally:
        cap.release()

    scale = float(params["scale"])
    image_size = (int(W * scale), int(H * scale))
    scan_lo = max(0, int(params["trim"]))
    scan_hi = total - 1
    intervals = [(scan_lo, scan_hi)]
    steps = [int(params["step"])] if params["no_adaptive"] else adaptive_halving_steps(params["step"], levels=4)
    visited = set()
    scanned = 0
    found_indices: List[int] = []
    max_scan = int(params["max_scan"])
    scan_source = "parallel_adaptive_halving_scan"

    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for pass_i, pass_step in enumerate(steps, start=1):
            remaining = max_scan - scanned
            if remaining <= 0 or scan_hi < scan_lo or len(intervals) == 0:
                break

            idx_list = build_scan_indices(intervals, pass_step, scan_lo, scan_hi)
            idx_list = [i for i in idx_list if i not in visited]
            if len(idx_list) == 0:
                continue
            if len(idx_list) > remaining:
                idx_list = idx_list[:remaining]

            msg = (
                f"[{params['label']}] {'simple' if params['no_adaptive'] else 'adaptive'} pass {pass_i}/{len(steps)}: "
                f"step={pass_step}, intervals={len(intervals)}, frames={len(idx_list)}, workers={workers}"
            )
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)
            if progress_queue is not None:
                progress_queue.put(("total", len(idx_list)))

            futures = []
            for chunk in chunk_indices(idx_list, workers):
                futures.append(executor.submit(detect_stats_chunk_worker, {
                    "video_path": params["video_path"],
                    "idx_list": chunk,
                    "pattern": params["pattern"],
                    "scale": params["scale"],
                    "label": params["label"],
                    "debug_every": params["debug_every"],
                    "out_dir": params["out_dir"],
                    "progress_queue": progress_queue,
                }))

            pass_found = []
            pending = set(futures)
            while pending:
                done = [future for future in pending if future.done()]
                if not done:
                    if progress_queue is not None and pbar is not None:
                        drain_progress_queue(progress_queue, pbar)
                    time.sleep(0.1)
                    continue
                for future in done:
                    pending.remove(future)
                    result = future.result()
                    scanned += int(result["scanned"])
                    pass_found.extend(int(i) for i in result["found_indices"])
                if progress_queue is not None and pbar is not None:
                    drain_progress_queue(progress_queue, pbar)

            visited.update(int(i) for i in idx_list)
            found_indices.extend(pass_found)

            if pass_i < len(steps) and found_indices:
                intervals = build_roi_intervals(found_indices, margin=pass_step * 2, min_idx=scan_lo, max_idx=scan_hi)

            if scanned >= max_scan:
                break

    found_indices = sorted(set(found_indices))
    stats = {
        "video": params["video_path"],
        "sync_start": int(params["trim"]),
        "scan_source": scan_source,
        "scale": scale,
        "step": int(params["step"]),
        "max_scan": max_scan,
        "scanned": int(scanned),
        "found": len(found_indices),
        "found_pct": 0.0 if scanned == 0 else 100.0 * len(found_indices) / scanned,
        "used_frames": len(found_indices),
        "pattern_inner_corners": list(params["pattern"]),
        "square_m": float(params["square_m"]),
        "image_size_scaled": list(image_size),
        "detected_frame_indices": found_indices,
    }
    return stats, image_size


def collect_samples_parallel_indices(params: dict, frame_indices: List[int], workers: int, progress_queue=None, pbar=None):
    ensure_dir(params["out_dir"])
    cap = open_video(params["video_path"])
    try:
        total, _fps, (W, H) = get_video_info(cap)
    finally:
        cap.release()

    seen = set()
    idx_list = []
    for i in frame_indices:
        ii = int(i)
        if ii < 0 or ii >= total or ii in seen:
            continue
        seen.add(ii)
        idx_list.append(ii)
    if params["max_scan"] > 0:
        idx_list = idx_list[:int(params["max_scan"])]

    if progress_queue is not None:
        progress_queue.put(("total", len(idx_list)))
    if pbar is not None:
        pbar.write(f"[{params['label']}] detecting {len(idx_list)} reused stats frames with {workers} workers")

    samples = []
    scanned = 0
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = []
        for chunk in chunk_indices(idx_list, workers):
            futures.append(executor.submit(collect_sample_chunk_worker, {
                "video_path": params["video_path"],
                "idx_list": chunk,
                "pattern": params["pattern"],
                "square_m": params["square_m"],
                "scale": params["scale"],
                "label": params["label"],
                "debug_every": params["debug_every"],
                "out_dir": params["out_dir"],
                "progress_queue": progress_queue,
            }))

        pending = set(futures)
        while pending:
            done = [future for future in pending if future.done()]
            if not done:
                if progress_queue is not None and pbar is not None:
                    drain_progress_queue(progress_queue, pbar)
                time.sleep(0.1)
                continue
            for future in done:
                pending.remove(future)
                result = future.result()
                scanned += int(result["scanned"])
                samples.extend(result["samples"])
            if progress_queue is not None and pbar is not None:
                drain_progress_queue(progress_queue, pbar)

    samples.sort(key=lambda s: int(s.frame_idx))
    image_size = (int(W * float(params["scale"])), int(H * float(params["scale"])))
    stats = {
        "video": params["video_path"],
        "sync_start": int(params["trim"]),
        "scan_source": "parallel_reused_stats_indices",
        "scale": float(params["scale"]),
        "step": int(params["step"]),
        "max_scan": int(params["max_scan"]),
        "scanned": int(scanned),
        "found": len(samples),
        "found_pct": 0.0 if scanned == 0 else 100.0 * len(samples) / scanned,
        "used_frames": len(samples),
        "pattern_inner_corners": list(params["pattern"]),
        "square_m": float(params["square_m"]),
        "image_size_scaled": list(image_size),
    }
    return samples, stats, image_size


def collect_stereo_pairs_parallel(args, paired_indices: List[Tuple[int, int]], pattern, square_m: float,
                                  workers: int, progress_queue=None, pbar=None):
    out_dir = os.path.join(os.path.dirname(args.out) or ".", "stereo_debug")
    ensure_dir(out_dir)
    pair_list = [(int(a), int(b)) for a, b in paired_indices]
    if args.max_scan > 0:
        pair_list = pair_list[:int(args.max_scan)]
    if progress_queue is not None:
        progress_queue.put(("total", len(pair_list)))
    if pbar is not None:
        pbar.write(f"[STEREO] detecting {len(pair_list)} reused stats pairs with {workers} workers")

    usedL = []
    usedR = []
    scanned = 0
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = []
        for chunk in chunk_indices(pair_list, workers):
            futures.append(executor.submit(collect_stereo_pair_chunk_worker, {
                "left_video": args.left,
                "right_video": args.right,
                "pair_list": chunk,
                "pattern": pattern,
                "square_m": square_m,
                "scale": args.scale,
                "debug_every": args.debug_every,
                "out_dir": out_dir,
                "progress_queue": progress_queue,
            }))

        pending = set(futures)
        while pending:
            done = [future for future in pending if future.done()]
            if not done:
                if progress_queue is not None and pbar is not None:
                    drain_progress_queue(progress_queue, pbar)
                time.sleep(0.1)
                continue
            for future in done:
                pending.remove(future)
                result = future.result()
                scanned += int(result["scanned"])
                usedL.extend(result["usedL"])
                usedR.extend(result["usedR"])
            if progress_queue is not None and pbar is not None:
                drain_progress_queue(progress_queue, pbar)

    pairs = sorted(zip(usedL, usedR), key=lambda pair: int(pair[0].frame_idx))
    usedL = [pair[0] for pair in pairs]
    usedR = [pair[1] for pair in pairs]
    return usedL, usedR, scanned


def compact_stats_for_print(stats: dict) -> dict:
    compact = dict(stats)
    indices = compact.get("detected_frame_indices")
    if isinstance(indices, list):
        compact["detected_frame_indices"] = {
            "count": len(indices),
            "first": indices[:10],
            "last": indices[-10:] if len(indices) > 10 else [],
        }
    return compact


def cmd_stats(args):
    ensure_dir(args.out)
    trimL, trimR = load_sync(args.sync)

    pattern = (args.cols, args.rows)
    square_m = args.square_mm / 1000.0

    print("\n[STATS] Checkerboard detection stats (no calibration).")
    jobs = [
        {
            "video_path": args.left,
            "trim": trimL,
            "pattern": pattern,
            "square_m": square_m,
            "scale": args.scale,
            "step": args.step,
            "max_scan": args.max_scan,
            "debug_every": args.debug_every,
            "label": "LEFT",
            "out_dir": os.path.join(args.out, "debug_left"),
            "use_cuda": args.use_cuda,
            "no_adaptive": args.no_adaptive,
            "show_progress": False,
            "progress_queue": None,
            "quiet": True,
        },
        {
            "video_path": args.right,
            "trim": trimR,
            "pattern": pattern,
            "square_m": square_m,
            "scale": args.scale,
            "step": args.step,
            "max_scan": args.max_scan,
            "debug_every": args.debug_every,
            "label": "RIGHT",
            "out_dir": os.path.join(args.out, "debug_right"),
            "use_cuda": args.use_cuda,
            "no_adaptive": args.no_adaptive,
            "show_progress": False,
            "progress_queue": None,
            "quiet": True,
        },
    ]
    print(f"[STATS] Scanning with {args.workers} worker process(es).")
    if args.workers == 1:
        sL, stL, sizeL = collect_samples_checker(
            args.left, trimL, pattern, square_m,
            scale=args.scale, step=args.step, max_scan=args.max_scan,
            debug_every=args.debug_every, label="LEFT",
            out_dir=os.path.join(args.out, "debug_left"),
            use_cuda=args.use_cuda,
            no_adaptive=args.no_adaptive,
        )
        sR, stR, sizeR = collect_samples_checker(
            args.right, trimR, pattern, square_m,
            scale=args.scale, step=args.step, max_scan=args.max_scan,
            debug_every=args.debug_every, label="RIGHT",
            out_dir=os.path.join(args.out, "debug_right"),
            use_cuda=args.use_cuda,
            no_adaptive=args.no_adaptive,
        )
        stL["detected_frame_indices"] = [int(s.frame_idx) for s in sL]
        stR["detected_frame_indices"] = [int(s.frame_idx) for s in sR]
    elif args.workers > 2:
        with Manager() as manager:
            progress_queue = manager.Queue()
            jobs[0]["progress_queue"] = progress_queue
            jobs[1]["progress_queue"] = progress_queue
            with progress_bar(desc="[Stats LEFT detection]", unit="frame") as pbar:
                stL, sizeL = collect_stats_parallel_side(jobs[0], args.workers, progress_queue, pbar)
                drain_progress_queue(progress_queue, pbar)
            with progress_bar(desc="[Stats RIGHT detection]", unit="frame") as pbar:
                stR, sizeR = collect_stats_parallel_side(jobs[1], args.workers, progress_queue, pbar)
                drain_progress_queue(progress_queue, pbar)
    else:
        with Manager() as manager:
            left_progress_queue = manager.Queue()
            right_progress_queue = manager.Queue()
            jobs[0]["progress_queue"] = left_progress_queue
            jobs[1]["progress_queue"] = right_progress_queue
            with ProcessPoolExecutor(max_workers=2) as executor:
                left_future = executor.submit(collect_stats_worker, jobs[0])
                right_future = executor.submit(collect_stats_worker, jobs[1])
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    left_task = progress.add_task("[Stats LEFT detection]", total=None)
                    right_task = progress.add_task("[Stats RIGHT detection]", total=None)
                    totals = {}
                    while not (left_future.done() and right_future.done()):
                        drain_progress_queue_to_task(left_progress_queue, progress, left_task, totals)
                        drain_progress_queue_to_task(right_progress_queue, progress, right_task, totals)
                        time.sleep(0.1)
                    drain_progress_queue_to_task(left_progress_queue, progress, left_task, totals)
                    drain_progress_queue_to_task(right_progress_queue, progress, right_task, totals)
                stL, sizeL = left_future.result()
                stR, sizeR = right_future.result()

    print("\n[STATS] LEFT:", json.dumps(compact_stats_for_print(stL), indent=2))
    print("\n[STATS] RIGHT:", json.dumps(compact_stats_for_print(stR), indent=2))
    print(f"\n[STATS] image_size_scaled: L={sizeL}, R={sizeR}")

    write_json(os.path.join(args.out, "stats_left.json"), stL)
    write_json(os.path.join(args.out, "stats_right.json"), stR)

def cmd_mono(args):
    ensure_dir(os.path.dirname(args.out) or ".")
    trimL, trimR = load_sync(args.sync)
    pattern = (args.cols, args.rows)
    square_m = args.square_mm / 1000.0

    frame_indices_left = None
    frame_indices_right = None
    if args.reuse_stats_indices:
        stats_dir = args.stats_dir or os.path.join(os.path.dirname(args.out) or ".", "stats")
        left_stats_path = os.path.join(stats_dir, "stats_left.json")
        right_stats_path = os.path.join(stats_dir, "stats_right.json")
        frame_indices_left = load_positive_indices(left_stats_path)
        frame_indices_right = load_positive_indices(right_stats_path)
        print(
            f"[MONO] Reusing stats indices from {stats_dir}: "
            f"LEFT={len(frame_indices_left)}, RIGHT={len(frame_indices_right)}"
        )

    print("\n[MONO] Collecting samples for LEFT...")
    if args.workers > 1 and frame_indices_left is not None:
        with Manager() as manager:
            progress_queue = manager.Queue()
            with progress_bar(desc="[MONO LEFT detection]", unit="frame") as pbar:
                samplesL, statsL, image_size = collect_samples_parallel_indices(
                    {
                        "video_path": args.left,
                        "trim": trimL,
                        "pattern": pattern,
                        "square_m": square_m,
                        "scale": args.scale,
                        "step": args.step,
                        "max_scan": args.max_scan,
                        "debug_every": args.debug_every,
                        "label": "LEFT",
                        "out_dir": os.path.join(os.path.dirname(args.out) or ".", "mono_debug_left"),
                    },
                    frame_indices_left,
                    args.workers,
                    progress_queue,
                    pbar,
                )
                drain_progress_queue(progress_queue, pbar)
    else:
        samplesL, statsL, image_size = collect_samples_checker(
            args.left, trimL, pattern, square_m,
            scale=args.scale, step=args.step, max_scan=args.max_scan,
            debug_every=args.debug_every, label="LEFT",
            out_dir=os.path.join(os.path.dirname(args.out) or ".", "mono_debug_left"),
            use_cuda=args.use_cuda,
            frame_indices=frame_indices_left,
            no_adaptive=args.no_adaptive,
        )
    print(f"[MONO] LEFT used={len(samplesL)} found_pct={statsL['found_pct']:.1f}%")

    print("\n[MONO] Collecting samples for RIGHT...")
    if args.workers > 1 and frame_indices_right is not None:
        with Manager() as manager:
            progress_queue = manager.Queue()
            with progress_bar(desc="[MONO RIGHT detection]", unit="frame") as pbar:
                samplesR, statsR, image_size_r = collect_samples_parallel_indices(
                    {
                        "video_path": args.right,
                        "trim": trimR,
                        "pattern": pattern,
                        "square_m": square_m,
                        "scale": args.scale,
                        "step": args.step,
                        "max_scan": args.max_scan,
                        "debug_every": args.debug_every,
                        "label": "RIGHT",
                        "out_dir": os.path.join(os.path.dirname(args.out) or ".", "mono_debug_right"),
                    },
                    frame_indices_right,
                    args.workers,
                    progress_queue,
                    pbar,
                )
                drain_progress_queue(progress_queue, pbar)
    else:
        samplesR, statsR, image_size_r = collect_samples_checker(
            args.right, trimR, pattern, square_m,
            scale=args.scale, step=args.step, max_scan=args.max_scan,
            debug_every=args.debug_every, label="RIGHT",
            out_dir=os.path.join(os.path.dirname(args.out) or ".", "mono_debug_right"),
            use_cuda=args.use_cuda,
            frame_indices=frame_indices_right,
            no_adaptive=args.no_adaptive,
        )
    print(f"[MONO] RIGHT used={len(samplesR)} found_pct={statsR['found_pct']:.1f}%")

    if image_size != image_size_r:
        print("[MONO] WARNING: scaled sizes differ. That’s unusual.")

    if len(samplesL) < 20 or len(samplesR) < 20:
        raise RuntimeError("[MONO] Not enough samples. Record a longer/better clip with more poses.")

    print("\n[MONO] Calibrating LEFT...")
    rmsL, K1, D1, rvecsL, tvecsL = calibrate_mono(samplesL, image_size)
    repL = reprojection_stats(samplesL, K1, D1, rvecsL, tvecsL)
    print(f"[MONO] LEFT RMS={rmsL:.3f}px  reproj(mean/median/max)={repL['mean_px']:.3f}/{repL['median_px']:.3f}/{repL['max_px']:.3f}px")

    print("\n[MONO] Calibrating RIGHT...")
    rmsR, K2, D2, rvecsR, tvecsR = calibrate_mono(samplesR, image_size)
    repR = reprojection_stats(samplesR, K2, D2, rvecsR, tvecsR)
    print(f"[MONO] RIGHT RMS={rmsR:.3f}px  reproj(mean/median/max)={repR['mean_px']:.3f}/{repR['median_px']:.3f}/{repR['max_px']:.3f}px")

    np.savez(args.out,
             image_size=np.array(image_size, np.int32),
             scale=np.array([args.scale], np.float32),
             K1=K1, D1=D1, K2=K2, D2=D2,
             mono_rms_left=np.array([rmsL], np.float32),
             mono_rms_right=np.array([rmsR], np.float32))

    write_json(os.path.splitext(args.out)[0] + "_report.json", {
        "mono_rms_left_px": float(rmsL),
        "mono_rms_right_px": float(rmsR),
        "reproj_left": repL,
        "reproj_right": repR,
        "left_collection": statsL,
        "right_collection": statsR,
        "pattern": list(pattern),
        "square_mm": float(args.square_mm),
        "scale": float(args.scale),
    })

    print(f"[MONO] Saved: {args.out}")
    print(f"[MONO] Report: {os.path.splitext(args.out)[0] + '_report.json'}")

def cmd_stereo(args):
    ensure_dir(os.path.dirname(args.out) or ".")
    trimL, trimR = load_sync(args.sync)
    pattern = (args.cols, args.rows)
    square_m = args.square_mm / 1000.0

    mono = np.load(args.mono)
    image_size = tuple(mono["image_size"].tolist())
    K1, D1 = mono["K1"], mono["D1"]
    K2, D2 = mono["K2"], mono["D2"]
    mono_scale = float(mono["scale"][0]) if "scale" in mono else None
    if mono_scale is not None and not np.isclose(mono_scale, args.scale):
        raise RuntimeError(f"[STEREO] Scale mismatch: mono scale={mono_scale} vs --scale={args.scale}")

    capL = open_video(args.left)
    capR = open_video(args.right)
    totalL, _, _ = get_video_info(capL)
    totalR, _, _ = get_video_info(capR)
    capL.release(); capR.release()

    obj_template = make_object_points(pattern, square_m)

    usedL, usedR = [], []
    scanned = 0
    ensure_dir(os.path.join(os.path.dirname(args.out) or ".", "stereo_debug"))

    paired_indices = None
    if args.reuse_stats_indices:
        stats_dir = args.stats_dir or os.path.join(os.path.dirname(args.out) or ".", "stats")
        left_stats_path = os.path.join(stats_dir, "stats_left.json")
        right_stats_path = os.path.join(stats_dir, "stats_right.json")
        left_pos = load_positive_indices(left_stats_path)
        right_pos = load_positive_indices(right_stats_path)
        paired_indices = build_paired_indices_from_stats(trimL, trimR, left_pos, right_pos)
        if args.max_scan > 0:
            paired_indices = paired_indices[:args.max_scan]
        print(
            f"[STEREO] Reusing stats indices from {stats_dir}: "
            f"left_pos={len(left_pos)}, right_pos={len(right_pos)}, paired={len(paired_indices)}"
        )
        if args.step != 5:
            print("[STEREO] NOTE: --step is ignored when --reuse-stats-indices is enabled.")

    print("\n[STEREO] Collecting paired samples...")
    if paired_indices is not None:
        if args.workers > 1:
            with Manager() as manager:
                progress_queue = manager.Queue()
                with progress_bar(desc="[STEREO detection]", unit="pair") as pbar:
                    usedL, usedR, scanned = collect_stereo_pairs_parallel(
                        args, paired_indices, pattern, square_m, args.workers, progress_queue, pbar
                    )
                    drain_progress_queue(progress_queue, pbar)
            if len(usedL) > args.max_pairs:
                usedL = usedL[:args.max_pairs]
                usedR = usedR[:args.max_pairs]
        else:
            capL = open_video(args.left)
            capR = open_video(args.right)
            pbar = progress_bar(total=len(paired_indices), desc="[Stereo pairing]", unit="frame")
            nextL = None
            nextR = None
            try:
                for idxL, idxR in paired_indices:
                    if idxL >= totalL or idxR >= totalR:
                        pbar.update(1)
                        continue

                    okL, fL, nextL = read_frame_progressive(capL, idxL, nextL)
                    okR, fR, nextR = read_frame_progressive(capR, idxR, nextR)
                    scanned += 1
                    pbar.update(1)
                    if not okL or not okR:
                        continue

                    fL, gL = prepare_frame_and_gray(fL, args.scale)
                    fR, gR = prepare_frame_and_gray(fR, args.scale)

                    foundL, cL = detect_checkerboard(gL, pattern)
                    foundR, cR = detect_checkerboard(gR, pattern)

                    if foundL and foundR:
                        usedL.append(Sample(idxL, cL.reshape(-1,2), obj_template.copy()))
                        usedR.append(Sample(idxR, cR.reshape(-1,2), obj_template.copy()))

                    if args.debug_every > 0 and (scanned % args.debug_every == 0):
                        visL = fL.copy()
                        visR = fR.copy()
                        if foundL: cv2.drawChessboardCorners(visL, pattern, cL, True)
                        if foundR: cv2.drawChessboardCorners(visR, pattern, cR, True)
                        both = np.hstack([visL, visR])
                        cv2.putText(both, f"scan={scanned} paired={len(usedL)}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                        cv2.imwrite(os.path.join(os.path.dirname(args.out) or ".", "stereo_debug",
                                                 f"pair_{scanned:05d}.png"), both)

                    if len(usedL) >= args.max_pairs:
                        break
            finally:
                pbar.close()
                capL.release(); capR.release()
    else:
        capL = open_video(args.left)
        capR = open_video(args.right)
        max_g = min(totalL - trimL, totalR - trimR) - 1
        try:
            if max_g < 0:
                raise RuntimeError("[STEREO] No overlapping synchronized frame range after trim.")

            steps = [int(args.step)] if args.no_adaptive else adaptive_halving_steps(args.step, levels=4)
            intervals_g = [(0, max_g)]
            visited_g = set()
            nextL = None
            nextR = None
            for pass_i, pass_step in enumerate(steps, start=1):
                remaining = args.max_scan - scanned
                if remaining <= 0 or len(intervals_g) == 0:
                    break

                g_list = build_scan_indices(intervals_g, pass_step, 0, max_g)
                g_list = [g for g in g_list if g not in visited_g]
                if len(g_list) == 0:
                    continue
                if len(g_list) > remaining:
                    g_list = g_list[:remaining]

                print(
                    f"[STEREO] {'simple' if args.no_adaptive else 'adaptive'} pass {pass_i}/{len(steps)}: "
                    f"step={pass_step}, intervals={len(intervals_g)}, pairs={len(g_list)}"
                )
                pbar = progress_bar(total=len(g_list), desc=f"[Stereo pairing s={pass_step}]", unit="frame")

                for g in g_list:
                    idxL = trimL + int(g)
                    idxR = trimR + int(g)

                    okL, fL, nextL = read_frame_progressive(capL, idxL, nextL)
                    okR, fR, nextR = read_frame_progressive(capR, idxR, nextR)
                    visited_g.add(int(g))
                    pbar.update(1)
                    if not okL or not okR:
                        continue

                    fL, gL = prepare_frame_and_gray(fL, args.scale)
                    fR, gR = prepare_frame_and_gray(fR, args.scale)

                    scanned += 1
                    foundL, cL = detect_checkerboard(gL, pattern)
                    foundR, cR = detect_checkerboard(gR, pattern)

                    if foundL and foundR:
                        usedL.append(Sample(idxL, cL.reshape(-1,2), obj_template.copy()))
                        usedR.append(Sample(idxR, cR.reshape(-1,2), obj_template.copy()))

                    if args.debug_every > 0 and (scanned % args.debug_every == 0):
                        visL = fL.copy()
                        visR = fR.copy()
                        if foundL: cv2.drawChessboardCorners(visL, pattern, cL, True)
                        if foundR: cv2.drawChessboardCorners(visR, pattern, cR, True)
                        both = np.hstack([visL, visR])
                        cv2.putText(both, f"scan={scanned} paired={len(usedL)}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                        cv2.imwrite(os.path.join(os.path.dirname(args.out) or ".", "stereo_debug",
                                                 f"pair_{scanned:05d}.png"), both)

                    if len(usedL) >= args.max_pairs:
                        break

                pbar.close()

                if len(usedL) >= args.max_pairs:
                    break

                if pass_i < len(steps) and len(usedL) > 0:
                    found_g = [s.frame_idx - trimL for s in usedL]
                    intervals_g = build_roi_intervals(found_g, margin=pass_step * 2, min_idx=0, max_idx=max_g)

                if scanned >= args.max_scan:
                    break
        finally:
            capL.release(); capR.release()

    print(f"[STEREO] scanned={scanned}, collected_pairs={len(usedL)}")
    if len(usedL) < 20:
        raise RuntimeError("[STEREO] Not enough paired detections. Record a better clip (board visible in both).")

    print("[STEREO] stereoCalibrate(FIX_INTRINSIC)...")
    rmsS, R, T, E, F = stereo_calibrate(usedL, usedR, image_size, K1, D1, K2, D2)

    baseline_m = float(np.linalg.norm(T))
    print(f"[STEREO] RMS={rmsS:.3f}px  baseline={baseline_m*1000:.2f} mm")

    np.savez(args.out,
             image_size=np.array(image_size, np.int32),
             scale=np.array([args.scale], np.float32),
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, E=E, F=F,
             stereo_rms=np.array([rmsS], np.float32),
             baseline_m=np.array([baseline_m], np.float32),
             used_pairs=np.array([len(usedL)], np.int32))

    write_json(os.path.splitext(args.out)[0] + "_report.json", {
        "stereo_rms_px": float(rmsS),
        "baseline_m": baseline_m,
        "baseline_mm": baseline_m * 1000.0,
        "used_pairs": len(usedL),
        "scale": float(args.scale),
        "image_size": list(image_size),
    })

    print(f"[STEREO] Saved: {args.out}")
    print(f"[STEREO] Report: {os.path.splitext(args.out)[0] + '_report.json'}")

def cmd_rectify(args):
    ensure_dir(args.out)
    trimL, trimR = load_sync(args.sync)

    stereo = np.load(args.stereo)
    image_size = tuple(stereo["image_size"].tolist())
    K1, D1 = stereo["K1"], stereo["D1"]
    K2, D2 = stereo["K2"], stereo["D2"]
    R, T = stereo["R"], stereo["T"]
    stereo_scale = float(stereo["scale"][0]) if "scale" in stereo else None
    if stereo_scale is not None and not np.isclose(stereo_scale, args.scale):
        raise RuntimeError(f"[RECTIFY] Scale mismatch: stereo scale={stereo_scale} vs --scale={args.scale}")

    (_, _, _, _, _), (map1x, map1y, map2x, map2y) = rectify_pair(K1, D1, K2, D2, R, T, image_size, alpha=args.alpha)

    capL = open_video(args.left)
    capR = open_video(args.right)
    idxL = trimL + args.frame_offset
    idxR = trimR + args.frame_offset

    okL, fL = read_frame_at(capL, idxL)
    okR, fR = read_frame_at(capR, idxR)
    capL.release(); capR.release()
    if not okL or not okR:
        raise RuntimeError("[RECTIFY] Could not read sample frames.")

    fL = downscale(fL, args.scale)
    fR = downscale(fR, args.scale)
    rectL, rectR = remap_pair_with_backend(fL, fR, map1x, map1y, map2x, map2y, args.use_cuda)

    both = np.hstack([rectL, rectR])
    h = both.shape[0]
    for y in range(0, h, 40):
        cv2.line(both, (0, y), (both.shape[1], y), (0,255,0), 1)
    cv2.putText(both, f"Rectified @ offset={args.frame_offset}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    out_png = os.path.join(args.out, "rectified_check.png")
    cv2.imwrite(out_png, both)
    print(f"[RECTIFY] Saved: {out_png}")
    print("[RECTIFY] Close the window when done.")
    show("RECTIFIED CHECK (epipolar lines)", both, wait=0)


def main():
    ap = argparse.ArgumentParser("Stereo checkerboard debug pipeline (loud + modular)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sync")
    sp.add_argument("--left", default=DEFAULT_LEFT_VIDEO)
    sp.add_argument("--right", default=DEFAULT_RIGHT_VIDEO)
    sp.add_argument("--out", default=DEFAULT_SYNC_JSON)
    sp.add_argument("--scale", type=float, default=1.0)
    sp.add_argument("--step", type=int, default=5)
    sp.add_argument("--max-frames", type=int, default=4000)
    sp.add_argument("--sync-mode", choices=["flash", "audio"], default="flash",
                    help="Sync mode: flash (brightness) or audio peak")
    sp.add_argument("--audio-sample-rate", type=int, default=16000,
                    help="Audio sample rate used for --sync-mode audio")
    sp.add_argument("--use-cuda", action="store_true", help="Use OpenCV CUDA ops when available; auto-fallback to CPU")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("stats")
    sp.add_argument("--left", default=DEFAULT_LEFT_VIDEO)
    sp.add_argument("--right", default=DEFAULT_RIGHT_VIDEO)
    sp.add_argument("--sync", default=DEFAULT_SYNC_JSON)
    sp.add_argument("--out", default=DEFAULT_STATS_DIR)
    sp.add_argument("--cols", type=int, default=9)
    sp.add_argument("--rows", type=int, default=7)
    sp.add_argument("--square-mm", type=float, default=40.0)
    sp.add_argument("--scale", type=float, default=1.0)
    sp.add_argument("--step", type=int, default=5)
    sp.add_argument("--max-scan", type=int, default=3000)
    sp.add_argument("--debug-every", type=int, default=0)
    sp.add_argument("--workers", type=int, default=2,
                    help="Stats worker processes. 1 scans serially; 2 scans left/right in parallel; >2 splits frames within each video.")
    sp.add_argument("--no-adaptive", action="store_true",
                    help="Disable adaptive halving; scan exactly once at --step.")
    sp.add_argument("--use-cuda", action="store_true", help="Use OpenCV CUDA ops when available; auto-fallback to CPU")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("mono")
    sp.add_argument("--left", default=DEFAULT_LEFT_VIDEO)
    sp.add_argument("--right", default=DEFAULT_RIGHT_VIDEO)
    sp.add_argument("--sync", default=DEFAULT_SYNC_JSON)
    sp.add_argument("--out", default=DEFAULT_MONO_NPZ)
    sp.add_argument("--cols", type=int, default=9)
    sp.add_argument("--rows", type=int, default=7)
    sp.add_argument("--square-mm", type=float, default=40.0)
    sp.add_argument("--scale", type=float, default=1.0)
    sp.add_argument("--step", type=int, default=5)
    sp.add_argument("--max-scan", type=int, default=6000)
    sp.add_argument("--debug-every", type=int, default=0)
    sp.add_argument("--workers", type=int, default=2,
                    help="Mono detection worker processes when --reuse-stats-indices is enabled.")
    sp.add_argument("--no-adaptive", action="store_true",
                    help="Disable adaptive halving; scan exactly once at --step.")
    sp.add_argument("--reuse-stats-indices", type=parse_bool, nargs="?", const=True, default=True,
                    metavar="true|false",
                    help="Reuse positive detection frame indices from stats output instead of rescanning. Default: true.")
    sp.add_argument("--stats-dir", default=DEFAULT_STATS_DIR,
                    help="Directory containing stats_left.json and stats_right.json")
    sp.add_argument("--use-cuda", action="store_true", help="Use OpenCV CUDA ops when available; auto-fallback to CPU")
    sp.set_defaults(func=cmd_mono)

    sp = sub.add_parser("stereo")
    sp.add_argument("--left", default=DEFAULT_LEFT_VIDEO)
    sp.add_argument("--right", default=DEFAULT_RIGHT_VIDEO)
    sp.add_argument("--sync", default=DEFAULT_SYNC_JSON)
    sp.add_argument("--mono", default=DEFAULT_MONO_NPZ)
    sp.add_argument("--out", default=DEFAULT_STEREO_NPZ)
    sp.add_argument("--cols", type=int, default=9)
    sp.add_argument("--rows", type=int, default=7)
    sp.add_argument("--square-mm", type=float, default=40.0)
    sp.add_argument("--scale", type=float, default=1.0)
    sp.add_argument("--step", type=int, default=5)
    sp.add_argument("--max-scan", type=int, default=12000)
    sp.add_argument("--max-pairs", type=int, default=200)
    sp.add_argument("--debug-every", type=int, default=0)
    sp.add_argument("--workers", type=int, default=2,
                    help="Stereo pair detection worker processes when --reuse-stats-indices is enabled.")
    sp.add_argument("--no-adaptive", action="store_true",
                    help="Disable adaptive halving; scan exactly once at --step.")
    sp.add_argument("--reuse-stats-indices", type=parse_bool, nargs="?", const=True, default=True,
                    metavar="true|false",
                    help="Reuse synchronized positive detection frame indices from stats output instead of rescanning. Default: true.")
    sp.add_argument("--stats-dir", default=DEFAULT_STATS_DIR,
                    help="Directory containing stats_left.json and stats_right.json")
    sp.add_argument("--use-cuda", action="store_true", help="Use OpenCV CUDA ops when available; auto-fallback to CPU")
    sp.set_defaults(func=cmd_stereo)

    sp = sub.add_parser("rectify")
    sp.add_argument("--left", default=DEFAULT_LEFT_VIDEO)
    sp.add_argument("--right", default=DEFAULT_RIGHT_VIDEO)
    sp.add_argument("--sync", default=DEFAULT_SYNC_JSON)
    sp.add_argument("--stereo", default=DEFAULT_STEREO_NPZ)
    sp.add_argument("--out", default=DEFAULT_RECTIFY_DIR)
    sp.add_argument("--scale", type=float, default=1.0)
    sp.add_argument("--frame-offset", type=int, default=100)
    sp.add_argument("--alpha", type=float, default=0.0, help="0=crop black borders, 1=keep all FOV")
    sp.add_argument("--use-cuda", action="store_true", help="Use OpenCV CUDA ops when available; auto-fallback to CPU")
    sp.set_defaults(func=cmd_rectify)

    args = ap.parse_args()
    validate_args(args)
    args.func(args)

if __name__ == "__main__":
    main()


"""run commands
python stereo_checker_debug.py sync --sync-mode audio
python stereo_checker_debug.py stats --step 50 --max-scan 1000 --workers 16
python stereo_checker_debug.py mono --max-scan 100 --workers 16
python stereo_checker_debug.py stereo --max-pairs 50 --workers 16
python stereo_checker_debug.py rectify --frame-offset 150 --alpha 0.2

For GPU usage, add --use-cuda to the above commands, e.g.:
python stereo_checker_debug.py mono --max-scan 100 --debug-every 100 --use-cuda
"""
