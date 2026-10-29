"""Track the painted-knot grid of one camera with CoTracker3.

Drop-in replacement for the SAM2 marker stage: it reads the same setup prompts
(``setup/prompts/points_left_right.json``) and writes the same ``tracks_2d.csv``
schema, so ``triangulation/points_to_3d.py`` runs on the output unchanged.

Frames are extracted with the exact ffmpeg seek used by ``run_sam2_markers.py``
so frame indices line up with the SAM2 tracks of the same run.

With ``--refine dtpeak`` each prediction the model calls visible is snapped to the
knot centre at full resolution. Predictions the model calls hidden keep the model's
own estimate (method ``cotracker_inferred``): snapping them would grab whatever
visible knot happens to be nearby.

Example::

    .venv-cotracker/Scripts/python experiments/cotracker3/track_cotracker.py \
        --run-dir work/trial15_baseline/run_006_v_0.3mps_a --side left \
        --video "raw videos/trial15left.MP4" --model-res 384x512 --refine dtpeak
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
FRAME_CACHE = ROOT / "work" / "cotracker3" / "frames"
VIS_THRESHOLD = 0.9  # the offline predictor's own cut on the sigmoid visibility

# An occluder edits a full-resolution BGR frame in place, given its frame index.
Occluder = Callable[[np.ndarray, int], None]


def video_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read fps from {video}")
    return fps


def extract_frames(video: Path, start: int, end: int, out_dir: Path) -> list[Path]:
    """Extract source frames [start, end) as JPEGs, matching run_sam2_markers' ffmpeg seek."""

    count = end - start
    existing = sorted(out_dir.glob("*.jpg"))
    if len(existing) == count:
        return existing
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    start_seconds = start / video_fps(video)
    pre_seek = min(5.0, start_seconds)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_seconds - pre_seek:.9f}", "-i", str(video),
        "-ss", f"{pre_seek:.9f}",
        "-frames:v", str(count), "-q:v", "2", "-start_number", "0",
        str(out_dir / "%06d.jpg"),
    ]
    print(f"[INFO] Extracting source frames [{start}, {end}) from {video.name}")
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("*.jpg"))
    if len(frames) != count:
        raise RuntimeError(f"Expected {count} frames, ffmpeg wrote {len(frames)}")
    return frames


def read_frame(path: Path, t: int, occluder: Occluder | None) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if occluder is not None:
        occluder(bgr, t)
    return bgr


def load_video_tensor(frames: list[Path], model_hw: tuple[int, int],
                      occluder: Occluder | None = None) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load frames resized straight to the model resolution as a (1, T, 3, H, W) float tensor."""

    h, w = model_hw
    video = np.empty((len(frames), h, w, 3), dtype=np.uint8)
    full_hw = None
    for t, path in enumerate(frames):
        bgr = read_frame(path, t, occluder)
        full_hw = bgr.shape[:2]
        video[t] = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(video).permute(0, 3, 1, 2)[None].float(), full_hw


def load_model(device: str):
    if device.startswith("cuda"):
        # Full fp32 on the GPU so results match CPU runs (TF32 convolutions shift tracks slightly).
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    return predictor.model.to(device).eval()


@contextmanager
def gpu_lock(device: str):
    """Let one process at a time onto the GPU when several run in parallel (a model call peaks ~17 GB).

    Uses an flock on Linux; a no-op on CPU or where fcntl is unavailable.
    """
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if not device.startswith("cuda") or fcntl is None:
        yield
        return
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "-")
    with open(ROOT / "work" / "cotracker3" / f".gpu{gpu}.lock", "w") as fh:  # one lock per physical GPU
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            fcntl.flock(fh, fcntl.LOCK_UN)


def support_grid(size: int, h: int, w: int) -> torch.Tensor:
    """(1, size*size, 3) frame-0 queries on a grid, as cotracker's get_points_on_a_grid places them."""
    margin = w / 64
    ys = torch.linspace(margin, h - margin, size)
    xs = torch.linspace(margin, w - margin, size)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2)
    return torch.cat([torch.zeros_like(pts[..., :1]), pts], dim=2)


@torch.inference_mode()
def track_points(model, video: torch.Tensor, points: np.ndarray, full_hw: tuple[int, int], device: str,
                 support: int = 6):
    """Run the offline model on a video already at model resolution.

    ``points`` are (N, 2) full-resolution pixel coordinates queried at frame 0. Returns
    full-resolution tracks (T, N, 2), visibility probability (T, N) and confidence (T, N).
    Mirrors CoTrackerPredictor._compute_sparse_tracks without its boolean threshold.
    """

    _, _, _, h, w = video.shape
    sx, sy = (w - 1) / (full_hw[1] - 1), (h - 1) / (full_hw[0] - 1)
    queries = torch.zeros((1, len(points), 3))
    queries[0, :, 1] = torch.from_numpy(points[:, 0] * sx)
    queries[0, :, 2] = torch.from_numpy(points[:, 1] * sy)
    # The official predictor adds a 6 x 6 grid of support points for sparse queries; they shape
    # the space attention, so the knot tracks differ without them. Their outputs are dropped.
    if support:
        queries = torch.cat([queries, support_grid(support, h, w)], dim=1)
    tracks, vis, conf, _ = model(video=video.to(device), queries=queries.to(device), iters=6)
    n = len(points)
    tracks, vis, conf = tracks[:, :, :n], vis[:, :, :n], conf[:, :, :n]
    tracks = tracks[0].float().cpu().numpy() / np.array([sx, sy], dtype=np.float32)
    tracks[0] = points  # the query frame is known exactly
    vis, conf = vis[0].float().cpu().numpy(), conf[0].float().cpu().numpy()
    vis[0] = 1.0
    return tracks, vis, conf


def refine_dtpeak(bgr: np.ndarray, u: float, v: float, radius: int) -> tuple[float, float, float, float] | None:
    """Snap a coarse prediction to the painted knot at full resolution.

    Inside a window around (u, v) the knot is the brightest orange blob on a dark
    background: threshold a red-minus-blue score with Otsu, keep the component nearest
    the window centre, and return the distance-transform peak (the same marker centre
    definition SAM2 used), refined to sub-pixel by a weighted centroid of the DT core.
    Returns (u, v, area_px, core_radius_px) or None when no knot is found.
    """

    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(round(u)) - radius), max(0, int(round(v)) - radius)
    x1, y1 = min(w, int(round(u)) + radius + 1), min(h, int(round(v)) + radius + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    win = bgr[y0:y1, x0:x1].astype(np.int16)
    score = np.clip(win[..., 2] - win[..., 0], 0, 255).astype(np.uint8)
    if score.max() < 30:
        return None
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    cu, cv_ = u - x0, v - y0
    best = min(range(1, n), key=lambda k: (cents[k][0] - cu) ** 2 + (cents[k][1] - cv_) ** 2)
    blob = (labels == best).astype(np.uint8)
    area = float(stats[best, cv2.CC_STAT_AREA])
    if area < 20:
        return None
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    peak = float(dist.max())
    core = dist >= 0.7 * peak
    ys, xs = np.nonzero(core)
    wts = dist[ys, xs]
    return x0 + float((xs * wts).sum() / wts.sum()), y0 + float((ys * wts).sum() / wts.sum()), area, peak


def build_rows(frames: list[Path], tracks: np.ndarray, vis: np.ndarray, conf: np.ndarray, full_hw,
               refine: str, radius: int, occluder: Occluder | None = None, gate: bool = True) -> list[dict]:
    """Turn model output into tracks_2d.csv rows, snapping visible points when refine == 'dtpeak'.

    ``gate=False`` snaps every point regardless of visibility; it exists only so the
    occlusion test can show what ungated snapping does to hidden knots.
    """

    full_h, full_w = full_hw
    rows = []
    for t in range(tracks.shape[0]):
        bgr = read_frame(frames[t], t, occluder) if refine == "dtpeak" else None
        for i in range(tracks.shape[1]):
            cu, cv_ = float(tracks[t, i, 0]), float(tracks[t, i, 1])
            visible = bool(vis[t, i] > VIS_THRESHOLD)
            u, v, method, area, core = cu, cv_, "cotracker", 0.0, 0.0
            if bgr is not None and (visible or not gate):
                hit = refine_dtpeak(bgr, cu, cv_, radius)
                if hit is not None:
                    u, v, area, core = hit
                    method = "cotracker+dtpeak"
            elif bgr is not None:
                method = "cotracker_inferred"
            rows.append({
                "frame": t, "obj_id": i, "u": u, "v": v, "u_local": u, "v_local": v,
                "method": method, "area_px": area, "core_radius_px": core, "solidity": 1.0,
                "quality": float(vis[t, i]), "valid": int(0 <= u < full_w and 0 <= v < full_h),
                "u_coarse": cu, "v_coarse": cv_, "visibility": float(vis[t, i]),
                "confidence": float(conf[t, i]), "visible": int(visible),
            })
    return rows


def load_prompts(run_dir: Path, side: str) -> tuple[int, int, np.ndarray]:
    prompts = json.loads((run_dir / "setup" / "prompts" / "points_left_right.json").read_text())
    start, end = int(prompts["frame_range"]["start_frame"]), int(prompts["frame_range"]["end_frame"])
    points = np.array([[p["x"], p["y"]] for p in prompts[side]], dtype=np.float32)
    return start, end, points


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path, help="Run folder containing setup/prompts/points_left_right.json")
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--video", required=True, type=Path, help="Raw video for this side")
    ap.add_argument("--model-res", default="384x512", help="Tracker input resolution HxW (trained at 384x512)")
    ap.add_argument("--refine", choices=["none", "dtpeak"], default="none")
    ap.add_argument("--refine-radius", type=int, default=40, help="Half-window (full-res px) for --refine dtpeak")
    ap.add_argument("--out", type=Path, required=True, help="Output folder; tracks_2d.csv is written inside")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--drop-frames", action="store_true", help="Delete the extracted frame cache when done")
    args = ap.parse_args()

    start, end, points = load_prompts(args.run_dir, args.side)
    model_hw = tuple(int(x) for x in args.model_res.lower().split("x"))
    if model_hw[0] % 4 or model_hw[1] % 4:
        raise SystemExit("--model-res must be divisible by 4")
    video_path = args.video if args.video.is_absolute() else ROOT / args.video
    frame_dir = FRAME_CACHE / f"{video_path.stem}_{start}_{end}"
    frames = extract_frames(video_path, start, end, frame_dir)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    video, full_hw = load_video_tensor(frames, model_hw)
    t_load = time.perf_counter() - t0
    model = load_model(device)
    t1 = time.perf_counter()
    tracks, vis, conf = track_points(model, video, points, full_hw, device)
    t_track = time.perf_counter() - t1
    t2 = time.perf_counter()
    rows = build_rows(frames, tracks, vis, conf, full_hw, args.refine, args.refine_radius)
    t_refine = time.perf_counter() - t2

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "tracks_2d.csv", index=False)
    meta = {
        "side": args.side, "video": str(video_path), "frame_range": [start, end],
        "model": "offline", "model_res": list(model_hw), "refine": args.refine,
        "refine_radius": args.refine_radius, "device": device,
        "n_frames": int(tracks.shape[0]), "n_points": int(tracks.shape[1]),
        "seconds": {"load": round(t_load, 2), "track": round(t_track, 2), "refine": round(t_refine, 2)},
        "n_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if args.drop_frames:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
