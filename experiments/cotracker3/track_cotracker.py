"""Track the painted-knot grid of one camera with CoTracker3.

Drop-in replacement for the SAM2 marker stage: it reads the same setup prompts
(``setup/prompts/points_left_right.json``) and writes the same ``tracks_2d.csv``
schema, so ``triangulation/points_to_3d.py`` runs on the output unchanged.

Frames are extracted with the exact ffmpeg seek used by ``run_sam2_markers.py``
so frame indices line up with the SAM2 tracks of the same run.

Example::

    .venv-cotracker/Scripts/python experiments/cotracker3/track_cotracker.py \
        --run-dir work/trial15_baseline/run_006_v_0.3mps_a --side left \
        --video "raw videos/trial15left.MP4" --model-res 384x512 --refine dtpeak
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
FRAME_CACHE = ROOT / "work" / "cotracker3" / "frames"


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


def load_video_tensor(frames: list[Path], model_hw: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load frames resized straight to the model resolution as a (1, T, 3, H, W) float tensor.

    Resizing on load keeps host memory small; the predictor then sees an input that is
    already at its interpolation size, so its own resize is a no-op.
    """

    h, w = model_hw
    first = cv2.imread(str(frames[0]))
    full_hw = first.shape[:2]
    video = np.empty((len(frames), h, w, 3), dtype=np.uint8)
    for i, path in enumerate(frames):
        bgr = cv2.imread(str(path))
        video[i] = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    return tensor, full_hw


def build_predictor(model: str, model_hw: tuple[int, int], device: str):
    if model == "offline":
        predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    else:
        predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    # Inputs are pre-resized to model_hw, so the predictor's own resize must target the same size.
    predictor.interp_shape = model_hw
    return predictor.to(device).eval()


@torch.inference_mode()
def run_tracker(predictor, model: str, video: torch.Tensor, queries: torch.Tensor):
    if model == "offline":
        tracks, vis = predictor(video, queries=queries)
        return tracks, vis
    predictor(video_chunk=video, is_first_step=True, queries=queries)
    tracks = vis = None
    for i in range(0, video.shape[1] - predictor.step, predictor.step):
        tracks, vis = predictor(video_chunk=video[:, i : i + predictor.step * 2])
    return tracks, vis


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path, help="Run folder containing setup/prompts/points_left_right.json")
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--video", required=True, type=Path, help="Raw video for this side")
    ap.add_argument("--model", choices=["offline", "online"], default="offline")
    ap.add_argument("--model-res", default="384x512", help="Tracker input resolution HxW (trained at 384x512)")
    ap.add_argument("--refine", choices=["none", "dtpeak"], default="none")
    ap.add_argument("--refine-radius", type=int, default=40, help="Half-window (full-res px) for --refine dtpeak")
    ap.add_argument("--out", type=Path, required=True, help="Output folder; tracks_2d.csv is written inside")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    prompts = json.loads((args.run_dir / "setup" / "prompts" / "points_left_right.json").read_text())
    start, end = int(prompts["frame_range"]["start_frame"]), int(prompts["frame_range"]["end_frame"])
    points = np.array([[p["x"], p["y"]] for p in prompts[args.side]], dtype=np.float32)
    model_hw = tuple(int(x) for x in args.model_res.lower().split("x"))
    if model_hw[0] % 4 or model_hw[1] % 4:
        raise SystemExit("--model-res must be divisible by 4")

    video_path = args.video if args.video.is_absolute() else ROOT / args.video
    frames = extract_frames(video_path, start, end, FRAME_CACHE / f"{video_path.stem}_{start}_{end}")

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    video, (full_h, full_w) = load_video_tensor(frames, model_hw)
    t_load = time.perf_counter() - t0

    # Queries are (t, x, y) in the tensor's pixel space, i.e. the model resolution.
    sx, sy = (model_hw[1] - 1) / (full_w - 1), (model_hw[0] - 1) / (full_h - 1)
    queries = torch.zeros((1, len(points), 3))
    queries[0, :, 1] = torch.from_numpy(points[:, 0] * sx)
    queries[0, :, 2] = torch.from_numpy(points[:, 1] * sy)

    predictor = build_predictor(args.model, model_hw, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    tracks, vis = run_tracker(predictor, args.model, video.to(device), queries.to(device))
    if device == "cuda":
        torch.cuda.synchronize()
    t_track = time.perf_counter() - t1
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0

    tracks = tracks[0].float().cpu().numpy()  # (T, N, 2) in model pixels
    vis = vis[0].float().cpu().numpy()
    if vis.ndim == 3:
        vis = vis[..., 0]
    coarse = tracks / np.array([sx, sy], dtype=np.float32)

    rows = []
    t2 = time.perf_counter()
    for t in range(coarse.shape[0]):
        bgr = cv2.imread(str(frames[t])) if args.refine == "dtpeak" else None
        for i in range(coarse.shape[1]):
            cu, cv_ = float(coarse[t, i, 0]), float(coarse[t, i, 1])
            u, v, method, area, core = cu, cv_, "cotracker", 0.0, 0.0
            if bgr is not None:
                hit = refine_dtpeak(bgr, cu, cv_, args.refine_radius)
                if hit is not None:
                    u, v, area, core = hit
                    method = "cotracker+dtpeak"
            inside = 0 <= u < full_w and 0 <= v < full_h
            rows.append({
                "frame": t, "obj_id": i, "u": u, "v": v, "u_local": u, "v_local": v,
                "method": method, "area_px": area, "core_radius_px": core, "solidity": 1.0,
                "quality": float(vis[t, i]), "valid": int(inside),
                "u_coarse": cu, "v_coarse": cv_, "visibility": float(vis[t, i]),
            })
    t_refine = time.perf_counter() - t2

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "tracks_2d.csv", index=False)
    meta = {
        "side": args.side, "video": str(video_path), "frame_range": [start, end],
        "model": args.model, "model_res": list(model_hw), "refine": args.refine,
        "refine_radius": args.refine_radius, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "n_frames": int(coarse.shape[0]), "n_points": int(coarse.shape[1]),
        "seconds": {"load": round(t_load, 2), "track": round(t_track, 2), "refine": round(t_refine, 2)},
        "peak_gpu_mem_gib": round(peak_mem, 2),
        "n_params_m": round(sum(p.numel() for p in predictor.parameters()) / 1e6, 2),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
