"""Synthetic lighting-robustness test for CoTracker3 on real towing-tank footage.

Both camera views of a run are degraded with the same lighting condition, the tracker
is rerun on each, and the results are scored against the reference tracks of the clean
run (CoTracker3 + full-resolution snap). Raw model output and snapped output are scored
separately, because the snap finds knots by colour and is the part most exposed to
lighting.

Conditions (each at mild / moderate / severe):
  dark        underexposed recording with sensor noise
  bright      overexposed, markers clip toward white
  uneven      brightness gradient across the frame plus a hotspot (no diffuser)
  flicker     frame-to-frame brightness changes with rolling-shutter banding
  caustics    moving bright ripples from light through the water surface
  murky       haze veil, red absorbed, forward-scatter blur
  background  a textured background in place of the black liner

Run with the CoTracker environment; triangulation runs in the base ``python``::

    .venv-cotracker/Scripts/python experiments/cotracker3/lighting_test.py --run run_006_v_0.3mps_a
    .venv-cotracker/Scripts/python experiments/cotracker3/lighting_test.py --preview   # stills only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402
import track_cotracker as tc  # noqa: E402

ROOT = tc.ROOT
GRID = 12
N = GRID * GRID
REF_VARIANT = "offline_384x512_dtpeak"
LEVELS = ("mild", "moderate", "severe")
MODEL_HW = (384, 512)


def smooth_noise(h: int, w: int, cells: int, seed: int) -> np.ndarray:
    """Low-frequency noise in [0, 1], made at a coarse grid and upscaled."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((cells, int(cells * w / h) + 1)).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


class Condition:
    """A per-frame lighting degradation. Call with a full-resolution BGR frame and its index."""

    def __init__(self, kind: str, level: str, hw: tuple[int, int]):
        self.kind, self.level = kind, level
        self.k = LEVELS.index(level)
        self.h, self.w = hw
        self.rng = np.random.default_rng(1000 + 10 * list(CONDITIONS).index(kind) + self.k)
        self.p = CONDITIONS[kind][self.k]
        h, w = hw
        if kind == "uneven":
            x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
            y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
            grad = 1.0 - (1.0 - self.p["far"]) * x                      # light falls off to the right
            spot = self.p["spot"] * np.exp(-(((x - 0.3) / 0.18) ** 2 + ((y - 0.3) / 0.22) ** 2))
            self.field = (grad * np.ones((h, 1), np.float32) + spot)[..., None]
        if kind == "background":
            base = smooth_noise(h, w, 6, 7) * 0.6 + smooth_noise(h, w, 40, 8) * 0.4
            # Vertical panel seams and a couple of dark ropes, like a tank wall or pen.
            seams = (np.abs(((np.arange(w) / w * 7) % 1) - 0.5) < 0.02).astype(np.float32)[None, :] * 0.35
            tex = (base + seams).clip(0, 1)
            tint = np.array([0.85, 1.0, 0.8], np.float32)                 # grey-green wall, BGR
            self.bg = (tex[..., None] * tint * self.p["level"]).astype(np.float32)

    def __call__(self, bgr: np.ndarray, t: int) -> np.ndarray:
        img = bgr.astype(np.float32)
        p = self.p
        if self.kind == "dark":
            img *= p["gain"]
            img += self.rng.normal(0, p["noise"], img.shape).astype(np.float32)
        elif self.kind == "bright":
            img *= p["gain"]
        elif self.kind == "uneven":
            img *= self.field
        elif self.kind == "flicker":
            # Beat between mains-powered lights and the frame rate, rolled across rows by the shutter.
            rows = np.arange(self.h, dtype=np.float32)[:, None, None] / self.h
            g = 1 + p["amp"] * np.sin(2 * np.pi * (2.3 * t / 29.97 + p["bands"] * rows))
            img *= g * (1 + 0.5 * p["amp"] * self.rng.uniform(-1, 1))
        elif self.kind == "caustics":
            img = self._caustics(img, t)
        elif self.kind == "murky":
            r = cv2.GaussianBlur(img, (0, 0), p["blur"])
            r *= np.array([1.0, 0.95, p["red"]], np.float32)            # BGR: red absorbed first
            veil = np.array([70, 62, 22], np.float32)                     # blue-green backscatter
            img = r * p["trans"] + veil * (1 - p["trans"])
        elif self.kind == "background":
            # Replace only the black liner (luminance below ~8); twine sits at 8-30 and knots far
            # above, so both stay in front of the new background as dark lines and orange dots.
            lum = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
            alpha = np.clip(1 - (lum - 5) / 4, 0, 1)[..., None]
            img = img + self.bg * alpha
        return np.clip(img, 0, 255).astype(np.uint8)

    def _caustics(self, img: np.ndarray, t: int) -> np.ndarray:
        s = 8  # build the pattern at 1/8 resolution, then upscale
        h, w = self.h // s, self.w // s
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x /= 40.0
        y /= 40.0
        tt = t / 29.97
        f = np.zeros((h, w), np.float32)
        for a, (dx, dy, sp) in enumerate([(1.0, 0.3, 0.9), (-0.4, 1.0, 0.7), (0.7, -0.8, 1.1), (0.2, 0.6, 1.4)]):
            f += np.sin(2 * np.pi * (dx * x + dy * y) * (1.5 + 0.4 * a) + sp * tt * 2 * np.pi)
        ridges = np.clip(1 - np.abs(f) / 1.2, 0, 1) ** 3                 # thin bright lines where waves cross zero
        ridges = cv2.resize(ridges, (self.w, self.h), interpolation=cv2.INTER_CUBIC)[..., None]
        p = self.p
        light = np.array([200, 220, 210], np.float32)                     # pale light, BGR
        return img * (1 + p["amp"] * ridges) + light * p["add"] * ridges


CONDITIONS = {
    "dark": [{"gain": 0.45, "noise": 2.0}, {"gain": 0.25, "noise": 3.0}, {"gain": 0.12, "noise": 4.0}],
    "bright": [{"gain": 2.0}, {"gain": 3.0}, {"gain": 4.5}],
    "uneven": [{"far": 0.6, "spot": 0.5}, {"far": 0.35, "spot": 1.0}, {"far": 0.15, "spot": 2.0}],
    "flicker": [{"amp": 0.2, "bands": 0.0}, {"amp": 0.4, "bands": 1.5}, {"amp": 0.6, "bands": 3.0}],
    "caustics": [{"amp": 0.5, "add": 0.08}, {"amp": 1.0, "add": 0.18}, {"amp": 2.0, "add": 0.35}],
    "murky": [{"trans": 0.75, "red": 0.8, "blur": 2.0}, {"trans": 0.5, "red": 0.6, "blur": 4.0}, {"trans": 0.3, "red": 0.4, "blur": 8.0}],
    "background": [{"level": 45.0}, {"level": 90.0}, {"level": 150.0}],
}
LABELS = {
    "dark": "Underexposed", "bright": "Overexposed", "uneven": "Uneven lighting", "flicker": "Flicker",
    "caustics": "Surface caustics", "murky": "Murky water", "background": "Non-black background",
}


class Degraded:
    """Frames with one condition applied, perturbed once and kept as JPEG bytes for the snap pass."""

    def __init__(self, frames: list[Path], cond: Condition):
        self.jpgs: list[bytes] = []
        h, w = MODEL_HW
        video = np.empty((len(frames), h, w, 3), np.uint8)
        for t, path in enumerate(frames):
            img = cond(cv2.imread(str(path)), t)
            self.jpgs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            video[t] = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        import torch
        self.video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        self.full_hw = img.shape[:2]

    def frame(self, t: int) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpgs[t], np.uint8), cv2.IMREAD_COLOR)


def build_rows(deg: Degraded, tracks, vis, conf) -> pd.DataFrame:
    """tc.build_rows, reading the degraded frames instead of the clean ones."""
    rows = []
    full_h, full_w = deg.full_hw
    for t in range(tracks.shape[0]):
        bgr = deg.frame(t)
        for i in range(N):
            cu, cv_ = float(tracks[t, i, 0]), float(tracks[t, i, 1])
            visible = bool(vis[t, i] > tc.VIS_THRESHOLD)
            u, v, method = cu, cv_, "cotracker_inferred"
            if visible:
                hit = tc.refine_dtpeak(bgr, cu, cv_, 40)
                u, v, method = (hit[0], hit[1], "cotracker+dtpeak") if hit else (cu, cv_, "cotracker")
            rows.append({"frame": t, "obj_id": i, "u": u, "v": v, "u_local": u, "v_local": v, "method": method,
                         "area_px": 0.0, "core_radius_px": 0.0, "solidity": 1.0, "quality": float(vis[t, i]),
                         "valid": int(0 <= u < full_w and 0 <= v < full_h), "u_coarse": cu, "v_coarse": cv_,
                         "visibility": float(vis[t, i]), "confidence": float(conf[t, i]), "visible": int(visible)})
    return pd.DataFrame(rows)


def stats(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"median": round(float(np.median(x)), 2), "p95": round(float(np.percentile(x, 95)), 2), "max": round(float(x.max()), 2)} if len(x) else {}


def score_2d(df: pd.DataFrame, ref: np.ndarray) -> dict:
    df = df.sort_values(["frame", "obj_id"])
    out = df[["u", "v"]].to_numpy().reshape(ref.shape)
    raw = df[["u_coarse", "v_coarse"]].to_numpy().reshape(ref.shape)
    e, er = np.linalg.norm(out - ref, axis=2), np.linalg.norm(raw - ref, axis=2)
    near = np.argmin(np.linalg.norm(ref[:, None] - out[:, :, None], axis=3), axis=2)
    return {
        "snap_err_px": stats(e), "raw_err_px": stats(er),
        "wrong_knot": round(float((near != np.arange(N)).mean()), 4),
        "flagged_hidden": round(float((df["visible"] == 0).mean()), 4),
        "snapped": round(float((df["method"] == "cotracker+dtpeak").mean()), 4),
    }


def score_3d(tri_csv: Path, ref_tri: pd.DataFrame) -> dict:
    tri = pd.read_csv(tri_csv)
    m = tri.merge(ref_tri, on=["gframe", "obj_id"], suffixes=("", "_ref"))
    m = m[(m["valid_3d"] == 1) & (m["valid_3d_ref"] == 1)]
    d = np.linalg.norm((m[["X", "Y", "Z"]].to_numpy() - m[["X_ref", "Y_ref", "Z_ref"]].to_numpy()) * 1000, axis=1)
    out = compare.metrics_3d(tri_csv)
    out["err_vs_clean_mm"] = stats(d)
    return out


def preview(run: str) -> None:
    """One still per condition and level from the middle of the run, for checking the look."""
    run_dir = ROOT / "work" / "trial15_baseline" / run
    start, end, _ = tc.load_prompts(run_dir, "left")
    frames = tc.extract_frames(ROOT / "raw videos" / "trial15left.MP4", start, end, tc.FRAME_CACHE / f"trial15left_{start}_{end}")
    src = cv2.imread(str(frames[100]))
    crop = (slice(1500, 3300), slice(1800, 4200))
    rows = []
    for kind in CONDITIONS:
        tiles = []
        for level in LEVELS:
            img = Condition(kind, level, src.shape[:2])(src, 100)[crop]
            img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{kind} {level}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    out = ROOT / "work" / "cotracker3" / "inspect" / "lighting_preview.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"[OK] {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="run_006_v_0.3mps_a")
    ap.add_argument("--only", nargs="*", help="Condition kinds to run (default: all)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--levels", nargs="*", default=list(LEVELS), choices=LEVELS)
    ap.add_argument("--threads", type=int, default=0, help="Torch CPU threads (0 = default); set when running several in parallel")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
    if args.preview:
        preview(args.run)
        return 0

    run_dir = ROOT / "work" / "trial15_baseline" / args.run
    ref_dir = ROOT / "work" / "cotracker3" / args.run / REF_VARIANT
    out_root = ROOT / "work" / "cotracker3" / "lighting" / args.run
    ref_tri = pd.read_csv(ref_dir / "tri" / "tri.csv")
    model = tc.load_model(args.device)
    sides = {}
    for side in ("left", "right"):
        start, end, points = tc.load_prompts(run_dir, side)
        frames = tc.extract_frames(ROOT / "raw videos" / f"trial15{side}.MP4", start, end, tc.FRAME_CACHE / f"trial15{side}_{start}_{end}")
        ref = pd.read_csv(ref_dir / side / "tracks_2d.csv").sort_values(["frame", "obj_id"])[["u", "v"]].to_numpy().reshape(len(frames), N, 2)
        sides[side] = (frames, points, ref)

    # Each case writes its own result.json, so several processes can run different cases at once.
    for kind in CONDITIONS:
        if args.only and kind not in args.only:
            continue
        for level in args.levels:
            if (out_root / kind / level / "result.json").exists():
                print(f"[skip] {kind} {level}")
                continue
            t0 = time.perf_counter()
            out = out_root / kind / level
            out.mkdir(parents=True, exist_ok=True)
            res = {"kind": kind, "label": LABELS[kind], "level": level, "params": CONDITIONS[kind][LEVELS.index(level)]}
            for side, (frames, points, ref) in sides.items():
                cond = Condition(kind, level, cv2.imread(str(frames[0])).shape[:2])
                deg = Degraded(frames, cond)
                tracks, vis, conf = tc.track_points(model, deg.video, points, deg.full_hw, args.device)
                df = build_rows(deg, tracks, vis, conf)
                df.to_csv(out / f"{side}.csv", index=False)
                res[f"2d_{side}"] = score_2d(df, ref)
                del deg
            tri = compare.triangulate(run_dir, out / "left.csv", out / "right.csv", out / "tri", python="python")
            res["3d"] = score_3d(tri, ref_tri)
            res["seconds"] = round(time.perf_counter() - t0, 1)
            (out / "result.json").write_text(json.dumps(res, indent=2))
            l2 = res["2d_left"]
            print(f"[{kind} {level}] raw={l2['raw_err_px'].get('median')}px snap={l2['snap_err_px'].get('median')}px "
                  f"wrong={l2['wrong_knot']} snapped={l2['snapped']} hidden={l2['flagged_hidden']} "
                  f"3d={res['3d']['err_vs_clean_mm'].get('median')}mm valid={res['3d']['valid_3d_frac']} ({res['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
