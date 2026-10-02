"""Render the occlusion-test results into one self-contained HTML viewer.

For every run and scenario under work/cotracker3/occlusion/ it encodes a short clip
cropped around the occluder, drawing where each covered knot really is (from the
unoccluded reference) and where the tracker put it, plus a per-frame error series.

Example::

    python experiments/cotracker3/build_occlusion_viz.py
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OCC = ROOT / "work" / "cotracker3" / "occlusion"
REF_VARIANT = "offline_384x512_dtpeak"
GRID = 12
CLIP_W = 560
TRUTH = (255, 255, 255)
SEEN = (90, 210, 90)
INFERRED = (43, 154, 243)
WRONG = (79, 90, 255)
RUNS = ["run_006_v_0.3mps_a", "run_003_v_0.15mps_a"]


def crop_box(polys: dict, full_hw, margin: float) -> tuple[int, int, int, int]:
    pts = np.vstack([np.array(p) for p in polys.values()])
    x0, y0 = pts.min(0) - margin
    x1, y1 = pts.max(0) + margin
    # Fixed 4:3 box so every clip has the same shape.
    cx, cy, w = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, (y1 - y0) * 4 / 3)
    h = w * 3 / 4
    x0, y0 = int(np.clip(cx - w / 2, 0, full_hw[1] - w)), int(np.clip(cy - h / 2, 0, full_hw[0] - h))
    return x0, y0, int(w), int(h)


def render_clip(frames: list[Path], polys: dict, ref: np.ndarray, out: np.ndarray, vis: np.ndarray,
                hidden: np.ndarray, wrong: np.ndarray, pitch: float) -> bytes:
    full_hw = cv2.imread(str(frames[0])).shape[:2]
    x0, y0, w, h = crop_box(polys, full_hw, 1.6 * pitch)
    scale = CLIP_W / w
    out_h = int(round(h * scale / 2) * 2)
    ever = hidden.any(0)
    r_truth, r_dot, thick = int(0.17 * pitch), int(0.09 * pitch), max(3, int(0.025 * pitch))
    with tempfile.TemporaryDirectory() as tmp:
        for t, path in enumerate(frames):
            img = cv2.imread(str(path))
            if str(t) in polys:
                cv2.fillPoly(img, [np.array(polys[str(t)], np.int32)], (14, 13, 12), lineType=cv2.LINE_AA)
            for i in np.nonzero(ever)[0]:
                rt, ot = tuple(map(int, ref[t, i])), tuple(map(int, out[t, i]))
                color = WRONG if wrong[t, i] else (SEEN if vis[t, i] else INFERRED)
                cv2.circle(img, rt, r_truth, TRUTH, thick, cv2.LINE_AA)
                if np.linalg.norm(ref[t, i] - out[t, i]) > r_truth:
                    cv2.line(img, rt, ot, color, thick, cv2.LINE_AA)
                cv2.circle(img, ot, r_dot, color, -1, cv2.LINE_AA)
            crop = img[y0:y0 + h, x0:x0 + w]
            cv2.imwrite(str(Path(tmp) / f"{t:06d}.png"), cv2.resize(crop, (CLIP_W, out_h), interpolation=cv2.INTER_AREA))
        mp4 = Path(tmp) / "clip.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "29.97",
            "-i", str(Path(tmp) / "%06d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "28",
            "-pix_fmt", "yuv420p", "-g", "10", "-movflags", "+faststart", str(mp4),
        ], check=True)
        return mp4.read_bytes()


def main() -> int:
    data = {"runs": []}
    for run in RUNS:
        prompts = json.loads((ROOT / "work" / "trial15_baseline" / run / "setup" / "prompts" / "points_left_right.json").read_text())
        start, end = prompts["frame_range"]["start_frame"], prompts["frame_range"]["end_frame"]
        frames = sorted((ROOT / "work" / "cotracker3" / "frames" / f"trial15left_{start}_{end}").glob("*.jpg"))
        ref = (pd.read_csv(ROOT / "work" / "cotracker3" / run / REF_VARIANT / "left" / "tracks_2d.csv")
               .sort_values(["frame", "obj_id"])[["u", "v"]].to_numpy().reshape(len(frames), GRID * GRID, 2))
        results = json.loads((OCC / run / "results.json").read_text())
        scen = []
        for res in results:
            d = OCC / run / res["id"]
            g = pd.read_csv(d / "left_gated.csv").sort_values(["frame", "obj_id"])
            out = g[["u", "v"]].to_numpy().reshape(ref.shape)
            vis = g["visible"].to_numpy().reshape(ref.shape[:2]).astype(bool)
            hidden = np.load(d / "hidden.npy")
            polys = json.loads((d / "polygons.json").read_text())
            err = np.linalg.norm(out - ref, axis=2)
            near = np.argmin(np.linalg.norm(ref[:, None, :, :] - out[:, :, None, :], axis=3), axis=2)
            wrong = near != np.arange(GRID * GRID)[None, :]
            ever = hidden.any(0)
            clip = render_clip(frames, polys, ref, out, vis, hidden, wrong, res["pitch_px"])
            scen.append({
                **res,
                "video": "data:video/mp4;base64," + base64.b64encode(clip).decode(),
                "err_max": np.round(err[:, ever].max(1), 1).tolist(),
                "err_med": np.round(np.median(err[:, ever], 1), 1).tolist(),
                "n_hidden": hidden.sum(1).astype(int).tolist(),
                "n_wrong": wrong[:, ever].sum(1).astype(int).tolist(),
            })
            print(f"[OK] {run} {res['id']} clip {len(clip) / 1e6:.2f} MB")
        data["runs"].append({"run": run, "speed": run.split("_v_")[1].split("mps")[0], "frames": len(frames), "scenarios": scen})
    html = (HERE / "occlusion_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out_path = ROOT / "work" / "cotracker3" / "viz" / "occlusion_lab.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
