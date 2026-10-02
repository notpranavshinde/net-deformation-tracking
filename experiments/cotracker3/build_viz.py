"""Package SAM2 vs CoTracker3 results of one run into a self-contained HTML viewer.

Reads the two triangulation CSVs written by compare.py, renders a left-camera overlay
video with both trackers drawn on it, and injects everything into viz_template.html.

Example::

    python experiments/cotracker3/build_viz.py --run run_006_v_0.3mps_a --variant offline_384x512_dtpeak
"""

from __future__ import annotations

import argparse
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
GRID = 12
VIDEO_W = 960
SAM2_BGR = (255, 200, 40)    # cyan-ish
COT_BGR = (60, 220, 90)      # green


def load_tri(path: Path) -> pd.DataFrame:
    tri = pd.read_csv(path)
    return tri[tri["valid_3d"] == 1]


def pack_positions(tri: pd.DataFrame, frames: list[int], origin: np.ndarray) -> list:
    """Per frame, a flat list of 144*3 ints in 0.1 mm (None for a missing knot)."""

    out = []
    by_frame = {g: df.set_index("obj_id") for g, df in tri.groupby("gframe")}
    for g in frames:
        df = by_frame.get(g)
        flat = []
        for oid in range(GRID * GRID):
            if df is None or oid not in df.index:
                flat.extend([None, None, None])
                continue
            xyz = (df.loc[oid, ["X", "Y", "Z"]].to_numpy(dtype=float) * 1000.0 - origin) * 10.0
            flat.extend(int(round(v)) for v in xyz)
        out.append(flat)
    return out


def render_overlay(run: str, variant: str, frames_l: list[int], disputed: list[int]) -> bytes:
    run_dir = ROOT / "work" / "trial15_baseline" / run
    prompts = json.loads((run_dir / "setup" / "prompts" / "points_left_right.json").read_text())
    start, end = prompts["frame_range"]["start_frame"], prompts["frame_range"]["end_frame"]
    frame_dir = ROOT / "work" / "cotracker3" / "frames" / f"trial15left_{start}_{end}"
    sam = pd.read_csv(run_dir / "sam2" / "left" / "tracks_2d.csv")
    sam = sam[sam["valid"] == 1]
    cot = pd.read_csv(ROOT / "work" / "cotracker3" / run / variant / "left" / "tracks_2d.csv")

    first = cv2.imread(str(frame_dir / "000000.jpg"))
    scale = VIDEO_W / first.shape[1]
    out_h = int(round(first.shape[0] * scale / 2) * 2)
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames_l):
            img = cv2.imread(str(frame_dir / f"{f:06d}.jpg"))
            img = cv2.resize(img, (VIDEO_W, out_h), interpolation=cv2.INTER_AREA)
            for df, color, label_dx in ((sam[sam["frame"] == f], SAM2_BGR, -34), (cot[cot["frame"] == f], COT_BGR, 10)):
                for r in df.itertuples():
                    p = (int(r.u * scale), int(r.v * scale))
                    hot = r.obj_id in disputed
                    cv2.circle(img, p, 9 if hot else 5, color, 2 if hot else 1, cv2.LINE_AA)
                    if hot:
                        cv2.putText(img, str(r.obj_id), (p[0] + label_dx, p[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(Path(tmp) / f"{i:06d}.png"), img)
        mp4 = Path(tmp) / "overlay.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "29.97",
            "-i", str(Path(tmp) / "%06d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "26",
            "-pix_fmt", "yuv420p", "-g", "10", "-movflags", "+faststart", str(mp4),
        ], check=True)
        return mp4.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="run_006_v_0.3mps_a")
    ap.add_argument("--variant", default="offline_384x512_dtpeak")
    ap.add_argument("--knot", type=int, default=66, help="Knot whose path is plotted over time")
    ap.add_argument("--out", type=Path, default=ROOT / "work" / "cotracker3" / "viz" / "net_tracker_faceoff.html")
    args = ap.parse_args()

    base = ROOT / "work" / "cotracker3" / args.run
    sam = load_tri(base / "sam2_tri" / "tri.csv")
    cot = load_tri(base / args.variant / "tri" / "tri.csv")
    frames = sorted(set(sam["gframe"]) | set(cot["gframe"]))
    frame_l = pd.concat([sam, cot]).groupby("gframe")["frame_L"].first().reindex(frames).astype(int).tolist()

    # Shared origin: the CoTracker grid centre in the first frame, so both nets sit in one space.
    origin = cot[cot["gframe"] == frames[0]][["X", "Y", "Z"]].to_numpy().mean(0) * 1000.0
    compare = json.loads((base / args.variant / "compare.json").read_text())
    disputed = sorted({int(k) for side in ("2d_left",) for k, v in compare[side]["worst_markers"].items() if v > 100})

    data = {
        "run": args.run,
        "speed": args.run.split("_v_")[1].split("mps")[0],
        "frames": frames,
        "frameL": frame_l,
        "fps": 29.97,
        "grid": GRID,
        "knot": args.knot,
        "disputed": disputed,
        "sam2": pack_positions(sam, frames, origin),
        "cot": pack_positions(cot, frames, origin),
        "metrics": {"sam2": compare["3d_sam2"], "cot": compare["3d_cotracker"]},
    }
    video = render_overlay(args.run, args.variant, frame_l, disputed)
    html = (HERE / "viz_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("__VIDEO_SRC__", "data:video/mp4;base64," + base64.b64encode(video).decode())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[OK] {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, video {len(video) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
