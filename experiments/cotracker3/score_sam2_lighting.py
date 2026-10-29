"""Score SAM2 and CoTracker3 on the lighting-degraded clips against one common reference.

Reference = CoTracker3 + snap on the clean footage, checked by eye (the same reference the
lighting test already uses). Both trackers are scored on absolute identity, not on how far each
moved from its own clean run:
  wrong_knot  share of knot-frames whose valid point sits closer to another knot's reference
  lost        share of knot-frames with no valid point (SAM2 drops masks; CoTracker3 always outputs)
  kept3d      share of knot-frames passing the 20 px reprojection gate after triangulation
Baselines: SAM2 on the clean footage as run for the paper, and SAM2 on the clean clip re-encoded
through the same path as the degraded clips.

    python experiments/cotracker3/score_sam2_lighting.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work" / "cotracker3"
OUT = Path(__file__).resolve().parent / "results" / "sam2_vs_cotracker3_lighting.csv"
REF = "offline_384x512_dtpeak"
SPEED = {"run_006_v_0.3mps_a": 0.30, "run_003_v_0.15mps_a": 0.15}
N = 144


def reference(run: str, side: str) -> np.ndarray:
    r = pd.read_csv(WORK / run / REF / side / "tracks_2d.csv").sort_values(["frame", "obj_id"])
    return r[["u", "v"]].to_numpy().reshape(-1, N, 2)


def identity(tracks_csv: Path, ref: np.ndarray) -> tuple[float, float]:
    df = pd.read_csv(tracks_csv)
    xy = np.full(ref.shape, np.nan)
    ok = (df["valid"] == 1) & df["frame"].between(0, ref.shape[0] - 1)
    d = df[ok]
    xy[d["frame"].to_numpy(), d["obj_id"].to_numpy()] = d[["u", "v"]].to_numpy()
    have = np.isfinite(xy[..., 0])
    dist = np.linalg.norm(ref[:, None] - xy[:, :, None], axis=3)
    near = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=2)
    wrong = have & (near != np.arange(N))
    return float(wrong.mean()), float((~have).mean())


def kept3d(run: str, left: Path, right: Path, out: Path) -> float:
    tri = out / "tri" / "tri.csv"
    if not tri.exists():
        compare.triangulate(ROOT / "work" / "trial15_baseline" / run, left, right, out / "tri")
    return compare.metrics_3d(tri)["valid_3d_frac"]  # valid points / (frames x 144), as for CoTracker3


def main() -> int:
    rows = []
    for run, speed in SPEED.items():
        ref = {s: reference(run, s) for s in ("left", "right")}

        def add(tracker, kind, level, left, right, out, kept=None):
            wl, ll = identity(left, ref["left"])
            wr, lr = identity(right, ref["right"])
            rows.append({"run": run, "tow_speed_mps": speed, "tracker": tracker, "condition": kind, "strength": level,
                         "wrong_knot_frac": max(wl, wr), "wrong_knot_left": wl, "wrong_knot_right": wr,
                         "lost_frac": max(ll, lr), "kept3d_frac": kept if kept is not None else kept3d(run, left, right, out)})

        base = ROOT / "work" / "trial15_baseline" / run / "sam2"
        add("SAM2", "clean (paper run)", "none", base / "left" / "tracks_2d.csv", base / "right" / "tracks_2d.csv",
            WORK / run / "sam2_tri", kept=compare.metrics_3d(WORK / run / "sam2_tri" / "tri.csv")["valid_3d_frac"])
        ct_clean = json.loads((WORK / run / REF / "compare.json").read_text())["3d_cotracker"]["valid_3d_frac"]
        add("CoTracker3", "clean", "none", WORK / run / REF / "left" / "tracks_2d.csv", WORK / run / REF / "right" / "tracks_2d.csv",
            None, kept=ct_clean)
        for d in sorted((WORK / "sam2_lighting" / run).glob("*/*")):
            if (d / "left" / "tracks_2d.csv").exists() and (d / "right" / "tracks_2d.csv").exists():
                kind = "clean (re-encoded)" if d.parent.name == "clean" else d.parent.name
                add("SAM2", kind, d.name, d / "left" / "tracks_2d.csv", d / "right" / "tracks_2d.csv", d)
                if d.parent.name != "clean":
                    c = WORK / "lighting" / run / d.parent.name / d.name
                    res = json.loads((c / "result.json").read_text())
                    add("CoTracker3", d.parent.name, d.name, c / "left.csv", c / "right.csv", c, kept=res["3d"]["valid_3d_frac"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200):
        print(df[["tow_speed_mps", "tracker", "condition", "strength", "wrong_knot_frac", "lost_frac", "kept3d_frac"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
