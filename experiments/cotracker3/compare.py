"""Compare CoTracker3 tracks against the SAM2 tracks of the same run.

There is no hand-labelled ground truth, so the comparison uses:

* 2D agreement with SAM2 per marker and frame, plus identity swaps (a CoTracker point
  that sits on a *different* SAM2 marker than its own);
* the 3D result after triangulating both through the unchanged tri3d pipeline:
  stereo reprojection error, the fraction of valid 3D points, temporal jitter
  (second difference of each knot's 3D path), and the stability of the 3D length of
  every grid edge between neighbouring knots, which are material points of the net.

Example::

    python experiments/cotracker3/compare.py --run-dir work/trial15_baseline/run_006_v_0.3mps_a \
        --cotracker work/cotracker3/run_006/offline_384x512_dtpeak
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRID = 12
LEFT_VIDEO = ROOT / "raw videos" / "trial15left.MP4"
RIGHT_VIDEO = ROOT / "raw videos" / "trial15right.MP4"


def compare_2d(sam2_csv: Path, cot_csv: Path) -> dict:
    sam = pd.read_csv(sam2_csv)
    sam = sam[sam["valid"] == 1]
    cot = pd.read_csv(cot_csv)
    merged = sam.merge(cot, on=["frame", "obj_id"], suffixes=("_s", "_c"))
    d = np.hypot(merged["u_s"] - merged["u_c"], merged["v_s"] - merged["v_c"])
    pitch = float(np.median(_grid_pitch_2d(sam)))

    swaps = 0
    for frame, group in merged.groupby("frame"):
        s_xy = sam[sam["frame"] == frame][["obj_id", "u", "v"]].to_numpy()
        for _, row in group.iterrows():
            nearest = s_xy[np.argmin(np.hypot(s_xy[:, 1] - row["u_c"], s_xy[:, 2] - row["v_c"])), 0]
            if int(nearest) != int(row["obj_id"]):
                swaps += 1
    out = {
        "pairs": int(len(merged)),
        "grid_pitch_px": round(pitch, 1),
        "dist_px": _stats(d),
        "frac_over_10px": round(float((d > 10).mean()), 4),
        "frac_over_half_pitch": round(float((d > pitch / 2).mean()), 4),
        "id_swaps": swaps,
        "worst_markers": (
            merged.assign(d=d).groupby("obj_id")["d"].max().sort_values(ascending=False).head(5).round(1).to_dict()
        ),
    }
    if "u_coarse" in cot.columns:
        dc = np.hypot(merged["u_s"] - merged["u_coarse"], merged["v_s"] - merged["v_coarse"])
        out["coarse_dist_px"] = _stats(dc)
    return out


def _grid_pitch_2d(tracks: pd.DataFrame) -> np.ndarray:
    first = tracks[tracks["frame"] == tracks["frame"].min()].set_index("obj_id")
    gaps = []
    for oid in first.index:
        if oid % GRID < GRID - 1 and oid + 1 in first.index:
            gaps.append(np.hypot(*(first.loc[oid + 1, ["u", "v"]].to_numpy() - first.loc[oid, ["u", "v"]].to_numpy())))
    return np.array(gaps)


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {k: round(float(v), 3) for k, v in
            {"median": np.median(x), "mean": x.mean(), "p95": np.percentile(x, 95), "max": x.max()}.items()}


def triangulate(run_dir: Path, left_csv: Path, right_csv: Path, out_dir: Path) -> Path:
    prompts = json.loads((run_dir / "setup" / "prompts" / "points_left_right.json").read_text())
    calib = run_dir.parent / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "tri.csv"
    cmd = [
        sys.executable, "triangulation/points_to_3d.py",
        "--left", str(left_csv), "--right", str(right_csv),
        "--left-video", str(LEFT_VIDEO), "--right-video", str(RIGHT_VIDEO),
        "--start-frame", str(prompts["frame_range"]["start_frame"]),
        "--end-frame", str(prompts["frame_range"]["end_frame"]),
        "--stereo", str(calib / "stereo.npz"), "--sync-json", str(calib / "sync.json"),
        "--out-csv", str(out_csv), "--out-summary", str(out_dir / "tri_summary.json"),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return out_csv


def metrics_3d(tri_csv: Path) -> dict:
    tri = pd.read_csv(tri_csv)
    n_expected = tri["gframe"].nunique() * GRID * GRID
    valid = tri[tri["valid_3d"] == 1]
    pos = valid.set_index(["gframe", "obj_id"])[["X", "Y", "Z"]] * 1000.0  # mm

    # Edge lengths between grid neighbours, per frame; their spread over time is pure error
    # for a net whose knot-to-knot twine length is fixed (beyond small elastic stretch).
    edge_std, edge_len = [], []
    for oid in range(GRID * GRID):
        for nb in ([oid + 1] if oid % GRID < GRID - 1 else []) + ([oid + GRID] if oid + GRID < GRID * GRID else []):
            try:
                a = pos.xs(oid, level="obj_id")
                b = pos.xs(nb, level="obj_id")
            except KeyError:
                continue
            ab = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            if len(ab) < 10:
                continue
            length = np.sqrt(((ab[["X_a", "Y_a", "Z_a"]].to_numpy() - ab[["X_b", "Y_b", "Z_b"]].to_numpy()) ** 2).sum(1))
            edge_len.append(np.median(length))
            edge_std.append(length.std())

    # Temporal jitter: |x(t+1) - 2x(t) + x(t-1)| per knot, over consecutive valid frames.
    jitter = []
    for _, g in valid.sort_values("gframe").groupby("obj_id"):
        f = g["gframe"].to_numpy()
        xyz = g[["X", "Y", "Z"]].to_numpy() * 1000.0
        ok = (f[2:] - f[1:-1] == 1) & (f[1:-1] - f[:-2] == 1)
        acc = np.linalg.norm(xyz[2:] - 2 * xyz[1:-1] + xyz[:-2], axis=1)[ok]
        jitter.extend(acc.tolist())
    return {
        "valid_3d_frac": round(len(valid) / n_expected, 4),
        "reproj_err_px": _stats(valid["err_px"]),
        "edge_len_mm": _stats(edge_len),
        "edge_len_temporal_std_mm": _stats(edge_std),
        "jitter_2nd_diff_mm": _stats(jitter),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cotracker", required=True, type=Path, help="Folder with left/ and right/ tracks_2d.csv")
    args = ap.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    cot_dir = args.cotracker if args.cotracker.is_absolute() else ROOT / args.cotracker

    report = {"run": run_dir.name, "cotracker": str(cot_dir.relative_to(ROOT))}
    for side in ("left", "right"):
        report[f"2d_{side}"] = compare_2d(run_dir / "sam2" / side / "tracks_2d.csv", cot_dir / side / "tracks_2d.csv")
        meta = cot_dir / side / "run_meta.json"
        if meta.exists():
            report[f"meta_{side}"] = json.loads(meta.read_text())
    sam_tri = run_dir.parent.parent / "cotracker3" / run_dir.name / "sam2_tri" / "tri.csv"
    if not sam_tri.exists():
        triangulate(run_dir, run_dir / "sam2" / "left" / "tracks_2d.csv",
                    run_dir / "sam2" / "right" / "tracks_2d.csv", sam_tri.parent)
    cot_tri = triangulate(run_dir, cot_dir / "left" / "tracks_2d.csv", cot_dir / "right" / "tracks_2d.csv", cot_dir / "tri")
    report["3d_sam2"] = metrics_3d(sam_tri)
    report["3d_cotracker"] = metrics_3d(cot_tri)
    (cot_dir / "compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
