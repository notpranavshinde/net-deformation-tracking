"""Collect every run's compare.json into one table (stdout + work/cotracker3/summary.csv)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "offline_384x512_dtpeak"


def main() -> int:
    rows = []
    for path in sorted((ROOT / "work" / "cotracker3").glob(f"run_*/{VARIANT}/compare.json")):
        r = json.loads(path.read_text())
        run = r["run"]
        row = {"run": run, "speed": float(run.split("_v_")[1].split("mps")[0]), "rep": run[-1]}
        for tag, key in (("sam2", "3d_sam2"), ("cot", "3d_cotracker")):
            m = r[key]
            row[f"valid_{tag}"] = m["valid_3d_frac"] * 100
            row[f"reproj_{tag}"] = m["reproj_err_px"]["median"]
            row[f"jitter_{tag}"] = m["jitter_2nd_diff_mm"]["median"]
            row[f"wobble_{tag}"] = m["edge_len_temporal_std_mm"]["median"]
            row[f"wobblemax_{tag}"] = m["edge_len_temporal_std_mm"]["max"]
        row["dist2d_med"] = (r["2d_left"]["dist_px"]["median"] + r["2d_right"]["dist_px"]["median"]) / 2
        row["far_markers"] = sum(1 for s in ("2d_left", "2d_right") for v in r[s]["worst_markers"].values() if v > 100)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["speed", "rep"])
    df.to_csv(ROOT / "work" / "cotracker3" / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
