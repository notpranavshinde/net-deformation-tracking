"""Export the occlusion and lighting test results to flat CSVs for the paper.

Writes experiments/cotracker3/results/occlusion_results.csv and lighting_results.csv
(one row per case) from the per-case result.json files under work/cotracker3/.
The non-black background lighting condition is left out (flawed software background).

    python experiments/cotracker3/export_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work" / "cotracker3"
OUT = Path(__file__).resolve().parent / "results"
SPEED = {"run_007_v_0.05mps_b": 0.05, "run_002_v_0.1mps_a": 0.10, "run_003_v_0.15mps_a": 0.15,
         "run_004_v_0.2mps_a": 0.20, "run_005_v_0.25mps_a": 0.25, "run_006_v_0.3mps_a": 0.30}
LEVELS = ["mild", "moderate", "severe"]


def occlusion() -> pd.DataFrame:
    rows = []
    for run, speed in SPEED.items():
        for r in json.loads((WORK / "occlusion" / run / "results.json").read_text()):
            h, d, rc = r["hidden_err_px"]["raw"], r["hidden_3d_err_mm"], r["recovery_err_px"] or {}
            rows.append({
                "run": run, "tow_speed_mps": speed, "setup_id": r["id"], "setup": r["scenario"], "kind": r["kind"],
                "frame_start": r["frames"][0], "frame_end": r["frames"][1], "knots_covered": r["knots_hidden"],
                "knot_frames_covered": r["knot_frames_hidden"], "pitch_px": r["pitch_px"],
                "err2d_median_px": h["median"], "err2d_p95_px": h["p95"], "err2d_max_px": h["max"],
                "err2d_median_pitch": r["hidden_err_pitch"]["raw"]["median"],
                "wrong_knot_frac": r["hidden_wrong_knot"]["gated_snap"],
                "flagged_hidden_frac": r["flagged_hidden"], "false_hidden_uncovered_frac": r["false_hidden_on_untouched"],
                "recovery_err_median_px": rc.get("median"), "recovery_err_max_px": rc.get("max"),
                "recovery_wrong_knot_frac": r["recovery_wrong_knot"],
                "err3d_median_mm": d.get("median"), "err3d_p95_mm": d.get("p95"), "err3d_max_mm": d.get("max"),
                "rejected3d_frac": r["hidden_3d_rejected"],
            })
    return pd.DataFrame(rows)


def lighting() -> pd.DataFrame:
    rows = []
    for f in sorted((WORK / "lighting").glob("*/*/*/result.json")):
        r = json.loads(f.read_text())
        if r["kind"] == "background":
            continue
        run = f.parts[-4]
        L, R, t = r["2d_left"], r["2d_right"], r["3d"]
        rows.append({
            "run": run, "tow_speed_mps": SPEED[run], "condition": r["kind"], "label": r["label"], "strength": r["level"],
            "params": json.dumps(r["params"]),
            "snap_shift_median_px_left": L["snap_err_px"]["median"], "snap_shift_median_px_right": R["snap_err_px"]["median"],
            "snap_shift_p95_px_left": L["snap_err_px"]["p95"], "raw_err_median_px_left": L["raw_err_px"]["median"],
            "wrong_knot_frac_left": L["wrong_knot"], "wrong_knot_frac_right": R["wrong_knot"],
            "snap_found_frac_left": L["snapped"], "snap_found_frac_right": R["snapped"],
            "flagged_hidden_frac_left": L["flagged_hidden"], "flagged_hidden_frac_right": R["flagged_hidden"],
            "err3d_vs_clean_median_mm": t["err_vs_clean_mm"]["median"], "err3d_vs_clean_p95_mm": t["err_vs_clean_mm"]["p95"],
            "err3d_vs_clean_max_mm": t["err_vs_clean_mm"]["max"], "kept3d_frac": t["valid_3d_frac"],
            "jitter_median_mm": t["jitter_2nd_diff_mm"]["median"], "reproj_median_px": t["reproj_err_px"]["median"],
        })
    df = pd.DataFrame(rows)
    df["_s"] = df["strength"].map(LEVELS.index)
    return df.sort_values(["tow_speed_mps", "condition", "_s"], ascending=[False, True, True]).drop(columns="_s")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    occ, light = occlusion(), lighting()
    occ.to_csv(OUT / "occlusion_results.csv", index=False)
    light.to_csv(OUT / "lighting_results.csv", index=False)
    print(f"[OK] occlusion {len(occ)} rows, lighting {len(light)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
