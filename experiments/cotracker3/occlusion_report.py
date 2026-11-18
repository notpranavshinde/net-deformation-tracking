"""Record how CoTracker3 handled hidden knots in one result: which knots it called hidden,
for how long, how close its guess was when they came back, and what that did in 3D.

The model gives every knot a visibility score each frame; below 0.9 it is called hidden
and the knot keeps the model's own guessed position (method ``cotracker_inferred``),
while visible knots are snapped to the marker. Written to ``<result>/occlusion/``:

  knot_frames.csv        one row per stereo observation (as in triangulated_3d.csv) with
                         both cameras' visible flag, visibility score, confidence and method,
                         next to the 3D reprojection error
  hidden_stretches.csv   one row per run of consecutive hidden frames of one knot in one
                         camera: when, how long, the reappearance gap and the 3D error
  summary.json           the totals: hidden share by camera and row, stretch lengths,
                         reappearance gaps, 3D error with both / one / no camera seeing the knot
  occlusion_timeline.png hidden knot-frames over time, both cameras on the stereo frame axis

Reappearance gap: the distance from the model's guess on the last hidden frame to the snapped
marker on the next frame (knots normally move a few px per frame; ``normal_step_px`` in the
summary gives the scale). The snap only searches a square of +-``refine_radius`` px around the
model's own estimate, so ``model_miss_px`` (model estimate vs snapped marker on the reappearance
frame) is limited by it; stretches where either axis is near that limit are flagged
``at_search_limit`` and may have missed by more.

Runs by itself on the queue's run_cotracker_queue.py output; to backfill a result:

    python experiments/cotracker3/occlusion_report.py triangulation/results/<run>_cotracker3 \\
        --tracks work/pipeline_queue/<queue>/<run dir>/cotracker3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

GRID = 12
SIDES = {"left": "L", "right": "R"}
AROUND = 10  # frames before/after a stretch used for the both-visible comparison


def load_tracks(tracks_dir: Path, side: str) -> tuple[pd.DataFrame, dict]:
    t = pd.read_csv(tracks_dir / side / "tracks_2d.csv")
    meta_path = tracks_dir / side / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return t.sort_values(["obj_id", "frame"]).reset_index(drop=True), meta


def stretches(t: pd.DataFrame, side: str, first_source: int, radius: float) -> list[dict]:
    rows = []
    for oid, x in t.groupby("obj_id"):
        f = x.frame.to_numpy()
        vis = x.visible.to_numpy()
        u, v = x.u.to_numpy(), x.v.to_numpy()
        uc, vc = x.u_coarse.to_numpy(), x.v_coarse.to_numpy()
        snapped = (x.method == "cotracker+dtpeak").to_numpy()
        i = 0
        while i < len(vis):
            if vis[i]:
                i += 1
                continue
            j = i
            while j < len(vis) and not vis[j]:
                j += 1
            back = j < len(vis)
            gap = miss = np.nan
            limit = False
            if back and snapped[j]:
                gap = float(np.hypot(u[j] - u[j - 1], v[j] - v[j - 1]))
                miss = float(np.hypot(u[j] - uc[j], v[j] - vc[j]))
                limit = max(abs(u[j] - uc[j]), abs(v[j] - vc[j])) >= 0.9 * radius
            rows.append({
                "camera": side, "obj_id": int(oid), "row": int(oid) // GRID, "col": int(oid) % GRID,
                "first_frame": int(f[i]), "last_frame": int(f[j - 1]), "n_frames": int(j - i),
                "first_source_frame": first_source + int(f[i]), "last_source_frame": first_source + int(f[j - 1]),
                "from_clip_start": bool(i == 0), "reappeared": bool(back),
                "reappear_gap_px": gap, "model_miss_px": miss,
                "at_search_limit": bool(limit),
            })
            i = j
    return rows


def stats(x) -> dict:
    x = pd.Series(x, dtype=float).dropna()
    if x.empty:
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(x.median()), 2),
            "p90": round(float(x.quantile(0.9)), 2), "max": round(float(x.max()), 2)}


def err_stats(x: pd.DataFrame) -> dict:
    out = {"n": int(len(x))}
    if len(x):
        out.update({"reproj_err_median_px": round(float(x.err_px.median()), 2),
                    "reproj_err_p95_px": round(float(x.err_px.quantile(0.95)), 2),
                    "valid_3d_pct": round(100 * float(x.valid_3d.mean()), 1)})
    return out


def write_report(result_dir: Path, tracks_dir: Path) -> Path:
    result_dir, tracks_dir = Path(result_dir), Path(tracks_dir)
    out = result_dir / "occlusion"
    out.mkdir(parents=True, exist_ok=True)
    tri = pd.read_csv(result_dir / "triangulated_3d.csv",
                      usecols=["gframe", "frame_L", "frame_R", "obj_id", "err_px", "valid_3d"])

    knot = tri.copy()
    knot.insert(2, "row", knot.obj_id // GRID)
    knot.insert(3, "col", knot.obj_id % GRID)
    all_stretches, cameras, tracks = [], {}, {}
    for side, s in SIDES.items():
        t, meta = load_tracks(tracks_dir, side)
        tracks[side] = t
        radius = float(meta.get("refine_radius", 40))
        first_source = int(meta.get("frame_range", [0])[0])
        cols = t[["frame", "obj_id", "visible", "visibility", "confidence", "method"]].rename(columns={
            "frame": f"frame_{s}", "visible": f"visible_{s}", "visibility": f"visibility_{s}",
            "confidence": f"confidence_{s}", "method": f"method_{s}"})
        knot = knot.merge(cols, on=[f"frame_{s}", "obj_id"], how="left")
        st = stretches(t, side, first_source, radius)
        all_stretches += st
        steps = t.groupby("obj_id")[["u", "v"]].diff()
        both_vis = (t.visible == 1) & (t.groupby("obj_id").visible.shift() == 1)
        hidden = t.visible == 0
        cameras[side] = {
            "knot_frames": int(len(t)),
            "hidden_pct": round(100 * float(hidden.mean()), 2),
            "hidden_pct_by_row": [round(100 * float(hidden[t.obj_id // GRID == r].mean()), 2) for r in range(GRID)],
            "knots_ever_hidden": int(t[hidden].obj_id.nunique()),
            "stretches": len(st),
            "stretches_3plus_frames": int(sum(x["n_frames"] >= 3 for x in st)),
            "stretch_frames": stats([x["n_frames"] for x in st]),
            "reappear_gap_px": stats([x["reappear_gap_px"] for x in st]),
            "model_miss_px": stats([x["model_miss_px"] for x in st]),
            "stretches_at_search_limit": int(sum(x["at_search_limit"] for x in st)),
            "normal_step_px": stats(np.hypot(steps.u, steps.v)[both_vis]),
            "refine_radius_px": radius,
            "first_source_frame": first_source,
            "track_frame_minus_gframe": int((tri[f"frame_{s}"] - tri.gframe).median()),
        }
    knot.to_csv(out / "knot_frames.csv", index=False)

    # 3D error during each stretch, and for the same knot just before/after with both cameras seeing it
    keyed = knot.set_index(["obj_id"])
    for st in all_stretches:
        s = SIDES[st["camera"]]
        k = keyed.loc[[st["obj_id"]]]
        f = k[f"frame_{s}"]
        during = k[(f >= st["first_frame"]) & (f <= st["last_frame"])]
        near = k[((f >= st["first_frame"] - AROUND) & (f < st["first_frame"])) |
                 ((f > st["last_frame"]) & (f <= st["last_frame"] + AROUND))]
        near = near[(near.visible_L == 1) & (near.visible_R == 1)]
        st["err_px_median_hidden"] = float(during.err_px.median()) if len(during) else np.nan
        st["err_px_median_nearby_both_visible"] = float(near.err_px.median()) if len(near) else np.nan
    pd.DataFrame(all_stretches, columns=[
        "camera", "obj_id", "row", "col", "first_frame", "last_frame", "n_frames", "first_source_frame",
        "last_source_frame", "from_clip_start", "reappeared", "reappear_gap_px", "model_miss_px",
        "at_search_limit", "err_px_median_hidden", "err_px_median_nearby_both_visible",
    ]).to_csv(out / "hidden_stretches.csv", index=False)

    vl, vr = knot.visible_L == 1, knot.visible_R == 1
    summary = {
        "result": result_dir.name,
        "visibility_threshold": 0.9,
        "hidden_means": "model visibility < 0.9; the knot keeps the model's guess (not snapped)",
        "frames_note": "first/last_frame are track frames per camera; source frames are video frame numbers",
        "cameras": cameras,
        "three_d_by_visibility": {
            "both_visible": err_stats(knot[vl & vr]),
            "left_hidden_only": err_stats(knot[~vl & vr]),
            "right_hidden_only": err_stats(knot[vl & ~vr]),
            "both_hidden": err_stats(knot[~vl & ~vr]),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=False)
    cmap = ListedColormap(["#ececec", "#2a78d6"])
    for ax, side in zip(axes, SIDES):
        t = tracks[side]
        hid = (t.pivot(index="obj_id", columns="frame", values="visible") == 0).reindex(range(GRID * GRID))
        shift = cameras[side]["track_frame_minus_gframe"]  # plot on the stereo frame axis so both panels line up
        x0, x1 = hid.columns.min() - shift - 0.5, hid.columns.max() - shift + 0.5
        ax.imshow(hid.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1,
                  extent=(x0, x1, GRID * GRID - 0.5, -0.5))
        ax.set_xlim(tri.gframe.min() - 0.5, tri.gframe.max() + 0.5)
        for r in range(1, GRID):
            ax.axhline(r * GRID - 0.5, color="#ffffff", lw=0.6)
        ax.set_yticks([r * GRID + GRID / 2 - 0.5 for r in range(GRID)], [f"row {r}" for r in range(GRID)], fontsize=7)
        c = cameras[side]
        ax.set_title(f"{side} camera: hidden (blue) {c['hidden_pct']:.1f}% of knot-frames, "
                     f"{c['stretches']} hidden stretches", fontsize=10, loc="left")
        ax.set_xlabel(f"stereo frame (gframe); {side} video frame = gframe + "
                      f"{c['first_source_frame'] + c['track_frame_minus_gframe']}")
    fig.suptitle(f"CoTracker3 visibility, {result_dir.name}", fontsize=11)
    plt.tight_layout()
    fig.savefig(out / "occlusion_timeline.png", dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result", type=Path, help="Result folder with triangulated_3d.csv")
    ap.add_argument("--tracks", type=Path, required=True, help="The run's cotracker3 folder (with left/ and right/)")
    args = ap.parse_args()
    out = write_report(args.result, args.tracks)
    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
