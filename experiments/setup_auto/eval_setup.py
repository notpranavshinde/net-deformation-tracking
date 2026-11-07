"""Score ways of producing the 144 setup points against the hand-corrected setups we already have.

Reference: the final points_left_right.json of each run (what was accepted after manual review).
The semi-auto guesses before review are not saved (semi_auto_meta.json is rewritten with the final
points), only which points were hand-moved ("edited"), so the current method is not re-scored here.
Methods, all on the setup frame of each camera:
  reuse_raw      the previous run's final points, unchanged
  reuse_check    the previous run's final points as hints to nettrack.bootstrap_mesh
  grow           detections labelled by growing the grid from the 4 reviewed corners
  grow_check     the grown grid as hints to bootstrap_mesh (adds the left/right check)
A point is "wrong" when it is more than 0.35 x local knot spacing from the reference, which means
another knot or empty water. "Flagged" points are the ones the method itself marks for review.

    python experiments/setup_auto/eval_setup.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nettrack.bootstrap import bootstrap_mesh  # noqa: E402
from nettrack import grid_grow  # noqa: E402
from nettrack.geometry import load_stereo_calibration  # noqa: E402
from nettrack.topology import NetTopology  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
VIDEOS = ROOT / "raw videos"
GRID = 12
WRONG = 0.35
FRAMES = 7
T15 = ROOT / "work" / "trial15_baseline"
WAVE = ROOT / "work" / "pipeline_queue" / "20261106_052812_545745"


def runs() -> list[dict]:
    """Run dirs in session order, with the run whose final points get reused."""
    t15 = sorted(p for p in T15.glob("run_*") if (p / "setup" / "prompts" / "points_left_right.json").exists())
    out = []
    for i, run in enumerate(t15):
        prev = t15[i - 1] if i else t15[1]  # the first run borrows from the second
        out.append({"run": run, "prev": prev, "stereo": T15 / "calibration" / "stereo.npz",
                    "videos": ("trial15left.MP4", "trial15right.MP4")})
    wave = next(WAVE.glob("run_*"))
    out.append({"run": wave, "prev": t15[-1], "stereo": WAVE / "calibration" / "stereo.npz",
                "videos": ("testWaveTrial2Left.MP4", "testWaveTrial2Right.MP4")})
    return out


def final_points(run: Path) -> tuple[np.ndarray, np.ndarray, int]:
    m = json.loads((run / "setup" / "prompts" / "points_left_right.json").read_text())
    pts = [np.array([[p["x"], p["y"]] for p in m[s]], float) for s in ("left", "right")]
    return pts[0], pts[1], int(m["frame_range"]["start_frame"])


def semi_auto(run: Path):
    m = json.loads((run / "setup" / "prompts" / "semi_auto_debug" / "semi_auto_meta.json").read_text())
    out = {}
    for s in ("left", "right"):
        pts = sorted(m[s]["points"], key=lambda p: p["obj_id"])
        out[s] = np.array([[p["x"], p["y"]] for p in pts], float)
        out[s + "_corners"] = np.array([[p["x"], p["y"]] for p in pts if p["source"] == "anchor"], float)
        out[s + "_lattice"] = np.array([p["source"] == "lattice" for p in pts])  # gap-filled guess, accepted unchecked
    out["edited"] = sum(p["source"] == "edited" for s in ("left", "right") for p in m[s]["points"])
    return out


def flat_grid(corners: np.ndarray) -> np.ndarray:
    return grid_grow.flat_grid(corners, GRID, GRID)


def orange_seeds(frame: np.ndarray, hints: np.ndarray) -> np.ndarray:
    return grid_grow.orange_seeds(frame, hints, float(np.median(local_spacing(hints))))


def grow(frame: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return grid_grow.grow_grid(grid_grow.detect(frame, corners, GRID, GRID), corners, GRID, GRID)


def read_frames(video: Path, start: int, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = [cap.read()[1] for _ in range(n)]
    cap.release()
    return frames


def local_spacing(pts: np.ndarray) -> np.ndarray:
    g = pts.reshape(GRID, GRID, 2)
    d = np.full((GRID, GRID), np.nan)
    for r in range(GRID):
        for c in range(GRID):
            nb = [g[r + dr, c + dc] for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)) if 0 <= r + dr < GRID and 0 <= c + dc < GRID]
            d[r, c] = np.median([np.linalg.norm(g[r, c] - q) for q in nb])
    return d.reshape(-1)


def check(frames_l, frames_r, hints_l, hints_r, calibration, color_points=None):
    """bootstrap_mesh output as points plus a review flag (absent or moved far from its hint)."""
    res = bootstrap_mesh(frames_l, frames_r, hints_l, hints_r, NetTopology.from_grid(GRID, GRID), calibration,
                         setup_check_frames=FRAMES, color_points=color_points)
    nodes = sorted(res.report["nodes"], key=lambda n: n["obj_id"])
    pts, flag = {}, {}
    for s in ("left", "right"):
        pts[s] = np.array([n[s] if n[s] is not None else n[f"predicted_{s}"] for n in nodes], float)
        flag[s] = np.array([n["status"] == "absent" for n in nodes])
    return pts, flag, {k: sum(n["status"] == k for n in nodes) for k in ("confirmed", "repaired", "absent")}


def score(pts, flag, ref, spacing):
    d = np.linalg.norm(pts - ref, axis=1)
    wrong = d > WRONG * spacing
    return {"wrong": int(wrong.sum()), "flagged": int(flag.sum()), "silent_wrong": int((wrong & ~flag).sum()),
            "median_px": float(np.median(d[~wrong])) if (~wrong).any() else float("nan")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, detail = [], {}
    for job in runs():
        run = job["run"]
        name = run.parent.name + "/" + run.name if run.parent != T15 else "trial15/" + run.name
        ref_l, ref_r, start = final_points(run)
        prev_l, prev_r, _ = final_points(job["prev"])
        auto = semi_auto(run)
        calib = load_stereo_calibration(job["stereo"])
        fl = read_frames(VIDEOS / job["videos"][0], start, FRAMES)
        fr = read_frames(VIDEOS / job["videos"][1], start, FRAMES)
        ref = {"left": ref_l, "right": ref_r}
        spacing = {s: local_spacing(ref[s]) for s in ref}
        t0 = time.perf_counter()
        grown = {}
        for s, frame in (("left", fl[0]), ("right", fr[0])):
            grown[s] = grow(frame, auto[s + "_corners"])
        seeds = lambda pl, pr: (orange_seeds(fl[0], pl), orange_seeds(fr[0], pr))  # noqa: E731
        methods = {
            "reuse_raw": ({"left": prev_l, "right": prev_r}, {s: np.zeros(144, bool) for s in ref}, None),
            "reuse_check": check(fl, fr, prev_l, prev_r, calib, color_points=seeds(prev_l, prev_r)),
            "grow": ({s: grown[s][0] for s in ref}, {s: grown[s][1] for s in ref}, None),
            "grow_check": check(fl, fr, grown["left"][0], grown["right"][0], calib,
                                color_points=(grown["left"][0][~grown["left"][1]], grown["right"][0][~grown["right"][1]])),
        }
        secs = time.perf_counter() - t0
        detail[name] = {}
        for method, (pts, flag, status) in methods.items():
            row = {"run": name, "method": method, "hand_fixed_in_reference": auto["edited"]}
            for s in ref:
                sc = score(pts[s], flag[s], ref[s], spacing[s])
                row.update({f"{k}_{s}": v for k, v in sc.items()})
            row["wrong"] = row["wrong_left"] + row["wrong_right"]
            row["flagged"] = row["flagged_left"] + row["flagged_right"]
            row["silent_wrong"] = row["silent_wrong_left"] + row["silent_wrong_right"]
            if status:
                row.update({f"bootstrap_{k}": v for k, v in status.items()})
            rows.append(row)
            detail[name][method] = {s: pts[s].tolist() for s in ref}
        print(f"[{name}] " + "  ".join(f"{r['method']}: wrong {r['wrong']}, silent {r['silent_wrong']}" for r in rows[-4:])
              + f"  ({secs:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "setup_methods.csv", index=False)
    (OUT / "setup_points.json").write_text(json.dumps(detail))
    print(df.groupby("method")[["wrong", "flagged", "silent_wrong"]].sum().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
