"""Synthetic occlusion test for CoTracker3 on real towing-tank footage.

Dark occluders are painted into the left-camera frames of an existing run, the tracker
is rerun on the edited video, and its output is scored against the reference tracks of
the same run without occlusion (CoTracker3 + full-resolution snap, verified visually).
The right camera keeps its reference tracks, as if only one lens were blocked.

Occluder kinds:
  tracked  covers a block of knots and moves with them (debris or fouling on the net)
  static   a fixed patch in the image the net slides under (a line in front of the lens)
  fish     a dark ellipse sweeping across the view between camera and net

For every scenario it reports, over the knot-frames that were hidden:
  2D error of the raw model output, the visibility-gated snap and an ungated snap,
  how often the model flagged them hidden, recovery once they reappear, and the
  3D error after triangulation against the reference right-camera tracks.

Run with the CoTracker environment; triangulation runs in the base ``python``::

    .venv-cotracker/Scripts/python experiments/cotracker3/occlusion_test.py --run run_006_v_0.3mps_a
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402
import track_cotracker as tc  # noqa: E402

ROOT = tc.ROOT
GRID = 12
REF_VARIANT = "offline_384x512_dtpeak"
FILL_BGR = (14, 13, 12)  # the black-lined tank
RECOVERY_FRAMES = 30


def block(r0: int, c0: int, n: int) -> list[int]:
    return [(r0 + r) * GRID + c0 + c for r in range(n) for c in range(n)]


SCENARIOS = [
    {"name": "1 knot, 0.5 s", "id": "k1_short", "kind": "tracked", "knots": [66], "frames": [40, 55]},
    {"name": "1 knot, 2 s", "id": "k1_long", "kind": "tracked", "knots": [66], "frames": [25, 85]},
    {"name": "2×2 knots, 2 s", "id": "b2_long", "kind": "tracked", "knots": block(5, 5, 2), "frames": [25, 85]},
    {"name": "3×3 knots, 2 s", "id": "b3_long", "kind": "tracked", "knots": block(4, 4, 3), "frames": [25, 85]},
    {"name": "4×4 knots, 2 s", "id": "b4_long", "kind": "tracked", "knots": block(4, 4, 4), "frames": [25, 85]},
    {"name": "3×3 knots, to end", "id": "b3_rest", "kind": "tracked", "knots": block(4, 4, 3), "frames": [25, None]},
    {"name": "Fixed patch, 3 s", "id": "static", "kind": "static", "knots": block(4, 4, 3), "frames": [25, 115]},
    {"name": "Passing fish, 2 s", "id": "fish", "kind": "fish", "row": 6, "frames": [30, 90]},
]


class Occlusion:
    """Builds the per-frame occluder polygon from the reference tracks and paints it."""

    def __init__(self, spec: dict, ref: np.ndarray, n_frames: int):
        self.spec, self.ref = spec, ref
        f0, f1 = spec["frames"]
        self.f0, self.f1 = f0, n_frames if f1 is None else f1
        # Grid pitch per frame from horizontally adjacent knots, used to size every occluder.
        right = [i for i in range(GRID * GRID) if i % GRID < GRID - 1]
        self.pitch = np.median(np.linalg.norm(ref[:, [i + 1 for i in right]] - ref[:, right], axis=2), axis=1)
        self.hidden = np.zeros(ref.shape[:2], dtype=bool)
        self.polys: dict[int, np.ndarray] = {}
        for t in range(self.f0, self.f1):
            poly = self._polygon(t)
            self.polys[t] = poly
            for i in range(ref.shape[1]):
                self.hidden[t, i] = cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, ref[t, i])), False) >= 0

    def _polygon(self, t: int) -> np.ndarray:
        s, pitch = self.spec, self.pitch[t]
        if s["kind"] in ("tracked", "static"):
            at = t if s["kind"] == "tracked" else (self.f0 + self.f1) // 2
            # Hull of a circle around every covered knot; its edge sits about halfway to the next knot.
            pad = 0.45 * self.pitch[at]
            ring = pad * np.stack([np.cos(np.linspace(0, 2 * np.pi, 24, endpoint=False)),
                                   np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False))], 1)
            pts = (self.ref[at, s["knots"]][:, None, :] + ring[None]).reshape(-1, 2)
            return cv2.convexHull(pts.astype(np.float32))[:, 0].astype(np.int32)
        # fish: an ellipse 2.5 x 1.3 pitches, crossing the grid left to right along one row.
        row = s["row"] * GRID
        x_left, x_right = self.ref[t, row, 0] - 2 * pitch, self.ref[t, row + GRID - 1, 0] + 2 * pitch
        y = self.ref[t, row:row + GRID, 1].mean()
        a = (t - self.f0) / max(1, self.f1 - 1 - self.f0)
        centre = (int(x_left + a * (x_right - x_left)), int(y))
        return cv2.ellipse2Poly(centre, (int(1.25 * pitch), int(0.65 * pitch)), -8, 0, 360, 8).astype(np.int32)

    def __call__(self, bgr: np.ndarray, t: int) -> None:
        poly = self.polys.get(t)
        if poly is not None:
            cv2.fillPoly(bgr, [poly], FILL_BGR, lineType=cv2.LINE_AA)


def errors(df: pd.DataFrame, ref: np.ndarray, ucol: str, vcol: str) -> np.ndarray:
    xy = df.sort_values(["frame", "obj_id"])[[ucol, vcol]].to_numpy().reshape(ref.shape)
    return np.linalg.norm(xy - ref, axis=2)


def stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {"median": round(float(np.median(x)), 2), "p95": round(float(np.percentile(x, 95)), 2), "max": round(float(x.max()), 2)}


def nearest_is_self(xy: np.ndarray, ref_t: np.ndarray, i: int) -> bool:
    return int(np.argmin(np.linalg.norm(ref_t - xy, axis=1))) == i


def score(spec: dict, occ: Occlusion, gated: pd.DataFrame, ungated: pd.DataFrame, ref: np.ndarray,
          tri: pd.DataFrame, ref_tri: pd.DataFrame) -> dict:
    H = occ.hidden
    pitch = occ.pitch[:, None] * np.ones_like(H, dtype=float)
    e_raw = errors(gated, ref, "u_coarse", "v_coarse")
    e_gate = errors(gated, ref, "u", "v")
    e_ungate = errors(ungated, ref, "u", "v")
    vis = gated.sort_values(["frame", "obj_id"])["visible"].to_numpy().reshape(H.shape).astype(bool)
    xy = gated.sort_values(["frame", "obj_id"])[["u", "v"]].to_numpy().reshape(ref.shape)

    # Wrong knot: the output sits closer to another knot's reference position than its own.
    def wrong_rate(e_xy: np.ndarray, mask: np.ndarray) -> float:
        ts, iz = np.nonzero(mask)
        if not len(ts):
            return float("nan")
        return float(np.mean([not nearest_is_self(e_xy[t, i], ref[t], i) for t, i in zip(ts, iz)]))

    xy_ungated = ungated.sort_values(["frame", "obj_id"])[["u", "v"]].to_numpy().reshape(ref.shape)
    ever = H.any(0)
    after = np.zeros_like(H)
    if occ.f1 < H.shape[0]:
        after[occ.f1:occ.f1 + RECOVERY_FRAMES, :] = ever
    never = ~ever[None, :].repeat(H.shape[0], 0)

    # 3D: error vs the unoccluded reference triangulation, in mm, for the hidden knot-frames.
    key = ["gframe", "obj_id"]
    m = tri.merge(ref_tri, on=key, suffixes=("", "_ref"))
    m = m[m["valid_3d_ref"] == 1]
    left_frame = m["frame_L"].to_numpy()
    in_h = H[np.clip(left_frame, 0, H.shape[0] - 1), m["obj_id"].to_numpy()] & (left_frame < H.shape[0])
    d3 = np.linalg.norm((m[["X", "Y", "Z"]].to_numpy() - m[["X_ref", "Y_ref", "Z_ref"]].to_numpy()) * 1000, axis=1)
    valid_h = m["valid_3d"].to_numpy()[in_h] == 1

    return {
        "scenario": spec["name"], "id": spec["id"], "kind": spec["kind"],
        "frames": [occ.f0, occ.f1], "knots_hidden": int(ever.sum()), "knot_frames_hidden": int(H.sum()),
        "pitch_px": round(float(np.median(occ.pitch[occ.f0:occ.f1])), 1),
        "hidden_err_px": {"raw": stats(e_raw[H]), "gated_snap": stats(e_gate[H]), "ungated_snap": stats(e_ungate[H])},
        "hidden_err_pitch": {"raw": stats((e_raw / pitch)[H]), "gated_snap": stats((e_gate / pitch)[H])},
        "hidden_wrong_knot": {"gated_snap": round(wrong_rate(xy, H), 4), "ungated_snap": round(wrong_rate(xy_ungated, H), 4)},
        "flagged_hidden": round(float((~vis[H]).mean()), 4) if H.any() else None,
        "false_hidden_on_untouched": round(float((~vis[never]).mean()), 4),
        "recovery_err_px": stats(e_gate[after]) if after.any() else None,
        "recovery_wrong_knot": round(wrong_rate(xy, after), 4) if after.any() else None,
        "untouched_err_px": stats(e_gate[never]),
        "hidden_3d_err_mm": stats(d3[in_h][valid_h]),
        "hidden_3d_rejected": round(float((~valid_h).mean()), 4) if in_h.any() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="run_006_v_0.3mps_a")
    ap.add_argument("--only", nargs="*", help="Scenario ids to run (default: all)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dir = ROOT / "work" / "trial15_baseline" / args.run
    ref_dir = ROOT / "work" / "cotracker3" / args.run / REF_VARIANT
    out_root = ROOT / "work" / "cotracker3" / "occlusion" / args.run
    start, end, points = tc.load_prompts(run_dir, "left")
    frames = tc.extract_frames(ROOT / "raw videos" / "trial15left.MP4", start, end,
                               tc.FRAME_CACHE / f"trial15left_{start}_{end}")
    ref_df = pd.read_csv(ref_dir / "left" / "tracks_2d.csv").sort_values(["frame", "obj_id"])
    ref = ref_df[["u", "v"]].to_numpy().reshape(len(frames), GRID * GRID, 2)
    ref_tri = pd.read_csv(ref_dir / "tri" / "tri.csv")
    model = tc.load_model(args.device)

    results = []
    for spec in SCENARIOS:
        if args.only and spec["id"] not in args.only:
            continue
        t0 = time.perf_counter()
        out = out_root / spec["id"]
        out.mkdir(parents=True, exist_ok=True)
        occ = Occlusion(spec, ref, len(frames))
        video, full_hw = tc.load_video_tensor(frames, (384, 512), occ)
        tracks, vis, conf = tc.track_points(model, video, points, full_hw, args.device)
        gated = pd.DataFrame(tc.build_rows(frames, tracks, vis, conf, full_hw, "dtpeak", 40, occ, gate=True))
        ungated = pd.DataFrame(tc.build_rows(frames, tracks, vis, conf, full_hw, "dtpeak", 40, occ, gate=False))
        gated.to_csv(out / "left_gated.csv", index=False)
        ungated.to_csv(out / "left_ungated.csv", index=False)
        tri_csv = compare.triangulate(run_dir, out / "left_gated.csv", ref_dir / "right" / "tracks_2d.csv", out / "tri", python="python")
        res = score(spec, occ, gated, ungated, ref, pd.read_csv(tri_csv), ref_tri)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        (out / "result.json").write_text(json.dumps(res, indent=2))
        np.save(out / "hidden.npy", occ.hidden)
        (out / "polygons.json").write_text(json.dumps({str(t): p.tolist() for t, p in occ.polys.items()}))
        results.append(res)
        print(f"[{spec['id']}] hidden={res['knot_frames_hidden']} raw={res['hidden_err_px']['raw'].get('median')}px "
              f"gated={res['hidden_err_px']['gated_snap'].get('median')}px ungated={res['hidden_err_px']['ungated_snap'].get('median')}px "
              f"flagged={res['flagged_hidden']} 3d={res['hidden_3d_err_mm'].get('median')}mm ({res['seconds']} s)", flush=True)
    (out_root / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
