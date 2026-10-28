"""Paper figures for the CoTracker3 results (PNG at 300 dpi + PDF).

Writes to experiments/cotracker3/results/figures/:
  fig_comparison_by_speed   SAM2 vs CoTracker3 across tow speeds (jitter, worst edge, 3D kept, reprojection)
  fig_occlusion_frames      the four occluder types: input beside CoTracker3 output with true positions
  fig_occlusion_by_speed    3D and 2D error on covered knots against tow speed, per setup
  fig_lighting_frames       clean + six lighting conditions (severe) with CoTracker3 output
  fig_lighting_by_strength  3D difference from clean and jitter against strength, per condition

    python experiments/cotracker3/paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lighting_test as lt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work" / "cotracker3"
RES = Path(__file__).resolve().parent / "results"
OUT = RES / "figures"
REF = "offline_384x512_dtpeak"
N = 144
C_SAM, C_COT = "#3a7bd5", "#2e9e5b"
GREEN, ORANGE, RED, WHITE = (90, 190, 70), (30, 150, 245), (60, 60, 230), (245, 245, 245)
LIGHT_KINDS = ["dark", "bright", "uneven", "flicker", "caustics", "murky"]
SETUP_ORDER = ["k1_short", "k1_long", "b2_long", "b3_long", "b4_long", "b3_rest", "static", "fish"]
SETUP_LABEL = {"k1_short": "1 knot, 0.5 s", "k1_long": "1 knot, 2 s", "b2_long": "2×2, 2 s", "b3_long": "3×3, 2 s",
               "b4_long": "4×4, 2 s", "b3_rest": "3×3, to end", "static": "fixed patch, 3 s", "fish": "passing fish, 2 s"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "legend.frameon": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}")


def save_img(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{name}.png"), img)
    print(f"[OK] {name}")


# ---------- comparison ----------
def fig_comparison():
    d = pd.read_csv(WORK / "summary.csv")
    d = d[d.run != "run_001_v_0.05mps_a"]
    g = d.groupby("speed").agg({**{c: "mean" for c in d.columns if c.endswith(("_sam2", "_cot")) and "wobblemax" not in c},
                                "wobblemax_sam2": "max", "wobblemax_cot": "max"})
    sp = g.index.to_numpy()
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))
    panels = [("jitter", "3D jitter, median (mm)", False), ("wobblemax", "Worst knot-to-knot error (mm)", True),
              ("valid", "3D points kept (%)", False), ("reproj", "Reprojection error, median (px)", False)]
    w = 0.018
    for a, (key, title, log) in zip(ax, panels):
        a.bar(sp - w / 2, g[f"{key}_sam2"], w, color=C_SAM, label="SAM2")
        a.bar(sp + w / 2, g[f"{key}_cot"], w, color=C_COT, label="CoTracker3")
        a.set_title(title, fontsize=9)
        a.set_xticks(sp, [f"{s:.2f}" for s in sp], fontsize=7.5)
        a.set_xlabel("Tow speed (m/s)")
        if log:
            a.set_yscale("log")
        if key == "valid":
            a.set_ylim(94, 100)
        a.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5)
    save(fig, "fig_comparison_by_speed")


# ---------- occlusion ----------
def tracks(csv, n_frames):
    df = pd.read_csv(csv).sort_values(["frame", "obj_id"])
    return df, df[["u", "v"]].to_numpy().reshape(n_frames, N, 2)


def frames_for(run, side="left"):
    p = json.loads((ROOT / "work" / "trial15_baseline" / run / "setup" / "prompts" / "points_left_right.json").read_text())
    s, e = p["frame_range"]["start_frame"], p["frame_range"]["end_frame"]
    return sorted((WORK / "frames" / f"trial15{side}_{s}_{e}").glob("*.jpg"))


def label(img, text, h=34):
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, h - 11), cv2.FONT_HERSHEY_DUPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def occ_pair(run, sid, t, title, width=520):
    frames = frames_for(run)
    d = WORK / "occlusion" / run / sid
    ref = tracks(WORK / run / REF / "left" / "tracks_2d.csv", len(frames))[1]
    df, xy = tracks(d / "left_gated.csv", len(frames))
    vis = df["visible"].to_numpy().reshape(len(frames), N).astype(bool)
    hidden = np.load(d / "hidden.npy")
    polys = {int(k): np.array(v, np.int32) for k, v in json.loads((d / "polygons.json").read_text()).items()}
    pitch = json.loads((d / "result.json").read_text())["pitch_px"]
    ever = hidden.any(0)
    pts = np.vstack(list(polys.values()))
    m = 1.5 * pitch
    x0, y0 = (pts.min(0) - m).astype(int)
    x1, y1 = (pts.max(0) + m).astype(int)
    cx, cy, w = (x0 + x1) / 2, (y0 + y1) / 2, min(max(x1 - x0, (y1 - y0) * 4 / 3), 2800)
    h = w * 3 / 4
    raw = cv2.convertScaleAbs(cv2.imread(str(frames[t])), alpha=1.7, beta=4)
    H, Wd = raw.shape[:2]
    x0, y0 = int(np.clip(cx - w / 2, 0, Wd - w)), int(np.clip(cy - h / 2, 0, H - h))
    w, h = int(w), int(h)
    s = width / w
    near = np.argmin(np.linalg.norm(ref[t][None] - xy[t][:, None], axis=2), axis=1)
    seen = raw.copy()
    if t in polys:
        cv2.fillPoly(seen, [polys[t]], (14, 13, 12), lineType=cv2.LINE_AA)
    left = cv2.resize(seen[y0:y0 + h, x0:x0 + w], (width, int(h * s)), interpolation=cv2.INTER_AREA)
    right = cv2.resize(raw[y0:y0 + h, x0:x0 + w], (width, int(h * s)), interpolation=cv2.INTER_AREA)
    if t in polys:
        poly = ((polys[t] - [x0, y0]) * s).astype(np.int32)
        shade = right.copy()
        cv2.fillPoly(shade, [poly], (14, 13, 12), lineType=cv2.LINE_AA)
        right = cv2.addWeighted(shade, 0.55, right, 0.45, 0)
        cv2.polylines(right, [poly], True, (170, 170, 170), 1, cv2.LINE_AA)
    to = lambda p: tuple(((p - [x0, y0]) * s).astype(int))
    for i in range(N):
        c = RED if near[i] != i else (GREEN if vis[t, i] else ORANGE)
        if ever[i]:
            cv2.circle(right, to(ref[t, i]), 11, WHITE, 2, cv2.LINE_AA)
            if np.linalg.norm(ref[t, i] - xy[t, i]) * s > 11:
                cv2.line(right, to(ref[t, i]), to(xy[t, i]), c, 2, cv2.LINE_AA)
        cv2.circle(right, to(xy[t, i]), 5 if ever[i] else 3, c, -1, cv2.LINE_AA)
    gap = np.full((left.shape[0], 6, 3), 255, np.uint8)
    return label(np.hstack([left, gap, right]), title)


def fig_occlusion_frames():
    run = "run_006_v_0.3mps_a"
    rows = [occ_pair(run, "fish", 60, "(a) Passing fish, 2 s"), occ_pair(run, "b4_long", 70, "(b) 4x4 block moving with the net, 2 s"),
            occ_pair(run, "static", 70, "(c) Fixed patch, 3 s"), occ_pair(run, "b3_rest", 150, "(d) 3x3 block hidden to the end of the clip")]
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 8, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    save_img(np.vstack(rows), "fig_occlusion_frames")


def fig_occlusion_speed():
    d = pd.read_csv(RES / "occlusion_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    cmap = plt.get_cmap("tab10")
    for k, sid in enumerate(SETUP_ORDER):
        s = d[d.setup_id == sid].sort_values("tow_speed_mps")
        style = dict(color=cmap(k), marker="o", ms=3.5, lw=1.4, label=SETUP_LABEL[sid])
        ax[0].plot(s.tow_speed_mps, s.err3d_median_mm, **style)
        ax[1].plot(s.tow_speed_mps, s.err2d_median_px, **style)
    ax[0].axhline(4.88, color="#888", lw=0.9, ls="--", label="FEM agreement (4.88 mm RMSE)")
    ax[0].set_ylabel("3D error, median (mm)"); ax[0].set_title("Covered knots, 3D", fontsize=9)
    ax[1].set_ylabel("2D error, median (px)"); ax[1].set_title("Covered knots, left image", fontsize=9)
    for a in ax:
        a.set_xlabel("Tow speed (m/s)")
        a.set_xticks(sorted(d.tow_speed_mps.unique()))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7.5)
    save(fig, "fig_occlusion_by_speed")


# ---------- lighting ----------
def fig_lighting_frames():
    run = "run_006_v_0.3mps_a"
    frames = frames_for(run)
    ref = tracks(WORK / run / REF / "left" / "tracks_2d.csv", len(frames))[1]
    t, (x, y, w, h) = 110, (1900, 1100, 2800, 2100)
    tile_w = 430
    s = tile_w / w
    src = cv2.imread(str(frames[t]))
    tiles = []
    for kind in [None] + LIGHT_KINDS:
        if kind is None:
            img, xy, vis = src.copy(), ref[t], np.ones(N, bool)
            title = "Clean"
        else:
            img = lt.Condition(kind, "severe", src.shape[:2])(src.copy(), t)
            df, arr = tracks(WORK / "lighting" / run / kind / "severe" / "left.csv", len(frames))
            xy, vis = arr[t], df["visible"].to_numpy().reshape(len(frames), N)[t].astype(bool)
            title = f"{lt.LABELS[kind]} (severe)"
        tile = cv2.resize(img[y:y + h, x:x + w], (tile_w, int(h * s)), interpolation=cv2.INTER_AREA)
        near = np.argmin(np.linalg.norm(ref[t][None] - xy[:, None], axis=2), axis=1)
        err = np.linalg.norm(xy - ref[t], axis=1)
        for i in range(N):
            q = (int((xy[i, 0] - x) * s), int((xy[i, 1] - y) * s))
            if 0 <= q[0] < tile.shape[1] and 0 <= q[1] < tile.shape[0]:
                c = RED if near[i] != i else (ORANGE if err[i] > 10 else GREEN)
                cv2.circle(tile, q, 4, c, -1 if vis[i] else 2, cv2.LINE_AA)
        tiles.append(label(tile, title))
    key = np.full_like(tiles[0], 255)
    for k, (txt, c) in enumerate([("CoTracker3 point, within 10 px of clean", GREEN), ("more than 10 px from clean", ORANGE),
                                  ("on the wrong knot", RED)]):
        yy = 90 + 45 * k
        cv2.circle(key, (30, yy - 6), 7, c, -1, cv2.LINE_AA)
        cv2.putText(key, txt, (48, yy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(key, "Left camera, run 006 (0.30 m/s), frame 110", (14, 260), cv2.FONT_HERSHEY_DUPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    tiles.append(key)
    gv = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    rows = [np.hstack(sum(([tl, gv] for tl in tiles[r * 4:(r + 1) * 4]), [])[:-1]) for r in range(2)]
    gh = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    save_img(np.vstack([rows[0], gh, rows[1]]), "fig_lighting_frames")


def fig_lighting_strength():
    d = pd.read_csv(RES / "lighting_results.csv")
    lv = {"mild": 0, "moderate": 1, "severe": 2}
    d["x"] = d.strength.map(lv)
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 5.4), sharex=True)
    cmap = plt.get_cmap("tab10")
    for col, speed in enumerate([0.30, 0.15]):
        for k, kind in enumerate(LIGHT_KINDS):
            s = d[(d.condition == kind) & (np.isclose(d.tow_speed_mps, speed))].sort_values("x")
            style = dict(color=cmap(k), marker="o", ms=3.5, lw=1.4, label=lt.LABELS[kind])
            ax[0, col].plot(s.x, s.err3d_vs_clean_median_mm, **style)
            ax[1, col].plot(s.x, s.jitter_median_mm, **style)
        ax[0, col].set_title(f"{speed:.2f} m/s", fontsize=9)
        ax[1, col].axhline(0.59 if speed == 0.30 else 0.56, color="#888", lw=0.9, ls="--", label="clean recording")
        ax[1, col].set_xticks([0, 1, 2], ["mild", "moderate", "severe"])
    ax[0, 0].set_ylabel("3D difference from clean,\nmedian (mm)")
    ax[1, 0].set_ylabel("3D jitter, median (mm)")
    h1, l1 = ax[0, 0].get_legend_handles_labels()
    h2, l2 = ax[1, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h1 + h2[-1:], l1 + l2[-1:], loc="lower center", ncol=7, fontsize=7.5)
    save(fig, "fig_lighting_by_strength")


def main() -> int:
    if sys.argv[1:] == ["charts"]:
        fig_comparison(); fig_occlusion_speed(); fig_lighting_strength()
        return 0
    fig_comparison()
    fig_occlusion_speed()
    fig_lighting_strength()
    fig_occlusion_frames()
    fig_lighting_frames()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
