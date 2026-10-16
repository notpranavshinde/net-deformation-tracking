"""Render the comparison and occlusion videos (plus poster frames) for the CoTracker3 deck.

Every clip is encoded at 15 fps from 29.97 fps footage, so it plays at half speed.
Output: work/cotracker3/deck/media/<name>.mp4 and <name>.png

    python experiments/cotracker3/make_deck_media.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work" / "cotracker3"
OUT = WORK / "deck" / "media"
REF = "offline_384x512_dtpeak"
GRID = 12
N = GRID * GRID
FPS_OUT = 15

# BGR palette shared with the deck
SAM = (240, 201, 76)
COT = (122, 210, 90)
INFER = (43, 154, 243)
WRONG = (79, 90, 255)
WHITE = (240, 240, 236)
INK_BG = (22, 20, 13)
FONT = cv2.FONT_HERSHEY_DUPLEX


def run_paths(run: str):
    prompts = json.loads((ROOT / "work" / "trial15_baseline" / run / "setup" / "prompts" / "points_left_right.json").read_text())
    s, e = prompts["frame_range"]["start_frame"], prompts["frame_range"]["end_frame"]
    return sorted((WORK / "frames" / f"trial15left_{s}_{e}").glob("*.jpg"))


def track_array(csv: Path, n_frames: int, valid_only: bool = False) -> np.ndarray:
    df = pd.read_csv(csv)
    if valid_only:
        df = df[df["valid"] == 1]
    out = np.full((n_frames, N, 2), np.nan)
    out[df["frame"].to_numpy(), df["obj_id"].to_numpy()] = df[["u", "v"]].to_numpy()
    return out


def header(img: np.ndarray, text: str, color, h: int = 46) -> np.ndarray:
    bar = np.full((h, img.shape[1], 3), INK_BG, np.uint8)
    cv2.circle(bar, (22, h // 2), 8, color, -1, cv2.LINE_AA)
    cv2.putText(bar, text, (40, h // 2 + 9), FONT, 0.8, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def stamp(img: np.ndarray, text: str) -> None:
    (w, h), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    x, y = 12, img.shape[0] - 12
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + 6), INK_BG, -1)
    cv2.putText(img, text, (x, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)


def encode(frames: list[np.ndarray], name: str, poster_at: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{name}.png"), frames[poster_at])
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            cv2.imwrite(str(Path(tmp) / f"{i:06d}.png"), f)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS_OUT),
            "-i", str(Path(tmp) / "%06d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "23",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(OUT / f"{name}.mp4"),
        ], check=True)
    print(f"[OK] {name}.mp4 {(OUT / f'{name}.mp4').stat().st_size / 1e6:.1f} MB")


def wrong_ids(tracks: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per frame and knot: True where the track sits closer to another knot's true position."""
    d = np.linalg.norm(ref[:, None, :, :] - tracks[:, :, None, :], axis=3)
    d = np.where(np.isnan(d), np.inf, d)
    near = np.argmin(d, axis=2)
    return (near != np.arange(N)[None, :]) & ~np.isnan(tracks[..., 0])


def draw_grid(img, pts, scale, color, thick=1):
    for i in range(N):
        for j in ([i + 1] if i % GRID < GRID - 1 else []) + ([i + GRID] if i + GRID < N else []):
            if np.isfinite(pts[i]).all() and np.isfinite(pts[j]).all():
                cv2.line(img, tuple((pts[i] * scale).astype(int)), tuple((pts[j] * scale).astype(int)), color, thick, cv2.LINE_AA)


def clip_tracking(run: str = "run_006_v_0.3mps_a") -> None:
    frames = run_paths(run)
    cot = track_array(WORK / run / REF / "left" / "tracks_2d.csv", len(frames))
    x0, y0, x1, y1 = 1250, 0, 5300, 4150
    W = 960
    s = W / (x1 - x0)
    out = []
    for t, p in enumerate(frames):
        img = cv2.resize(cv2.imread(str(p))[y0:y1, x0:x1], (W, int((y1 - y0) * s)), interpolation=cv2.INTER_AREA)
        pts = cot[t] - [x0, y0]
        draw_grid(img, pts, s, (70, 120, 60), 1)
        for q in pts:
            cv2.circle(img, tuple((q * s).astype(int)), 4, COT, -1, cv2.LINE_AA)
        stamp(img, f"0.30 m/s tow  |  frame {t:3d}  |  half speed")
        out.append(header(img, "CoTracker3: all 144 knots, left camera", COT))
    encode(out, "cotracker_tracking", 110)


def clip_sam2_vs_cot(run: str = "run_006_v_0.3mps_a") -> None:
    frames = run_paths(run)
    sam = track_array(ROOT / "work" / "trial15_baseline" / run / "sam2" / "left" / "tracks_2d.csv", len(frames), valid_only=True)
    cot = track_array(WORK / run / REF / "left" / "tracks_2d.csv", len(frames))
    bad = wrong_ids(sam, cot)
    x0, y0, x1, y1 = 1250, 0, 5300, 4150
    W = 760
    s = W / (x1 - x0)
    out = []
    for t, p in enumerate(frames):
        base = cv2.resize(cv2.imread(str(p))[y0:y1, x0:x1], (W, int((y1 - y0) * s)), interpolation=cv2.INTER_AREA)
        left, right = base.copy(), base.copy()
        for i in range(N):
            if np.isfinite(sam[t, i]).all():
                q = tuple(((sam[t, i] - [x0, y0]) * s).astype(int))
                if bad[t, i]:
                    cv2.circle(left, q, 13, WRONG, 2, cv2.LINE_AA)
                    cv2.putText(left, str(i), (q[0] + 14, q[1] - 8), FONT, 0.5, WRONG, 1, cv2.LINE_AA)
                cv2.circle(left, q, 4, WRONG if bad[t, i] else SAM, -1, cv2.LINE_AA)
            q = tuple(((cot[t, i] - [x0, y0]) * s).astype(int))
            cv2.circle(right, q, 4, COT, -1, cv2.LINE_AA)
        nb = int(bad[t].sum()) + int(np.isnan(sam[t, :, 0]).sum())
        stamp(left, f"frame {t:3d}  |  {nb} knots lost or on the wrong knot")
        stamp(right, f"frame {t:3d}  |  0.30 m/s, half speed")
        gap = np.full((left.shape[0] + 46, 8, 3), INK_BG, np.uint8)
        out.append(np.hstack([header(left, "SAM2", SAM), gap, header(right, "CoTracker3", COT)]))
    encode(out, "sam2_vs_cotracker_full", 120)


def clip_identity_zoom(run: str = "run_006_v_0.3mps_a") -> None:
    frames = run_paths(run)
    sam = track_array(ROOT / "work" / "trial15_baseline" / run / "sam2" / "left" / "tracks_2d.csv", len(frames), valid_only=True)
    cot = track_array(WORK / run / REF / "left" / "tracks_2d.csv", len(frames))
    focus = [115, 125, 137]
    x0, y0, x1, y1 = 2300, 2300, 4000, 4300
    W = 640
    s = W / (x1 - x0)
    out = []
    for t, p in enumerate(frames):
        base = cv2.resize(cv2.imread(str(p))[y0:y1, x0:x1], (W, int((y1 - y0) * s)), interpolation=cv2.INTER_AREA)
        panels = []
        for tracks, color, name in ((sam, SAM, "SAM2"), (cot, COT, "CoTracker3")):
            img = base.copy()
            for i in focus:
                if not np.isfinite(tracks[t, i]).all():
                    continue
                q = tuple(((tracks[t, i] - [x0, y0]) * s).astype(int))
                if 0 <= q[0] < W and 0 <= q[1] < img.shape[0]:
                    cv2.circle(img, q, 18, color, 3, cv2.LINE_AA)
                    cv2.putText(img, str(i), (q[0] + 20, q[1] - 14), FONT, 0.8, color, 2, cv2.LINE_AA)
            lost = [i for i in focus if not np.isfinite(tracks[t, i]).all()]
            off = [i for i in focus if np.isfinite(tracks[t, i]).all() and not (x0 <= tracks[t, i, 0] < x1 and y0 <= tracks[t, i, 1] < y1)]
            note = "  ".join([f"{i}: lost" for i in lost] + [f"{i}: off-screen" for i in off])
            stamp(img, f"frame {t:3d}  {note}")
            panels.append(header(img, f"{name}: knots 115, 125, 137", color))
        gap = np.full((panels[0].shape[0], 8, 3), INK_BG, np.uint8)
        out.append(np.hstack([panels[0], gap, panels[1]]))
    encode(out, "identity_zoom", 125)


def clip_nets3d(run: str = "run_006_v_0.3mps_a") -> None:
    def load(csv):
        tri = pd.read_csv(csv)
        tri = tri[tri["valid_3d"] == 1]
        frames = sorted(tri["gframe"].unique())
        a = np.full((len(frames), N, 3), np.nan)
        idx = {g: k for k, g in enumerate(frames)}
        a[[idx[g] for g in tri["gframe"]], tri["obj_id"].to_numpy()] = tri[["X", "Y", "Z"]].to_numpy() * 1000
        return a

    sam, cot = load(WORK / run / "sam2_tri" / "tri.csv"), load(WORK / run / REF / "tri" / "tri.csv")
    T = min(len(sam), len(cot))
    edges = [(i, i + 1) for i in range(N) if i % GRID < GRID - 1] + [(i, i + GRID) for i in range(N - GRID)]
    centre = np.nanmedian(cot.reshape(-1, 3), axis=0)  # middle of the whole motion, so the net stays in frame

    def lengths(a):
        return np.stack([np.linalg.norm(a[:, i] - a[:, j], axis=1) for i, j in edges], 1)

    ref_len = {k: np.nanmedian(lengths(a), 0) for k, a in (("sam", sam), ("cot", cot))}
    S = 760

    def project(p, yaw, pitch=0.35):
        x, y, z = (p - centre).T
        y, z = -y, -z
        x1, z1 = np.cos(yaw) * x + np.sin(yaw) * z, -np.sin(yaw) * x + np.cos(yaw) * z
        y2 = np.cos(pitch) * y - np.sin(pitch) * z1
        k = S * 0.0011
        return np.stack([S / 2 + x1 * k, S / 2 - y2 * k], 1)

    def col(err):
        e = min(abs(err), 10)
        stops = [(0, (112, 100, 79)), (1, (112, 100, 79)), (5, (43, 154, 243)), (10, (79, 90, 255))]
        for (e0, c0), (e1, c1) in zip(stops, stops[1:]):
            if e <= e1:
                f = (e - e0) / (e1 - e0 or 1)
                return tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))
        return stops[-1][1]

    out = []
    for t in range(T):
        yaw = -0.95 + 0.5 * np.sin(2 * np.pi * t / T)
        panels = []
        for key, a, color, name in (("sam", sam, SAM, "SAM2"), ("cot", cot, COT, "CoTracker3")):
            img = np.full((S, S, 3), (26, 23, 17), np.uint8)
            ghost = project(a[0], yaw)
            for i, j in edges:
                if np.isfinite(ghost[[i, j]]).all():
                    cv2.line(img, tuple(ghost[i].astype(int)), tuple(ghost[j].astype(int)), (60, 56, 46), 1, cv2.LINE_AA)
            pts = project(a[t], yaw)
            ln = lengths(a[t:t + 1])[0]
            order = np.argsort(np.nan_to_num(np.abs(ln - ref_len[key]), nan=-1))
            worst = 0.0
            for e in order:
                i, j = edges[e]
                if not np.isfinite(pts[[i, j]]).all():
                    continue
                err = abs(ln[e] - ref_len[key][e])
                worst = max(worst, err)
                cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), col(err), 3 if err > 5 else 2, cv2.LINE_AA)
            for q in pts:
                if np.isfinite(q).all() and 0 <= q[0] < S and 0 <= q[1] < S:
                    cv2.circle(img, tuple(q.astype(int)), 2, WHITE, -1, cv2.LINE_AA)
            stamp(img, f"worst edge error {worst:6.1f} mm")
            panels.append(header(img, f"{name}: 3D net, 0.30 m/s", color))
        gap = np.full((panels[0].shape[0], 8, 3), INK_BG, np.uint8)
        out.append(np.hstack([panels[0], gap, panels[1]]))
    encode(out, "nets3d", 59)  # SAM2 identity errors peak around frames 50-70


def clip_occlusion(run: str, sid: str, name: str, title: str) -> None:
    frames = run_paths(run)
    d = WORK / "occlusion" / run / sid
    ref = track_array(WORK / run / REF / "left" / "tracks_2d.csv", len(frames))
    g = pd.read_csv(d / "left_gated.csv").sort_values(["frame", "obj_id"])
    out_xy = g[["u", "v"]].to_numpy().reshape(len(frames), N, 2)
    vis = g["visible"].to_numpy().reshape(len(frames), N).astype(bool)
    hidden = np.load(d / "hidden.npy")
    polys = {int(k): np.array(v, np.int32) for k, v in json.loads((d / "polygons.json").read_text()).items()}
    res = json.loads((d / "result.json").read_text())
    wrong = wrong_ids(out_xy, ref)
    ever = hidden.any(0)
    pitch = res["pitch_px"]

    pts = np.vstack(list(polys.values()))
    m = 1.7 * pitch
    bx0, by0 = pts.min(0) - m
    bx1, by1 = pts.max(0) + m
    # Cap the width so knots stay visible; a passing fish only needs part of its path in view.
    cx, cy, w = (bx0 + bx1) / 2, (by0 + by1) / 2, min(max(bx1 - bx0, (by1 - by0) * 4 / 3), 2800)
    h = w * 3 / 4
    full = cv2.imread(str(frames[0])).shape[:2]
    x0, y0 = int(np.clip(cx - w / 2, 0, full[1] - w)), int(np.clip(cy - h / 2, 0, full[0] - h))
    w, h = int(w), int(h)
    W = 720
    s = W / w

    def to(p):
        return tuple(((p - [x0, y0]) * s).astype(int))

    out = []
    for t, p in enumerate(frames):
        raw = cv2.convertScaleAbs(cv2.imread(str(p)), alpha=1.7, beta=4)  # the tank footage is dark on a projector
        seen = raw.copy()
        if t in polys:
            cv2.fillPoly(seen, [polys[t]], (14, 13, 12), lineType=cv2.LINE_AA)
        left = cv2.resize(seen[y0:y0 + h, x0:x0 + w], (W, int(h * s)), interpolation=cv2.INTER_AREA)
        right = cv2.resize(raw[y0:y0 + h, x0:x0 + w], (W, int(h * s)), interpolation=cv2.INTER_AREA)
        if t in polys:
            shade = right.copy()
            poly = ((polys[t] - [x0, y0]) * s).astype(np.int32)
            cv2.fillPoly(shade, [poly], (14, 13, 12), lineType=cv2.LINE_AA)
            right = cv2.addWeighted(shade, 0.55, right, 0.45, 0)
            cv2.polylines(right, [poly], True, (150, 150, 150), 1, cv2.LINE_AA)
        for i in range(N):
            if ever[i]:
                cv2.circle(right, to(ref[t, i]), 13, WHITE, 2, cv2.LINE_AA)
            c = WRONG if wrong[t, i] else (COT if vis[t, i] else INFER)
            if ever[i] and np.linalg.norm(ref[t, i] - out_xy[t, i]) * s > 13:
                cv2.line(right, to(ref[t, i]), to(out_xy[t, i]), c, 2, cv2.LINE_AA)
            cv2.circle(right, to(out_xy[t, i]), 5 if ever[i] else 3, c, -1, cv2.LINE_AA)
        n_cov = int(hidden[t].sum())
        stamp(left, f"frame {t:3d}  |  {n_cov} knots covered")
        if n_cov:
            e = np.linalg.norm(out_xy[t, hidden[t]] - ref[t, hidden[t]], axis=1)
            stamp(right, f"covered now: median error {np.median(e):5.1f} px, worst {e.max():5.1f} px")
        else:
            e = np.linalg.norm(out_xy[t, ever] - ref[t, ever], axis=1)
            stamp(right, f"no knots covered  |  worst error on these knots {e.max():5.1f} px")
        gap = np.full((left.shape[0] + 46, 8, 3), INK_BG, np.uint8)
        out.append(np.hstack([header(left, f"Input: {title}", (150, 150, 150)), gap,
                              header(right, "CoTracker3 output (patch see-through)", COT)]))
    mid = (res["frames"][0] + min(res["frames"][1], len(frames) - 1)) // 2
    encode(out, name, mid)


OCCLUSION_CLIPS = [
    ("run_006_v_0.3mps_a", "fish", "occ_fish", "passing fish"),
    ("run_006_v_0.3mps_a", "b4_long", "occ_block4", "4x4 knots hidden 2 s"),
    ("run_006_v_0.3mps_a", "static", "occ_static", "fixed patch 3 s"),
    ("run_003_v_0.15mps_a", "b3_rest", "occ_rest_015", "3x3 hidden to end, 0.15 m/s"),
    ("run_006_v_0.3mps_a", "b3_rest", "occ_rest_030", "3x3 hidden to end, 0.30 m/s"),
]


def main() -> int:
    import sys
    if sys.argv[1:] == ["nets3d"]:
        clip_nets3d()
        return 0
    if sys.argv[1:] == ["occlusion"]:
        for args in OCCLUSION_CLIPS:
            clip_occlusion(*args)
        return 0
    clip_tracking()
    clip_sam2_vs_cot()
    clip_identity_zoom()
    clip_nets3d()
    for args in OCCLUSION_CLIPS:
        clip_occlusion(*args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
