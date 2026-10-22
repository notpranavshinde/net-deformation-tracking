"""Render lighting-test media for the weekly deck (severe level of every condition, run 006).

Outputs in work/cotracker3/deck/media/:
  light_<kind>.mp4/.png   side by side: degraded input | CoTracker3 output coloured by error
  light_montage.mp4/.png  clean + all seven conditions, 4 x 2, playing together
  light_snap.png          one knot under each condition with the snap's colour mask outlined
Clips are 15 fps from 29.97 fps footage, i.e. half speed.

    python experiments/cotracker3/make_lighting_media.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lighting_test as lt  # noqa: E402
import track_cotracker as tc  # noqa: E402

ROOT = tc.ROOT
RUN = "run_006_v_0.3mps_a"
OUT = ROOT / "work" / "cotracker3" / "deck" / "media"
LROOT = ROOT / "work" / "cotracker3" / "lighting" / RUN
KINDS = list(lt.CONDITIONS)
# 4:3 box around the middle of the grid (left camera), full-resolution pixels.
BOX = (1900, 1100, 2800, 2100)  # x, y, w, h
PANEL_W, TILE_W = 720, 470
OK, OFF, WRONG, WHITE, INK = (122, 210, 90), (43, 154, 243), (79, 90, 255), (240, 240, 236), (22, 20, 13)
FONT = cv2.FONT_HERSHEY_DUPLEX


def frames_and_ref():
    start, end, _ = tc.load_prompts(ROOT / "work" / "trial15_baseline" / RUN, "left")
    frames = sorted((tc.FRAME_CACHE / f"trial15left_{start}_{end}").glob("*.jpg"))
    ref = (pd.read_csv(ROOT / "work" / "cotracker3" / RUN / lt.REF_VARIANT / "left" / "tracks_2d.csv")
           .sort_values(["frame", "obj_id"])[["u", "v"]].to_numpy().reshape(len(frames), lt.N, 2))
    return frames, ref


def header(img, text, color, h=40):
    bar = np.full((h, img.shape[1], 3), INK, np.uint8)
    cv2.circle(bar, (18, h // 2), 7, color, -1, cv2.LINE_AA)
    cv2.putText(bar, text, (34, h // 2 + 8), FONT, 0.65, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def stamp(img, text, scale=0.5):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = 10, img.shape[0] - 10
    cv2.rectangle(img, (x - 5, y - h - 7), (x + w + 5, y + 5), INK, -1)
    cv2.putText(img, text, (x, y), FONT, scale, WHITE, 1, cv2.LINE_AA)


def overlay(img, xy, err, wrong, hidden, s, r=4):
    x0, y0 = BOX[:2]
    for i in range(lt.N):
        q = (int((xy[i, 0] - x0) * s), int((xy[i, 1] - y0) * s))
        if not (0 <= q[0] < img.shape[1] and 0 <= q[1] < img.shape[0]):
            continue
        c = WRONG if wrong[i] else (OFF if err[i] > 10 else OK)
        if hidden[i]:
            cv2.circle(img, q, r + 3, c, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, q, r, c, -1, cv2.LINE_AA)


def crop(img, width):
    x, y, w, h = BOX
    return cv2.resize(img[y:y + h, x:x + w], (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def encode(dir_: Path, name: str, poster: int) -> None:
    first = cv2.imread(str(dir_ / f"{poster:06d}.png"))
    cv2.imwrite(str(OUT / f"{name}.png"), first)
    cv2.imwrite(str(OUT / f"{name}_cover.png"), cv2.resize(first, (1000, int(first.shape[0] * 1000 / first.shape[1])), interpolation=cv2.INTER_AREA))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(dir_ / "%06d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
                    "-movflags", "+faststart", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(OUT / f"{name}.mp4")], check=True)
    print(f"[OK] {name}.mp4 {(OUT / f'{name}.mp4').stat().st_size / 1e6:.1f} MB", flush=True)


def tracks_for(kind: str | None, ref: np.ndarray):
    if kind is None:  # clean: the reference itself
        return ref, np.zeros(ref.shape[:2]), np.zeros(ref.shape[:2], bool), np.zeros(ref.shape[:2], bool)
    df = pd.read_csv(LROOT / kind / "severe" / "left.csv").sort_values(["frame", "obj_id"])
    xy = df[["u", "v"]].to_numpy().reshape(ref.shape)
    err = np.linalg.norm(xy - ref, axis=2)
    near = np.argmin(np.linalg.norm(ref[:, None] - xy[:, :, None], axis=3), axis=2)
    return xy, err, near != np.arange(lt.N), (df["visible"].to_numpy() == 0).reshape(ref.shape[:2])


def render_condition(kind: str | None) -> str:
    """Write montage tiles for one condition (and the side-by-side clip for real conditions)."""
    frames, ref = frames_and_ref()
    xy, err, wrong, hidden = tracks_for(kind, ref)
    cond = lt.Condition(kind, "severe", cv2.imread(str(frames[0])).shape[:2]) if kind else None
    tile_dir = OUT / "_tiles" / (kind or "clean")
    tile_dir.mkdir(parents=True, exist_ok=True)
    label = lt.LABELS[kind] if kind else "Clean recording"
    with tempfile.TemporaryDirectory() as tmp:
        for t, p in enumerate(frames):
            img = cv2.imread(str(p))
            if cond:
                img = cond(img, t)
            st = TILE_W / BOX[2]
            tile = crop(img, TILE_W)
            overlay(tile, xy[t], err[t], wrong[t], hidden[t], st, r=3)
            cv2.imwrite(str(tile_dir / f"{t:06d}.png"), header(tile, label, OK if kind is None else (150, 150, 150), h=34))
            if kind is None:
                continue
            sp = PANEL_W / BOX[2]
            left, right = crop(img, PANEL_W), crop(img, PANEL_W)
            overlay(right, xy[t], err[t], wrong[t], hidden[t], sp)
            e = err[t]
            stamp(left, f"frame {t:3d}  |  {label.lower()}, severe")
            stamp(right, f"median {np.median(e):4.1f} px from the clean run  |  {int(wrong[t].sum())} on wrong knot")
            gap = np.full((left.shape[0] + 40, 8, 3), INK, np.uint8)
            frame = np.hstack([header(left, f"Input: {label.lower()}", (150, 150, 150)), gap, header(right, "CoTracker3 output", OK)])
            cv2.imwrite(str(Path(tmp) / f"{t:06d}.png"), frame)
        if kind is not None:
            encode(Path(tmp), f"light_{kind}", 110)
    return kind or "clean"


def montage() -> None:
    names = ["clean"] + KINDS
    frames, _ = frames_and_ref()
    with tempfile.TemporaryDirectory() as tmp:
        for t in range(len(frames)):
            tiles = [cv2.imread(str(OUT / "_tiles" / n / f"{t:06d}.png")) for n in names]
            gap_v = np.full((tiles[0].shape[0], 6, 3), INK, np.uint8)
            rows = [np.hstack(sum(([tile, gap_v] for tile in tiles[r * 4:(r + 1) * 4]), [])[:-1]) for r in range(2)]
            gap_h = np.full((6, rows[0].shape[1], 3), INK, np.uint8)
            cv2.imwrite(str(Path(tmp) / f"{t:06d}.png"), np.vstack([rows[0], gap_h, rows[1]]))
        encode(Path(tmp), "light_montage", 110)


def snap_figure() -> None:
    """One knot, 5 x 5 pitch-free crop, under each condition, with the snap's Otsu mask outlined."""
    frames, ref = frames_and_ref()
    t, i = 110, 66
    src = cv2.imread(str(frames[t]))
    u, v = ref[t, i]
    tiles = []
    for kind in [None, "dark", "bright", "murky", "caustics"]:
        img = lt.Condition(kind, "severe", src.shape[:2])(src.copy(), t) if kind else src.copy()
        r = 40
        win = img[int(v) - r:int(v) + r + 1, int(u) - r:int(u) + r + 1]
        score = np.clip(win[..., 2].astype(np.int16) - win[..., 0], 0, 255).astype(np.uint8)
        big = cv2.resize(win, (300, 300), interpolation=cv2.INTER_NEAREST)
        if score.max() >= 30:
            _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(cv2.resize(mask, (300, 300), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(big, cnts, -1, (255, 255, 0), 2, cv2.LINE_AA)
            note = "snap finds a blob"
        else:
            note = "too little orange: no snap"
        cv2.drawMarker(big, (150, 150), WHITE, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        stamp(big, note, 0.45)
        tiles.append(header(big, lt.LABELS[kind] if kind else "Clean", OK if kind is None else (150, 150, 150), h=34))
    gap = np.full((tiles[0].shape[0], 8, 3), INK, np.uint8)
    cv2.imwrite(str(OUT / "light_snap.png"), np.hstack(sum(([tl, gap] for tl in tiles), [])[:-1]))
    print("[OK] light_snap.png", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    snap_figure()
    with Pool(8) as pool:
        for name in pool.imap_unordered(render_condition, [None] + KINDS):
            print(f"[tiles] {name}", flush=True)
    montage()
    m = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else {}
    for name in ["light_montage"] + [f"light_{k}" for k in KINDS]:
        im = cv2.imread(str(OUT / f"{name}.png"))
        m[name] = {"w": im.shape[1], "h": im.shape[0]}
    (OUT / "manifest.json").write_text(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
