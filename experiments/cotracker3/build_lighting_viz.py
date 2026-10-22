"""Render the lighting-test results (whatever has finished so far) into one HTML viewer.

For each finished condition and level it re-applies the same degradation to the left
camera, draws CoTracker3's output coloured by its distance from the clean reference,
and encodes a short clip; the page shows a condition-by-severity grid of the scores.

    .venv-cotracker/Scripts/python experiments/cotracker3/build_lighting_viz.py
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lighting_test as lt  # noqa: E402
import track_cotracker as tc  # noqa: E402

ROOT = tc.ROOT
RUN = "run_006_v_0.3mps_a"
CROP = (slice(700, 3900), slice(1500, 5050))  # rows, cols: the marked grid in the left view
CLIP_W = 560
OK, OFF, WRONG = (90, 210, 90), (43, 154, 243), (79, 90, 255)


def clip(kind: str, level: str, frames: list, ref: np.ndarray, out_csv: Path) -> bytes:
    df = pd.read_csv(out_csv).sort_values(["frame", "obj_id"])
    xy = df[["u", "v"]].to_numpy().reshape(ref.shape)
    hidden = (df["visible"].to_numpy() == 0).reshape(ref.shape[:2])
    err = np.linalg.norm(xy - ref, axis=2)
    near = np.argmin(np.linalg.norm(ref[:, None] - xy[:, :, None], axis=3), axis=2)
    wrong = near != np.arange(lt.N)
    cond = lt.Condition(kind, level, cv2.imread(str(frames[0])).shape[:2])
    y0, x0 = CROP[0].start, CROP[1].start
    s = CLIP_W / (CROP[1].stop - CROP[1].start)
    with tempfile.TemporaryDirectory() as tmp:
        for t, p in enumerate(frames):
            img = cond(cv2.imread(str(p)), t)[CROP]
            img = cv2.resize(img, (CLIP_W, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            for i in range(lt.N):
                q = (int((xy[t, i, 0] - x0) * s), int((xy[t, i, 1] - y0) * s))
                c = WRONG if wrong[t, i] else (OFF if err[t, i] > 10 else OK)
                if hidden[t, i]:
                    cv2.circle(img, q, 6, c, 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, q, 3, c, -1, cv2.LINE_AA)
            cv2.imwrite(str(Path(tmp) / f"{t:06d}.png"), img)
        mp4 = Path(tmp) / "c.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(Path(tmp) / "%06d.png"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(mp4)], check=True)
        return mp4.read_bytes()


def main() -> int:
    run_dir = ROOT / "work" / "trial15_baseline" / RUN
    start, end, _ = tc.load_prompts(run_dir, "left")
    frames = sorted((tc.FRAME_CACHE / f"trial15left_{start}_{end}").glob("*.jpg"))
    ref = (pd.read_csv(ROOT / "work" / "cotracker3" / RUN / lt.REF_VARIANT / "left" / "tracks_2d.csv")
           .sort_values(["frame", "obj_id"])[["u", "v"]].to_numpy().reshape(len(frames), lt.N, 2))
    root = ROOT / "work" / "cotracker3" / "lighting" / RUN
    order = {(k, l): i for i, (k, l) in enumerate((k, l) for k in lt.CONDITIONS for l in lt.LEVELS)}
    results = sorted((json.loads(f.read_text()) for f in root.glob("*/*/result.json")), key=lambda r: order[(r["kind"], r["level"])])
    clean = json.loads((ROOT / "work" / "cotracker3" / RUN / lt.REF_VARIANT / "compare.json").read_text())["3d_cotracker"]
    for r in results:
        r["video"] = "data:video/mp4;base64," + base64.b64encode(clip(r["kind"], r["level"], frames, ref, root / r["kind"] / r["level"] / "left.csv")).decode()
        print(f"[OK] {r['kind']} {r['level']}")
    data = {"run": RUN, "conditions": [k for k in lt.CONDITIONS], "labels": lt.LABELS, "levels": list(lt.LEVELS),
            "results": results, "clean": clean}
    html = (Path(__file__).resolve().parent / "lighting_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "work" / "cotracker3" / "viz" / "lighting_lab.html"
    out.write_text(html, encoding="utf-8")
    print(f"[OK] {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
