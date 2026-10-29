"""Run the paper's SAM2 tracker on the lighting-degraded clips (for a head-to-head with CoTracker3).

For each run / condition / strength it renders the degraded left and right clips with the same
Condition code as lighting_test.py, encodes them as near-lossless H.264, and calls the paper's
run_sam2_objectwise.py with the queue settings used for trial 15 (SAM2.1 Hiera-L, scale 0.25,
batch 36, dual GPU, setup prompts unchanged). Condition "clean" goes through the same path with no
degradation, as the encode baseline. Tracks land in work/cotracker3/sam2_lighting/<run>/<kind>/<level>/.

Runs on the GPU box (Linux) from the cotracker3 workspace root:
    SAM2_REPO=~/net-deformation-tracking python experiments/cotracker3/sam2_lighting.py --levels severe
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lighting_test as lt  # noqa: E402
import track_cotracker as tc  # noqa: E402

ROOT = tc.ROOT
RUNS = ["run_006_v_0.3mps_a", "run_003_v_0.15mps_a"]
KINDS = ["clean", "dark", "bright", "uneven", "flicker", "caustics", "murky"]


def write_clip(frames: list[Path], kind: str, level: str, out: Path) -> None:
    """Pipe degraded frames into ffmpeg as near-lossless H.264 (CRF 10)."""
    h, w = cv2.imread(str(frames[0])).shape[:2]
    cond = None if kind == "clean" else lt.Condition(kind, level, (h, w))
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", "29.97", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "10",
         "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for t, p in enumerate(frames):
        img = cv2.imread(str(p))
        if cond is not None:
            img = cond(img, t)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed for {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--kinds", nargs="*", default=KINDS)
    ap.add_argument("--levels", nargs="*", default=list(lt.LEVELS))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    repo = Path(os.environ.get("SAM2_REPO", "~/net-deformation-tracking")).expanduser()
    script = repo / "sam2" / "sam2" / "run_sam2_objectwise.py"

    for level in args.levels:
        for run in args.runs:
            run_dir = ROOT / "work" / "trial15_baseline" / run
            start, end, _ = tc.load_prompts(run_dir, "left")
            frames = {s: sorted((tc.FRAME_CACHE / f"trial15{s}_{start}_{end}").glob("*.jpg")) for s in ("left", "right")}
            for kind in args.kinds:
                lvl = "none" if kind == "clean" else level
                out = ROOT / "work" / "cotracker3" / "sam2_lighting" / run / kind / lvl
                if (out / "left" / "tracks_2d.csv").exists() and (out / "right" / "tracks_2d.csv").exists():
                    print(f"[skip] {run} {kind} {lvl}", flush=True)
                    continue
                t0 = time.perf_counter()
                clips = out / "clips"
                clips.mkdir(parents=True, exist_ok=True)
                for side in ("left", "right"):
                    write_clip(frames[side], kind, lvl, clips / f"{side}.mp4")
                cmd = [args.python, str(script), "--left-input", str(clips / "left.mp4"), "--right-input", str(clips / "right.mp4"),
                       "--start-frame", "0", "--end-frame", str(end - start),
                       "--setup-json", str(run_dir / "setup" / "prompts" / "points_left_right.json"),
                       "--out", str(out), "--scale", "0.25", "--model-id", "facebook/sam2.1-hiera-large",
                       "--gpu-mode", "dual", "--batch-size", "36", "--preview", "false"]
                with open(out / "objectwise.log", "w") as log:
                    rc = subprocess.run(cmd, cwd=script.parent, stdout=log, stderr=subprocess.STDOUT).returncode
                shutil.rmtree(clips, ignore_errors=True)
                for side in ("left", "right"):
                    shutil.rmtree(out / side / "objectwise_frames", ignore_errors=True)
                status = "DONE" if rc == 0 and (out / "left" / "tracks_2d.csv").exists() else f"FAILED rc={rc}"
                print(f"[{status}] {run} {kind} {lvl} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
