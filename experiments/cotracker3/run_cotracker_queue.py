"""Process a prepared pipeline queue with CoTracker3 instead of SAM2.

Use it in place of ``run_pipeline_queue.py --process-only``. Make the queue and do the
interactive setup with ``run_pipeline_queue.py`` as usual (splitter, calibration sync,
velocity labels, point setup). Then, on the GPU box:

    python experiments/cotracker3/run_cotracker_queue.py --resume <queue dir>

For each run with a finished setup it tracks both cameras with track_cotracker.py
(snap on, hidden points not snapped) and triangulates them with the queue's own
triangulation stage, so viewer and summary are the same as for SAM2. Calibration
stats/mono/stereo are computed first if the queue has not done them yet.

The queue script is imported, not modified (``--queue-script``, default the one at the
repo root), so the queue's path handling and checks apply. SAM2 outputs and SAM2 stage
state are left alone:
  tracks   <run dir>/cotracker3/{left,right}/tracks_2d.csv
  results  <SAM2 result dir>_cotracker3/   e.g. triangulation/results/v_0.3mps_a_cotracker3
A finished run is skipped on resume; changing the videos, setup or tracker settings re-runs it.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK_SCRIPT = HERE / "track_cotracker.py"
TRACKER_META = "queue_cotracker_meta.json"
TRACKER_VERSION = 1  # bump when track_cotracker.py output changes


def load_queue_module(path: Path):
    spec = importlib.util.spec_from_file_location("run_pipeline_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_queue(q, resume: str):
    if hasattr(q, "load_existing_queue"):  # remote-split queue: resolves portable paths and videos
        args = argparse.Namespace(resume=resume, check_only=False, sam2_scale=None, sam2_model_id=None)
        return q.load_existing_queue(args)
    manifest_path, manifest = q.load_manifest(resume)
    if q.migrate_manifest(manifest):
        q.save_manifest(manifest_path, manifest)
    return manifest_path, manifest


def prepare_calibration(q, manifest_path, manifest) -> bool:
    if len(inspect.signature(q.prepare_calibration).parameters) >= 3:
        return q.prepare_calibration(manifest_path, manifest, ("stats", "mono", "stereo"))
    return q.prepare_calibration(manifest_path, manifest)


def cotracker_view(job: dict, results_root: Path | None) -> dict:
    """The job as the queue's triangulation stage should see it: CoTracker3 tracks in, own result dir out."""
    view = copy.deepcopy(job)
    view["sam2_out"] = str(Path(job["run_dir"]) / "cotracker3")
    sam2_results = Path(job["result_dir"])
    view["result_dir"] = str((results_root or sam2_results.parent) / f"{sam2_results.name}_cotracker3")
    view["stages"] = {"triangulation": {"status": "pending"}}  # kept out of the manifest
    return view


def tracking_fingerprint(q, job: dict, side: str, settings: dict) -> str:
    return q.stable_hash({
        "version": TRACKER_VERSION,
        "side": side,
        "video": q.file_signature(job[f"{side}_video"]),
        "frame_range": q.frame_range_signature(job["start_frame"], job["end_frame"]),
        "setup_json": q.file_signature(job["setup_json"], include_hash=True),
        "settings": settings,
    })


def side_complete(q, manifest, job, view, side, settings) -> bool:
    out = Path(view["sam2_out"]) / side
    meta = out / TRACKER_META
    if not meta.is_file() or not (out / "tracks_2d.csv").is_file():
        return False
    try:
        if json.loads(meta.read_text()).get("fingerprint") != tracking_fingerprint(q, job, side, settings):
            return False
    except Exception:
        return False
    frames = int(job["end_frame"]) - int(job["start_frame"])
    return q.track_grid_is_complete(out / "tracks_2d.csv", frames, q.expected_object_ids(manifest, job))


def crop_applied(job: dict) -> bool:
    crops = json.loads(Path(job["setup_json"]).read_text()).get("crops", {})
    return any(c.get("crop_applied", False) for c in crops.values())


def run_tracking(q, manifest, job, view, args, settings) -> bool:
    for side in ("left", "right"):
        if side_complete(q, manifest, job, view, side, settings):
            print(f"[SKIP][COTRACKER3] {job['velocity']} {side} already complete.")
            continue
        out = Path(view["sam2_out"]) / side
        command = [
            args.python, str(TRACK_SCRIPT),
            "--run-dir", job["run_dir"], "--side", side, "--video", job[f"{side}_video"],
            "--model-res", settings["model_res"], "--refine", settings["refine"],
            "--refine-radius", str(settings["refine_radius"]),
            "--out", str(out), "--device", args.device, "--drop-frames",
        ]
        rc = q.run_logged(command, HERE.parents[1], Path(job["logs_dir"]) / f"cotracker3_{side}.log", use_pty=False)
        if rc != 0:
            print(f"[WARN] CoTracker3 {side} exited with code {rc}.")
            return False
        q.atomic_write_json(out / TRACKER_META, {
            "version": TRACKER_VERSION, "queue_id": manifest.get("queue_id"), "job_index": int(job["index"]),
            "velocity": job["velocity"], "fingerprint": tracking_fingerprint(q, job, side, settings),
            "written_at": q.utc_now(),
        })
        if not side_complete(q, manifest, job, view, side, settings):
            print(f"[WARN] CoTracker3 {side} tracks are incomplete: {out / 'tracks_2d.csv'}")
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resume", required=True, help="Queue directory or its queue_manifest.json")
    ap.add_argument("--queue-script", type=Path, default=HERE.parents[1] / "run_pipeline_queue.py",
                    help="The run_pipeline_queue.py whose paths and stages to use")
    ap.add_argument("--jobs", type=int, nargs="*", help="Only these job indices (default: all with a finished setup)")
    ap.add_argument("--results-root", type=Path, help="Folder for result dirs (default: next to the SAM2 results)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--refine-radius", type=int, default=40)
    ap.add_argument("--python", default=sys.executable, help="Python with torch for track_cotracker.py")
    args = ap.parse_args()

    if args.results_root:  # the queue's stages run from the queue script's repo, so no relative paths
        args.results_root = args.results_root.expanduser().resolve()
    q = load_queue_module(args.queue_script.expanduser().resolve())
    manifest_path, manifest = load_queue(q, args.resume)
    print(f"[QUEUE] {manifest_path}")
    if not prepare_calibration(q, manifest_path, manifest):
        print("[QUEUE] Calibration compute is incomplete. Resume this queue to try again.")
        return 1

    settings = {"tracker": "cotracker3_offline", "model_res": "384x512", "support_grid": 6,
                "refine": "dtpeak", "refine_radius": int(args.refine_radius)}
    done, failed = [], []
    for job in manifest["jobs"]:
        if args.jobs and int(job["index"]) not in args.jobs:
            continue
        print(f"\n[QUEUE] Run {job['index']}/{len(manifest['jobs'])}: {job['velocity']}")
        if not q.setup_complete(job, manifest):
            print("[SKIP] Setup is incomplete; finish it with run_pipeline_queue.py and resume.")
            failed.append(job["velocity"])
            continue
        if crop_applied(job):
            print("[SKIP] Setup used a crop; track_cotracker.py expects full-frame points.")
            failed.append(job["velocity"])
            continue
        view = cotracker_view(job, args.results_root)
        if not run_tracking(q, manifest, job, view, args, settings) or not q.run_triangulation(manifest_path, manifest, view):
            failed.append(job["velocity"])
            continue
        done.append(job["velocity"])
        print(f"[OK] Results: {view['result_dir']}")

    print(f"\n[QUEUE] CoTracker3 complete: {len(done)}, incomplete: {len(failed)} {failed if failed else ''}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
