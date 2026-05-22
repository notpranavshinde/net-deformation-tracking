r"""
Batched/objectwise SAM2 marker tracking wrapper.

Paste-ready commands from this directory:

    python .\run_sam2_objectwise.py

Run only a few IDs for testing:

    python .\run_sam2_objectwise.py --ids 0-2 --scale 0.25 --preview false

Run in memory-friendly batches of 24 objects:

    python .\run_sam2_objectwise.py --batch-size 24 --gpu-mode 4090-only --preview false

Useful full-frame test:

    python .\run_sam2_objectwise.py --left-crop none --right-crop none

This script reuses the setup from:

    work\manual_sam2_setup\prompts\points_left_right.json

Final merged outputs stay compatible with triangulation:

    out\left\tracks_2d.csv
    out\right\tracks_2d.csv
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch

import run_sam2_markers as sam2run


def parse_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y", "on")


def parse_ids(value, max_count):
    if value is None or str(value).strip().lower() in ("", "all"):
        return list(range(max_count))
    ids = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            ids.extend(range(start, end + step, step))
        else:
            ids.append(int(part))
    out = []
    seen = set()
    for obj_id in ids:
        if obj_id < 0 or obj_id >= max_count:
            raise ValueError(f"Object id {obj_id} is outside 0-{max_count - 1}")
        if obj_id not in seen:
            out.append(obj_id)
            seen.add(obj_id)
    return out


def chunk_ids(ids, batch_size):
    batch_size = max(1, int(batch_size))
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def stable_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_track_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def load_track_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_track_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (int(r["frame"]), int(r["obj_id"])))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sam2run.TRACK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def video_signature(video_path):
    path = Path(video_path)
    stat = path.stat()
    cap = sam2run.cv2.VideoCapture(str(path))
    frame_count = int(cap.get(sam2run.cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    width = int(cap.get(sam2run.cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else -1
    height = int(cap.get(sam2run.cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else -1
    fps = float(cap.get(sam2run.cv2.CAP_PROP_FPS)) if cap.isOpened() else -1.0
    cap.release()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM2 one marker object at a time, then merge tracks.")
    parser.add_argument("--left-input", default=str(Path("in") / "left.mp4"))
    parser.add_argument("--right-input", default=str(Path("in") / "right.mp4"))
    parser.add_argument("--out", default=str(sam2run.DEFAULT_OUT_DIR))
    parser.add_argument("--setup-json", default=str(sam2run.DEFAULT_LOCAL_SETUP_DIR / "prompts" / "points_left_right.json"))
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--ids", default=None, help="Object IDs to run, e.g. 0,4,8-12. Default: all.")
    parser.add_argument("--batch-size", type=int, default=1, help="Objects per SAM2 run. Try 24 on a 4090.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--left-crop", default=None, help="Optional x,y,w,h or none")
    parser.add_argument("--right-crop", default=None, help="Optional x,y,w,h or none")
    parser.add_argument("--frame-extractor", choices=["auto", "ffmpeg", "opencv"], default="auto")
    parser.add_argument("--preview", type=parse_bool, default=True)
    parser.add_argument("--preview-max-width", type=int, default=1400)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-mode", choices=["auto", "single", "dual", "4090-only"], default="auto")
    parser.add_argument(
        "--single-gpu-index",
        type=int,
        default=None,
        help="GPU index used when --gpu-mode=single. If omitted with multiple GPUs, prompt interactively.",
    )
    parser.add_argument("--offload-video-to-cpu", type=parse_bool, default=True)
    parser.add_argument("--offload-state-to-cpu", type=parse_bool, default=False)
    parser.add_argument("--async-loading-frames", type=parse_bool, default=False)
    parser.add_argument("--save-masks", type=parse_bool, default=False)
    parser.add_argument("--save-overlay", type=parse_bool, default=False)
    parser.add_argument(
        "--correction-restart-mode",
        choices=["full", "reseed-from-correction-frame"],
        default="full",
        help="Correction restart behavior passed through to run_sam2_markers.py.",
    )
    parser.add_argument(
        "--auto-pause-missing",
        type=parse_bool,
        default=False,
        help="Automatically pause preview when fewer valid markers are visible than expected.",
    )
    return parser.parse_args()


def choose_device(args, side_name=None):
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    mode = str(args.gpu_mode).lower()
    if mode == "dual":
        if gpu_count < 2:
            raise RuntimeError("--gpu-mode dual requires at least 2 CUDA GPUs")
        return "cuda:0" if side_name == "left" else "cuda:1"
    if mode == "4090-only":
        idx = next((i for i, name in enumerate(gpu_names) if "4090" in str(name).upper()), None)
        if idx is None:
            raise RuntimeError(f"--gpu-mode 4090-only requested, but no 4090 was found. GPUs: {gpu_names}")
        return f"cuda:{idx}"
    if mode == "single":
        if gpu_count <= 0:
            return "cpu"
        if args.single_gpu_index is None:
            if not hasattr(args, "_chosen_single_gpu_index"):
                args._chosen_single_gpu_index = sam2run.choose_single_gpu_index(gpu_names)
            idx = int(args._chosen_single_gpu_index)
        else:
            idx = int(args.single_gpu_index)
        if idx < 0 or idx >= gpu_count:
            raise RuntimeError(f"--single-gpu-index is outside 0-{gpu_count - 1}")
        return f"cuda:{idx}"
    return "cuda:0" if gpu_count > 0 else "cpu"


def load_setup(args):
    setup_path = Path(args.setup_json)
    if not setup_path.exists():
        raise FileNotFoundError(f"Missing setup JSON: {setup_path}. Run python .\\run_sam2_markers.py --setup first.")

    left_points = sam2run.load_points_file(str(setup_path), side="left", prefer_local=False)
    right_points = sam2run.load_points_file(str(setup_path), side="right", prefer_local=False)
    loaded_crop_left, loaded_crop_right = sam2run.load_crops_from_points_json(str(setup_path))

    left_forced_none = sam2run.is_crop_none_arg(args.left_crop)
    right_forced_none = sam2run.is_crop_none_arg(args.right_crop)
    crop_left = sam2run.parse_crop_arg(args.left_crop)
    crop_right = sam2run.parse_crop_arg(args.right_crop)

    if crop_left is None and not left_forced_none:
        crop_left = loaded_crop_left
    if crop_right is None and not right_forced_none:
        crop_right = loaded_crop_right

    if left_forced_none and loaded_crop_left is not None:
        left_points = sam2run.offset_points(left_points, loaded_crop_left)
        print(f"[INFO] LEFT crop disabled; shifted saved points by x={loaded_crop_left[0]}, y={loaded_crop_left[1]}")
    if right_forced_none and loaded_crop_right is not None:
        right_points = sam2run.offset_points(right_points, loaded_crop_right)
        print(f"[INFO] RIGHT crop disabled; shifted saved points by x={loaded_crop_right[0]}, y={loaded_crop_right[1]}")

    corrections_path = sam2run.DEFAULT_CORRECTIONS_PATH
    corrections = sam2run.load_corrections(corrections_path)
    if left_forced_none or right_forced_none:
        corrections_path = sam2run.NO_CROP_CORRECTIONS_PATH
        if corrections_path.exists():
            corrections = sam2run.load_corrections(corrections_path)
        else:
            corrections = sam2run.offset_corrections_payload(
                corrections,
                {
                    "left": loaded_crop_left if left_forced_none else None,
                    "right": loaded_crop_right if right_forced_none else None,
                },
            )
    return {
        "left": {"points": left_points, "crop": crop_left, "video": args.left_input},
        "right": {"points": right_points, "crop": crop_right, "video": args.right_input},
        "corrections": corrections,
        "corrections_path": corrections_path,
    }


def count_cached_frames(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    return sum(1 for _ in frames_dir.glob("*.jpg"))


def frame_cache_status(frames_dir: Path, expected_count: int):
    cached_count = count_cached_frames(frames_dir)
    if cached_count <= 0:
        return False, cached_count, "no JPEG frames found"
    if expected_count <= 0:
        return True, cached_count, "OpenCV could not report the video frame count"
    first_frame = frames_dir / "000000.jpg"
    last_frame = frames_dir / f"{expected_count - 1:06d}.jpg"
    if cached_count != expected_count:
        return False, cached_count, f"expected {expected_count} JPEG frames, found {cached_count}"
    if not first_frame.exists():
        return False, cached_count, f"missing first frame {first_frame.name}"
    if not last_frame.exists():
        return False, cached_count, f"missing last frame {last_frame.name}"
    return True, cached_count, ""


def prepare_frames(side_name, video_path, crop, args, side_out: Path):
    frames_dir = side_out / "objectwise_frames"
    meta_path = side_out / "frames_objectwise_meta.json"
    if args.overwrite and frames_dir.exists():
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if meta_path.exists():
            meta_path.unlink()
    frames_dir, cached_count = sam2run.prepare_frame_cache(
        video_path,
        frames_dir,
        crop=crop,
        scale=args.scale,
        frame_extractor=args.frame_extractor,
        meta_name="frames_objectwise_meta.json",
        label=side_name.upper(),
    )
    return frames_dir, cached_count


def object_fingerprint(side_name, obj_id, point, crop, video_path, args, corrections):
    obj_corrections = [
        corr for corr in sam2run.corrections_for_side(corrections, side_name)
        if int(corr.get("obj_id", -1)) == int(obj_id)
    ]
    return stable_hash({
        "version": 1,
        "side": side_name,
        "obj_id": int(obj_id),
    "video": video_signature(video_path),
        "crop": list(crop) if crop is not None else None,
        "scale": float(args.scale),
        "point": point,
        "corrections": obj_corrections,
        "model_id": sam2run.MODEL_ID,
    })


def batch_fingerprint(side_name, batch_ids, side_data, args, corrections):
    batch_id_set = {int(obj_id) for obj_id in batch_ids}
    batch_corrections = [
        corr for corr in sam2run.corrections_for_side(corrections, side_name)
        if int(corr.get("obj_id", -1)) in batch_id_set
    ]
    return stable_hash({
        "version": 2,
        "side": side_name,
        "object_ids": [int(obj_id) for obj_id in batch_ids],
        "video": video_signature(side_data["video"]),
        "crop": list(side_data["crop"]) if side_data["crop"] is not None else None,
        "scale": float(args.scale),
        "points": [side_data["points"][obj_id] for obj_id in batch_ids],
        "corrections": batch_corrections,
        "model_id": sam2run.MODEL_ID,
        "batch_size": int(args.batch_size),
    })


def object_is_complete(obj_dir: Path, fingerprint: str, frame_count: int):
    meta_path = obj_dir / "objectwise_meta.json"
    tracks_path = obj_dir / "tracks_2d.csv"
    if not meta_path.exists() or not tracks_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if meta.get("fingerprint") != fingerprint or meta.get("status") != "ok":
        return False
    return count_track_rows(tracks_path) >= int(frame_count)


def batch_is_complete(batch_dir: Path, fingerprint: str, frame_count: int, batch_count: int):
    meta_path = batch_dir / "batch_meta.json"
    tracks_path = batch_dir / "tracks_2d.csv"
    if not meta_path.exists() or not tracks_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if meta.get("fingerprint") != fingerprint or meta.get("status") != "ok":
        return False
    return count_track_rows(tracks_path) >= int(frame_count) * int(batch_count)


def run_side(side_name, side_data, ids, args, out_root: Path, corrections, corrections_path: Path):
    side_out = out_root / side_name
    side_out.mkdir(parents=True, exist_ok=True)
    objectwise_root = side_out / "objectwise"
    objectwise_root.mkdir(parents=True, exist_ok=True)

    frames_dir, frame_count = prepare_frames(side_name, side_data["video"], side_data["crop"], args, side_out)
    device = choose_device(args, side_name=side_name)
    batches = chunk_ids(ids, args.batch_size)
    print(
        f"[INFO][{side_name.upper()}] device={device} objects={len(ids)} "
        f"batch_size={args.batch_size} batches={len(batches)} frames={frame_count}"
    )

    if device.startswith("cuda"):
        dev_idx = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.set_device(dev_idx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from sam2.sam2_video_predictor import SAM2VideoPredictor

    predictor = SAM2VideoPredictor.from_pretrained(sam2run.MODEL_ID).to(device)
    shared_state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
        async_loading_frames=args.async_loading_frames,
    )

    results = []
    try:
        for batch_index, batch_ids in enumerate(batches, start=1):
            batch_name = f"batch_{batch_ids[0]:03d}_{batch_ids[-1]:03d}"
            batch_dir = objectwise_root / batch_name
            batch_points = [side_data["points"][obj_id] for obj_id in batch_ids]
            fingerprint = batch_fingerprint(side_name, batch_ids, side_data, args, corrections)
            if not args.overwrite and batch_is_complete(batch_dir, fingerprint, frame_count, len(batch_ids)):
                print(
                    f"[SKIP][{side_name.upper()}] {batch_name} "
                    f"({batch_index}/{len(batches)}) ids={batch_ids[0]}-{batch_ids[-1]}"
                )
                results.append({
                    "batch": batch_name,
                    "object_ids": batch_ids,
                    "status": "skipped",
                    "rows": count_track_rows(batch_dir / "tracks_2d.csv"),
                    "error": None,
                })
                continue

            print(
                f"[RUN][{side_name.upper()}] {batch_name} "
                f"({batch_index}/{len(batches)}) ids={batch_ids[0]}-{batch_ids[-1]} count={len(batch_ids)}"
            )
            status = "ok"
            error = None
            try:
                _, stopped = sam2run.run_single_video_tracking(
                    side_name=side_name.upper(),
                    video_path=side_data["video"],
                    out_dir=batch_dir,
                    points=batch_points,
                    crop=side_data["crop"],
                    scale=args.scale,
                    frame_extractor=args.frame_extractor,
                    predictor=predictor,
                    device=device,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                    async_loading_frames=args.async_loading_frames,
                    save_masks=args.save_masks,
                    save_tracks=True,
                    save_overlay=args.save_overlay,
                    preview=args.preview,
                    preview_max_width=args.preview_max_width,
                    corrections_payload=corrections,
                    corrections_path=corrections_path,
                    object_ids=batch_ids,
                    frames_dir_override=frames_dir,
                    shared_state=shared_state,
                    release_state=False,
                    correction_restart_mode=args.correction_restart_mode,
                    auto_pause_missing=args.auto_pause_missing,
                )
                if stopped:
                    status = "stopped"
            except Exception as exc:
                status = "failed"
                error = str(exc)
                print(f"[WARN][{side_name.upper()}] {batch_name} failed: {error}")

            final_fingerprint = batch_fingerprint(side_name, batch_ids, side_data, args, corrections)
            rows = count_track_rows(batch_dir / "tracks_2d.csv")
            (batch_dir / "batch_meta.json").write_text(json.dumps({
                "side": side_name,
                "batch": batch_name,
                "object_ids": [int(obj_id) for obj_id in batch_ids],
                "status": status,
                "error": error,
                "rows": int(rows),
                "expected_rows": int(frame_count) * int(len(batch_ids)),
                "frames": int(frame_count),
                "fingerprint": final_fingerprint,
            }, indent=2))
            results.append({
                "batch": batch_name,
                "object_ids": batch_ids,
                "status": status,
                "rows": rows,
                "error": error,
            })
    finally:
        sam2run._release_sam2_state(shared_state, device)

    merged_rows = []
    for batch_ids in batches:
        batch_name = f"batch_{batch_ids[0]:03d}_{batch_ids[-1]:03d}"
        merged_rows.extend(load_track_rows(objectwise_root / batch_name / "tracks_2d.csv"))
    write_track_rows(side_out / "tracks_2d.csv", merged_rows)

    summary = {
        "side": side_name,
        "object_ids": ids,
        "batch_size": int(args.batch_size),
        "batches": batches,
        "frames": int(frame_count),
        "rows": len(merged_rows),
        "ok_batches": [r["batch"] for r in results if r["status"] in ("ok", "skipped")],
        "stopped_batches": [r["batch"] for r in results if r["status"] == "stopped"],
        "failed_batches": [r["batch"] for r in results if r["status"] == "failed"],
        "results": results,
    }
    (side_out / "objectwise_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK][{side_name.upper()}] merged tracks: {side_out / 'tracks_2d.csv'} rows={len(merged_rows)}")
    return summary


def _write_batch_meta(batch_dir: Path,
                      side_name,
                      batch_name,
                      batch_ids,
                      status,
                      error,
                      rows,
                      frame_count,
                      fingerprint):
    (batch_dir / "batch_meta.json").write_text(json.dumps({
        "side": side_name,
        "batch": batch_name,
        "object_ids": [int(obj_id) for obj_id in batch_ids],
        "status": status,
        "error": error,
        "rows": int(rows),
        "expected_rows": int(frame_count) * int(len(batch_ids)),
        "frames": int(frame_count),
        "fingerprint": fingerprint,
    }, indent=2))


def _merge_side_batches(side_name, side_out: Path, batches, results, frame_count, args):
    objectwise_root = side_out / "objectwise"
    merged_rows = []
    for batch_ids in batches:
        batch_name = f"batch_{batch_ids[0]:03d}_{batch_ids[-1]:03d}"
        merged_rows.extend(load_track_rows(objectwise_root / batch_name / "tracks_2d.csv"))
    write_track_rows(side_out / "tracks_2d.csv", merged_rows)
    summary = {
        "side": side_name,
        "object_ids": [int(obj_id) for batch in batches for obj_id in batch],
        "batch_size": int(args.batch_size),
        "batches": batches,
        "frames": int(frame_count),
        "rows": len(merged_rows),
        "ok_batches": [r["batch"] for r in results if r["status"] in ("ok", "skipped")],
        "stopped_batches": [r["batch"] for r in results if r["status"] == "stopped"],
        "failed_batches": [r["batch"] for r in results if r["status"] == "failed"],
        "results": results,
    }
    (side_out / "objectwise_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK][{side_name.upper()}] merged tracks: {side_out / 'tracks_2d.csv'} rows={len(merged_rows)}")
    return summary


def run_dual_preview_sides(setup, ids, args, out_root: Path):
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 2:
        raise RuntimeError("--gpu-mode dual requires at least 2 CUDA GPUs")

    side_out = {
        "left": out_root / "left",
        "right": out_root / "right",
    }
    objectwise_root = {}
    for side_name in ("left", "right"):
        side_out[side_name].mkdir(parents=True, exist_ok=True)
        objectwise_root[side_name] = side_out[side_name] / "objectwise"
        objectwise_root[side_name].mkdir(parents=True, exist_ok=True)

    left_frames_dir, left_frame_count = prepare_frames(
        "left", setup["left"]["video"], setup["left"]["crop"], args, side_out["left"]
    )
    right_frames_dir, right_frame_count = prepare_frames(
        "right", setup["right"]["video"], setup["right"]["crop"], args, side_out["right"]
    )
    batches = chunk_ids(ids, args.batch_size)
    print(
        f"[INFO][DUAL] LEFT->cuda:0 RIGHT->cuda:1 objects={len(ids)} "
        f"batch_size={args.batch_size} batches={len(batches)} "
        f"frames_left={left_frame_count} frames_right={right_frame_count}"
    )
    print("[INFO][DUAL] Objectwise preview/corrections apply to the currently running batch only.")

    results = {"left": [], "right": []}
    corrections = setup["corrections"]
    corrections_path = setup["corrections_path"]

    for batch_index, batch_ids in enumerate(batches, start=1):
        batch_name = f"batch_{batch_ids[0]:03d}_{batch_ids[-1]:03d}"
        left_batch_dir = objectwise_root["left"] / batch_name
        right_batch_dir = objectwise_root["right"] / batch_name
        left_fingerprint = batch_fingerprint("left", batch_ids, setup["left"], args, corrections)
        right_fingerprint = batch_fingerprint("right", batch_ids, setup["right"], args, corrections)
        left_complete = (
            not args.overwrite
            and batch_is_complete(left_batch_dir, left_fingerprint, left_frame_count, len(batch_ids))
        )
        right_complete = (
            not args.overwrite
            and batch_is_complete(right_batch_dir, right_fingerprint, right_frame_count, len(batch_ids))
        )
        if left_complete and right_complete:
            print(
                f"[SKIP][DUAL] {batch_name} ({batch_index}/{len(batches)}) "
                f"ids={batch_ids[0]}-{batch_ids[-1]}"
            )
            results["left"].append({
                "batch": batch_name,
                "object_ids": batch_ids,
                "status": "skipped",
                "rows": count_track_rows(left_batch_dir / "tracks_2d.csv"),
                "error": None,
            })
            results["right"].append({
                "batch": batch_name,
                "object_ids": batch_ids,
                "status": "skipped",
                "rows": count_track_rows(right_batch_dir / "tracks_2d.csv"),
                "error": None,
            })
            continue

        if left_complete != right_complete:
            print(
                f"[INFO][DUAL] {batch_name} has only one completed side; rerunning both sides "
                "to keep dual preview paired."
            )

        print(
            f"[RUN][DUAL] {batch_name} ({batch_index}/{len(batches)}) "
            f"ids={batch_ids[0]}-{batch_ids[-1]} count={len(batch_ids)}"
        )
        left_result, right_result = sam2run.run_dual_gpu_with_parent_preview(
            left_video=setup["left"]["video"],
            right_video=setup["right"]["video"],
            left_out=left_batch_dir,
            right_out=right_batch_dir,
            left_points=[setup["left"]["points"][obj_id] for obj_id in batch_ids],
            right_points=[setup["right"]["points"][obj_id] for obj_id in batch_ids],
            crop_left=setup["left"]["crop"],
            crop_right=setup["right"]["crop"],
            scale=args.scale,
            frame_extractor=args.frame_extractor,
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=args.async_loading_frames,
            save_masks=args.save_masks,
            save_tracks=True,
            save_overlay=args.save_overlay,
            preview_max_width=args.preview_max_width,
            corrections_payload=corrections,
            corrections_path=corrections_path,
            correction_restart_mode=args.correction_restart_mode,
            auto_pause_missing=args.auto_pause_missing,
            left_object_ids=batch_ids,
            right_object_ids=batch_ids,
            left_frames_dir=left_frames_dir,
            right_frames_dir=right_frames_dir,
        )
        side_results = {
            "left": (left_result, left_batch_dir, left_frame_count),
            "right": (right_result, right_batch_dir, right_frame_count),
        }
        for side_name, (result, batch_dir, frame_count) in side_results.items():
            status = "ok"
            error = result.get("error")
            if not result.get("ok", False):
                status = "failed"
            elif result.get("stopped", False):
                status = "stopped"
            final_fingerprint = batch_fingerprint(side_name, batch_ids, setup[side_name], args, corrections)
            rows = count_track_rows(batch_dir / "tracks_2d.csv")
            _write_batch_meta(
                batch_dir,
                side_name,
                batch_name,
                batch_ids,
                status,
                error,
                rows,
                frame_count,
                final_fingerprint,
            )
            results[side_name].append({
                "batch": batch_name,
                "object_ids": batch_ids,
                "status": status,
                "rows": rows,
                "error": error,
            })

    return {
        "left": _merge_side_batches("left", side_out["left"], batches, results["left"], left_frame_count, args),
        "right": _merge_side_batches("right", side_out["right"], batches, results["right"], right_frame_count, args),
    }


def main():
    args = parse_args()
    if args.scale <= 0:
        raise RuntimeError("--scale must be > 0")

    setup = load_setup(args)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    max_count = min(len(setup["left"]["points"]), len(setup["right"]["points"]))
    ids = parse_ids(args.ids, max_count)
    sides = ["left", "right"] if args.side == "both" else [args.side]

    if args.gpu_mode == "dual" and args.preview and sides == ["left", "right"]:
        summaries = run_dual_preview_sides(setup, ids, args, out_root)
    else:
        summaries = {}
        for side_name in sides:
            summaries[side_name] = run_side(
                side_name,
                setup[side_name],
                ids,
                args,
                out_root,
                setup["corrections"],
                setup["corrections_path"],
            )

    (out_root / "objectwise_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"[OK] objectwise summary: {out_root / 'objectwise_summary.json'}")


if __name__ == "__main__":
    main()
