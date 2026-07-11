import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from plot_gopro_imu import extract_imu, moving_average
from video_splitter import (
    draw_hud,
    export_regions,
    format_time,
    overlaps,
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _file_signature(path):
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    sig = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    try:
        cap = cv2.VideoCapture(str(resolved))
        sig.update({
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else -1,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else -1,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else -1.0,
        })
        cap.release()
    except Exception:
        sig.update({"frame_count": -1, "width": -1, "height": -1, "fps": -1.0})
    return sig


def _portable_signature(sig):
    return {
        key: value for key, value in sig.items()
        if key not in {"path", "mtime_ns"}
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _otsu_threshold(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return float(high)
    histogram, edges = np.histogram(finite, bins=256, range=(low, high))
    centers = (edges[:-1] + edges[1:]) * 0.5
    weights_left = np.cumsum(histogram)
    weights_right = finite.size - weights_left
    sums_left = np.cumsum(histogram * centers)
    total_sum = sums_left[-1]
    valid = (weights_left > 0) & (weights_right > 0)
    variance = np.zeros_like(centers)
    left_mean = np.divide(
        sums_left, weights_left, out=np.zeros_like(sums_left), where=weights_left > 0
    )
    right_mean = np.divide(
        total_sum - sums_left,
        weights_right,
        out=np.zeros_like(sums_left),
        where=weights_right > 0,
    )
    variance[valid] = (
        weights_left[valid]
        * weights_right[valid]
        * (left_mean[valid] - right_mean[valid]) ** 2
    )
    return float(centers[int(np.argmax(variance))])


def _fill_short_gaps(mask, max_gap_samples):
    result = mask.copy()
    false_indices = np.flatnonzero(~result)
    if false_indices.size == 0:
        return result
    runs = np.split(false_indices, np.where(np.diff(false_indices) != 1)[0] + 1)
    for run in runs:
        if (
            len(run) <= max_gap_samples
            and run[0] > 0
            and run[-1] < len(result) - 1
            and result[run[0] - 1]
            and result[run[-1] + 1]
        ):
            result[run] = True
    return result


def _event_intervals(times, active, min_duration_s=0.20):
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        return []
    runs = np.split(active_indices, np.where(np.diff(active_indices) != 1)[0] + 1)
    events = []
    for run in runs:
        start = float(times[run[0]])
        end_index = min(run[-1] + 1, len(times) - 1)
        end = float(times[end_index])
        if end - start >= min_duration_s:
            events.append((start, end))
    return events


class ImuActivity:
    def __init__(self, video_path, fps):
        accel_t, accel, gyro_t, gyro = extract_imu(Path(video_path))
        accel_rate = len(accel_t) / max(accel_t[-1] - accel_t[0], 1e-9)
        gyro_rate = len(gyro_t) / max(gyro_t[-1] - gyro_t[0], 1e-9)

        gravity = moving_average(accel, round(accel_rate))
        dynamic_accel = np.linalg.norm(accel - gravity, axis=1)
        gyro_magnitude = np.linalg.norm(gyro, axis=1)
        self.accel = moving_average(
            dynamic_accel[:, None], round(accel_rate * 0.10)
        )[:, 0]
        gyro_smooth = moving_average(
            gyro_magnitude[:, None], round(gyro_rate * 0.10)
        )[:, 0]
        self.times = accel_t
        self.gyro = np.interp(accel_t, gyro_t, gyro_smooth)
        self.accel_threshold = _otsu_threshold(self.accel)
        self.gyro_threshold = _otsu_threshold(self.gyro)

        active = (self.accel >= self.accel_threshold) | (
            self.gyro >= self.gyro_threshold
        )
        active = _fill_short_gaps(active, round(accel_rate * 0.75))
        self.events = _event_intervals(self.times, active)
        self.event_frames = [round(start * fps) for start, _ in self.events]
        self.pre_roll_frames = max(1, round(0.5 * fps))
        self._graph_cache = {}

    def values_at(self, time_s):
        index = int(np.searchsorted(self.times, time_s, side="left"))
        index = max(0, min(index, len(self.times) - 1))
        return float(self.accel[index]), float(self.gyro[index])

    def next_event_frame(self, current_frame):
        minimum_start = current_frame + self.pre_roll_frames + 1
        for event_frame in self.event_frames:
            if event_frame >= minimum_start:
                return max(0, event_frame - self.pre_roll_frames), event_frame
        return None

    def graph(self, width):
        cached = self._graph_cache.get(width)
        if cached is not None:
            return cached
        plot_times = np.linspace(self.times[0], self.times[-1], width)
        accel = np.interp(plot_times, self.times, self.accel) / max(
            self.accel_threshold, 1e-9
        )
        gyro = np.interp(plot_times, self.times, self.gyro) / max(
            self.gyro_threshold, 1e-9
        )
        score = np.maximum(accel, gyro)
        self._graph_cache[width] = score
        return score


def draw_imu_overlay(display, imu, frame_idx, total_frames, fps, fsize):
    height, width = display.shape[:2]
    time_s = frame_idx / fps
    accel, gyro = imu.values_at(time_s)
    next_event = imu.next_event_frame(frame_idx)
    next_text = (
        f"next={format_time(next_event[1], fps)}" if next_event else "next=end"
    )
    cv2.putText(
        display,
        f"IMU dynamic={accel:.2f} m/s^2  gyro={gyro:.3f} rad/s  "
        f"events={len(imu.events)}  {next_text}",
        (20, max(35, int(350 * fsize))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, fsize * 0.8),
        (0, 220, 255),
        max(1, int(fsize * 2)),
    )

    graph_bottom = height - max(28, int(8 * fsize))
    graph_height = max(70, int(38 * fsize))
    graph_top = max(0, graph_bottom - graph_height)
    overlay = display.copy()
    cv2.rectangle(overlay, (0, graph_top), (width - 1, graph_bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, display, 0.40, 0, display)

    score = np.clip(imu.graph(width), 0.0, 3.0)
    threshold_y = graph_bottom - int(graph_height / 3.0)
    cv2.line(display, (0, threshold_y), (width - 1, threshold_y), (80, 120, 120), 1)
    points = np.column_stack(
        (
            np.arange(width, dtype=np.int32),
            graph_bottom - (score / 3.0 * graph_height).astype(np.int32),
        )
    ).reshape(-1, 1, 2)
    cv2.polylines(display, [points], False, (0, 200, 255), max(1, int(fsize)))
    for start, _ in imu.events:
        x = int(start / max(imu.times[-1], 1e-9) * (width - 1))
        cv2.line(display, (x, graph_top), (x, graph_bottom), (255, 0, 255), 1)
    playhead_x = int(frame_idx / max(total_frames - 1, 1) * (width - 1))
    cv2.line(
        display,
        (playhead_x, graph_top),
        (playhead_x, graph_bottom),
        (255, 255, 255),
        max(1, int(fsize)),
    )


def _on_trackbar(val):
    pass


def _video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return cap, fps, total_frames


def _prompt_path(label):
    return input(f"Enter path to {label} video: ").strip().strip('"')


def _read_key(delay_ms=30):
    key = cv2.waitKey(delay_ms)
    if key == -1:
        return -1
    return key & 0xFF


def pick_regions(reference_video):
    cap, fps, total_frames = _video_metadata(reference_video)
    imu = None
    try:
        print("[INFO] Loading GoPro accelerometer and gyroscope data...")
        imu = ImuActivity(reference_video, fps)
        print(
            f"[OK] Detected {len(imu.events)} IMU event(s); "
            f"thresholds: acceleration={imu.accel_threshold:.3f} m/s^2, "
            f"gyro={imu.gyro_threshold:.4f} rad/s"
        )
    except Exception as exc:
        print(f"[WARN] IMU event navigation unavailable: {exc}")

    jump_small = max(1, int(fps))
    jump_large = max(10, int(fps * 10))

    current_frame = 0
    regions = []
    pending_start = None
    digit_buf = ""
    g_mode = False

    window = "Dual Video Clip Picker Linux"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Frame", window, 0, max(1, total_frames - 1), _on_trackbar)
    last_trackbar = 0

    print("\nLinux controls:")
    print("  s         mark start of a clip")
    print("  e         mark end of clip (after marking start)")
    print("  x         remove last clip (or cancel open start)")
    print("  d / a     next / previous frame")
    print("  l / j     next / previous frame, alternate keys")
    print(f"  f / r     forward / reverse ~1s ({jump_small} frames)")
    print(f"  v / c     forward / reverse ~10s ({jump_large} frames)")
    print("  n         jump to next acceleration event")
    print("  g         go to frame number")
    print("  q         finish and export both videos")
    print("  Esc       quit without exporting\n")

    while True:
        current_frame = max(0, min(current_frame, total_frames - 1))

        tb_val = cv2.getTrackbarPos("Frame", window)
        if tb_val != last_trackbar:
            current_frame = tb_val
            last_trackbar = tb_val
        else:
            cv2.setTrackbarPos("Frame", window, current_frame)
            last_trackbar = current_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            current_frame += 1
            continue

        display = frame.copy()
        fsize = max(0.5, display.shape[1] / 1280)
        draw_hud(display, current_frame, total_frames, fps, regions, pending_start, g_mode, digit_buf, fsize)
        if imu is not None:
            draw_imu_overlay(display, imu, current_frame, total_frames, fps, fsize)
        cv2.putText(
            display,
            "Linux keys: d/a frame  f/r ~1s  v/c ~10s  n next IMU  s start  e end  x undo  g goto  q export",
            (20, display.shape[0] - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, fsize * 0.75),
            (255, 255, 255),
            max(1, int(fsize * 2)),
        )
        cv2.imshow(window, display)
        key = _read_key(30)

        if key == -1:
            continue

        if key == 27:
            print("Exited without exporting.")
            cap.release()
            cv2.destroyAllWindows()
            return None, fps

        if key == ord("q"):
            if not regions:
                print("No clips selected. Mark at least one clip before finishing.")
                continue
            if pending_start is not None:
                print("You have an open start mark -- press e to close it first.")
                continue
            break

        if key == ord("g"):
            g_mode = True
            digit_buf = ""
            continue

        if g_mode:
            if key in (10, 13):
                try:
                    current_frame = max(0, min(int(digit_buf), total_frames - 1))
                    print(f"Jumped to frame {current_frame}")
                except ValueError:
                    print("Invalid frame number.")
                g_mode = False
                digit_buf = ""
            elif key in (8, 127):
                digit_buf = digit_buf[:-1]
            elif ord("0") <= key <= ord("9"):
                digit_buf += chr(key)
            continue

        if key in (ord("d"), ord("l")):
            current_frame += 1
        elif key in (ord("a"), ord("j")):
            current_frame -= 1
        elif key == ord("f"):
            current_frame += jump_small
        elif key == ord("r"):
            current_frame -= jump_small
        elif key == ord("v"):
            current_frame += jump_large
        elif key == ord("c"):
            current_frame -= jump_large
        elif key == ord("n"):
            if imu is None:
                print("IMU event navigation is unavailable for this video.")
            else:
                next_event = imu.next_event_frame(current_frame)
                if next_event is None:
                    print("No later acceleration event.")
                else:
                    current_frame, event_frame = next_event
                    print(
                        f"Jumped to next acceleration event @ frame {event_frame} "
                        f"({format_time(event_frame, fps)}), with 0.5s pre-roll"
                    )
        elif key == ord("s"):
            if pending_start is not None:
                print("Already have an open start -- press e to set the end, or x to cancel.")
            else:
                pending_start = current_frame
                print(f"Start marked @ frame {current_frame}  ({format_time(current_frame, fps)})")
        elif key == ord("e"):
            if pending_start is None:
                print("No start marked yet -- press s first.")
            elif current_frame <= pending_start:
                print("End must be after start.")
            elif overlaps(regions, pending_start, current_frame):
                print("Region overlaps an existing clip -- adjust start or end.")
            else:
                regions.append((pending_start, current_frame))
                regions.sort()
                print(
                    f"Clip {len(regions)} added: "
                    f"frame {pending_start} -> {current_frame}  "
                    f"({(current_frame - pending_start) / fps:.1f}s)"
                )
                pending_start = None
        elif key == ord("x"):
            if pending_start is not None:
                print(f"Cancelled open start @ frame {pending_start}")
                pending_start = None
            elif regions:
                removed = regions.pop()
                print(f"Removed clip: frame {removed[0]} -> {removed[1]}")
            else:
                print("Nothing to remove.")

    cap.release()
    cv2.destroyAllWindows()
    return regions, fps


def export_dual(left_video, right_video, regions, fps, output_root):
    left_dir = os.path.join(output_root, f"{Path(left_video).stem}_clips")
    right_dir = os.path.join(output_root, f"{Path(right_video).stem}_clips")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    print(f"\nLEFT output folder:  {left_dir}")
    print(f"RIGHT output folder: {right_dir}")

    print("\nExporting LEFT and RIGHT videos concurrently...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(
            export_regions, left_video, regions, left_dir, fps
        )
        right_future = executor.submit(
            export_regions, right_video, regions, right_dir, fps
        )
        left_future.result()
        right_future.result()

    clips = []
    for index, (start_frame, end_frame) in enumerate(regions, start=1):
        clips.append(
            {
                "index": index,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "left_video": str(
                    Path(left_dir) / f"{Path(left_video).stem}_clip_{index}.mp4"
                ),
                "right_video": str(
                    Path(right_dir) / f"{Path(right_video).stem}_clip_{index}.mp4"
                ),
            }
        )
    return clips


def _split_manifest_payload(
    left_video,
    right_video,
    output_root,
    regions,
    fps,
    select_only=False,
):
    left_dir = Path(output_root) / f"{Path(left_video).stem}_clips"
    right_dir = Path(output_root) / f"{Path(right_video).stem}_clips"
    clips = []
    for index, (start_frame, end_frame) in enumerate(regions, start=1):
        clips.append(
            {
                "index": index,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "left_video": (
                    str(Path(left_video).resolve())
                    if select_only
                    else str(
                        (left_dir / f"{Path(left_video).stem}_clip_{index}.mp4").resolve()
                    )
                ),
                "right_video": (
                    str(Path(right_video).resolve())
                    if select_only
                    else str(
                        (right_dir / f"{Path(right_video).stem}_clip_{index}.mp4").resolve()
                    )
                ),
            }
        )
    return {
        "version": 1,
        "mode": "virtual" if select_only else "physical",
        "status": "complete" if select_only else "selected",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "left_source": _file_signature(left_video),
        "right_source": _file_signature(right_video),
        "reference_side": "right",
        "output_root": str(Path(output_root).resolve()),
        "fps": float(fps),
        "regions": [
            {"start_frame": int(start), "end_frame": int(end)}
            for start, end in regions
        ],
        "clips": clips,
    }


def _manifest_matches_sources(payload, left_video, right_video, output_root):
    try:
        return (
            payload.get("version") == 1
            and _portable_signature(payload.get("left_source", {}))
            == _portable_signature(_file_signature(left_video))
            and _portable_signature(payload.get("right_source", {}))
            == _portable_signature(_file_signature(right_video))
            and Path(payload.get("output_root", "")).resolve()
            == Path(output_root).resolve()
        )
    except (OSError, TypeError, ValueError):
        return False


def _manifest_clips_complete(payload):
    clips = payload.get("clips", [])
    if payload.get("mode") == "virtual":
        return bool(clips) and all(
            int(clip.get("end_frame", 0)) > int(clip.get("start_frame", -1))
            and Path(clip[side]).is_file()
            for clip in clips
            for side in ("left_video", "right_video")
        )
    return bool(clips) and all(
        Path(clip[side]).is_file() and Path(clip[side]).stat().st_size > 0
        for clip in clips
        for side in ("left_video", "right_video")
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively select matching regions from one raw stereo pair."
    )
    parser.add_argument("--left", help="Raw LEFT video path")
    parser.add_argument("--right", help="Raw RIGHT video path")
    parser.add_argument(
        "--output-root",
        help="Output root. Default: dual_clips beside the RIGHT video.",
    )
    parser.add_argument(
        "--manifest",
        help="Write/resume a machine-readable split manifest at this path.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Continue immediately after region selection without confirmation.",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Save source paths and frame ranges without exporting video files.",
    )
    args = parser.parse_args()
    if bool(args.left) != bool(args.right):
        parser.error("--left and --right must be supplied together")
    return args


def main():
    args = parse_args()
    if not args.select_only and shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found in PATH.")
        print("Install FFmpeg and make sure 'ffmpeg' works in your terminal.")
        return 2

    left_video = args.left or _prompt_path("LEFT")
    right_video = args.right or _prompt_path("RIGHT")
    left_video = str(Path(left_video).expanduser().resolve())
    right_video = str(Path(right_video).expanduser().resolve())

    if not os.path.isfile(left_video):
        print(f"Error: LEFT file not found: {left_video}")
        return 2
    if not os.path.isfile(right_video):
        print(f"Error: RIGHT file not found: {right_video}")
        return 2

    reference_video = right_video
    output_root = args.output_root or os.path.join(
        os.path.dirname(os.path.abspath(reference_video)), "dual_clips"
    )
    output_root = str(Path(output_root).expanduser().resolve())
    os.makedirs(output_root, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    payload = None

    if manifest_path is not None and manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: could not read split manifest {manifest_path}: {exc}")
            return 2
        if not _manifest_matches_sources(payload, left_video, right_video, output_root):
            print("Error: existing split manifest does not match the requested source videos.")
            return 2
        requested_mode = "virtual" if args.select_only else "physical"
        if (
            payload.get("status") == "complete"
            and payload.get("mode") == requested_mode
            and _manifest_clips_complete(payload)
        ):
            print(f"[SKIP] Split clips already complete: {manifest_path}")
            return 0
        regions = [
            (int(region["start_frame"]), int(region["end_frame"]))
            for region in payload.get("regions", [])
        ]
        fps = float(payload.get("fps", 0.0))
        if not regions or fps <= 0:
            print(f"Error: split manifest has no reusable regions: {manifest_path}")
            return 2
        if args.select_only:
            payload = _split_manifest_payload(
                left_video,
                right_video,
                output_root,
                regions,
                fps,
                select_only=True,
            )
            payload["completed_at"] = _utc_now()
            _atomic_write_json(manifest_path, payload)
            print(
                f"[RESUME] Converted {len(regions)} selected region(s) to virtual clips: "
                f"{manifest_path}"
            )
        else:
            print(f"[RESUME] Exporting {len(regions)} selected clip pair(s) from {manifest_path}")
    else:
        print("\nPicking clip regions on RIGHT video:")
        print(reference_video)
        regions, fps = pick_regions(reference_video)
        if not regions:
            return 1

        action = "selected as virtual ranges" if args.select_only else "to export from BOTH videos"
        print(f"\n{len(regions)} clip(s) {action}:")
        for i, (s, e) in enumerate(regions):
            dur = (e - s) / fps
            print(
                f"  Clip {i + 1}: {format_time(s, fps)}  ->  "
                f"{format_time(e, fps)}  ({dur:.1f}s)"
            )

        if not args.yes:
            prompt = (
                "\nSave these virtual clip ranges? [Y/n]: "
                if args.select_only
                else "\nProceed with export for both videos? [Y/n]: "
            )
            confirm = input(prompt).strip().lower()
            if confirm == "n":
                print("Cancelled.")
                return 1
        payload = _split_manifest_payload(
            left_video,
            right_video,
            output_root,
            regions,
            fps,
            select_only=args.select_only,
        )
        if manifest_path is not None:
            if args.select_only:
                payload["completed_at"] = _utc_now()
            _atomic_write_json(manifest_path, payload)

    if args.select_only:
        if manifest_path is None:
            print("Error: --select-only requires --manifest.")
            return 2
        print(f"[OK] Saved {len(regions)} virtual clip range(s): {manifest_path}")
        return 0

    print(f"Output root: {output_root}")
    clips = export_dual(left_video, right_video, regions, fps, output_root)
    if not all(
        Path(clip[side]).is_file() and Path(clip[side]).stat().st_size > 0
        for clip in clips
        for side in ("left_video", "right_video")
    ):
        print("Error: one or more expected split clips were not created.")
        return 1
    if payload is not None and manifest_path is not None:
        payload["status"] = "complete"
        payload["updated_at"] = _utc_now()
        payload["completed_at"] = _utc_now()
        payload["clips"] = clips
        _atomic_write_json(manifest_path, payload)
        print(f"[OK] Split manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
