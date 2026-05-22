import os
import shutil
from pathlib import Path

import cv2

from video_splitter import (
    draw_hud,
    export_regions,
    format_time,
    overlaps,
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
        cv2.putText(
            display,
            "Linux keys: d/a frame  f/r ~1s  v/c ~10s  s start  e end  x undo  g goto  q export",
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

    print("\nExporting LEFT video...")
    export_regions(left_video, regions, left_dir, fps)

    print("\nExporting RIGHT video with the same frame ranges...")
    export_regions(right_video, regions, right_dir, fps)


def main():
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found in PATH.")
        print("Install FFmpeg and make sure 'ffmpeg' works in your terminal.")
        return

    left_video = _prompt_path("LEFT")
    right_video = _prompt_path("RIGHT")

    if not os.path.isfile(left_video):
        print(f"Error: LEFT file not found: {left_video}")
        return
    if not os.path.isfile(right_video):
        print(f"Error: RIGHT file not found: {right_video}")
        return

    reference_video = left_video
    print("\nPicking clip regions on LEFT video:")
    print(reference_video)

    regions, fps = pick_regions(reference_video)
    if not regions:
        return

    print(f"\n{len(regions)} clip(s) to export from BOTH videos:")
    for i, (s, e) in enumerate(regions):
        dur = (e - s) / fps
        print(f"  Clip {i + 1}: {format_time(s, fps)}  ->  {format_time(e, fps)}  ({dur:.1f}s)")

    confirm = input("\nProceed with export for both videos? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Cancelled.")
        return

    output_root = os.path.join(os.path.dirname(os.path.abspath(reference_video)), "dual_clips")
    os.makedirs(output_root, exist_ok=True)
    print(f"Output root: {output_root}")

    export_dual(left_video, right_video, regions, fps, output_root)


if __name__ == "__main__":
    main()
