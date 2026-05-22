import cv2
import os
import subprocess
import shutil
from pathlib import Path


def format_time(frame_idx, fps):
    seconds = frame_idx / fps
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def frame_to_seconds(frame_idx, fps):
    return frame_idx / fps


def run_ffmpeg_command(cmd, label=""):
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"\nFFmpeg failed on {label}:")
        print(result.stderr[-2000:])
        raise subprocess.CalledProcessError(result.returncode, cmd)


def export_regions(video_path, regions, output_dir, fps):
    """
    Export a list of (start_frame, end_frame) regions.
    Uses dual -ss for accurate + fast seeking.
    """
    video_stem = Path(video_path).stem
    print("\nExporting clips (accurate + fast)...")

    for i, (start_f, end_f) in enumerate(regions):
        start_sec   = frame_to_seconds(start_f, fps)
        end_sec     = frame_to_seconds(end_f,   fps)
        duration    = end_sec - start_sec
        out_path    = os.path.join(output_dir, f"{video_stem}_clip_{i + 1}.mp4")

        pre_seek    = min(5.0, start_sec)
        fast_seek   = start_sec - pre_seek
        fine_offset = pre_seek

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{fast_seek:.6f}",
            "-i", video_path,
            "-ss", f"{fine_offset:.6f}",
            "-t",  f"{duration:.6f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-stats",
            out_path,
        ]

        print(
            f"  Clip {i + 1}: "
            f"{format_time(start_f, fps)} -> {format_time(end_f, fps)} "
            f"({duration:.1f}s)"
        )
        run_ffmpeg_command(cmd, label=f"clip {i + 1}")

    print("\nDone.")


# ── trackbar callback (no-op; position is read in the loop) ──────────────────
def _on_trackbar(val):
    pass


def overlaps(regions, start, end):
    """Return True if [start, end) overlaps any existing region."""
    for s, e in regions:
        if start < e and end > s:
            return True
    return False


def draw_hud(display, frame_idx, total_frames, fps, regions, pending_start,
             g_mode, digit_buf, fsize):
    h, w = display.shape[:2]
    row  = int(35 * fsize * 2)

    def put(text, y, color=(0, 255, 0)):
        cv2.putText(display, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fsize, color, max(1, int(fsize * 2)))

    put(f"Frame: {frame_idx} / {total_frames - 1}", row)
    put(f"Time:  {format_time(frame_idx, fps)}",     row * 2)

    if pending_start is not None:
        put(f"Start marked @ {pending_start}  — navigate to end, press E",
            row * 3, (0, 200, 255))
    else:
        put(f"Clips: {len(regions)}  — press S to mark start",
            row * 3, (0, 255, 255))

    if g_mode:
        put(f"Go to frame: {digit_buf}_", row * 4, (255, 200, 0))

    # Draw confirmed regions as green bar along the bottom
    for s, e in regions:
        x1 = int(s / total_frames * w)
        x2 = int(e / total_frames * w)
        cv2.rectangle(display, (x1, h - 18), (x2, h - 4), (0, 200, 80), -1)

    # Draw pending region in yellow
    if pending_start is not None:
        x1 = int(pending_start / total_frames * w)
        x2 = int(frame_idx    / total_frames * w)
        if x2 > x1:
            cv2.rectangle(display, (x1, h - 18), (x2, h - 4), (0, 200, 255), -1)

    # Playhead
    px = int(frame_idx / total_frames * w)
    cv2.line(display, (px, h - 22), (px, h), (255, 255, 255), 2)


def main():
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found in PATH.")
        print("Install FFmpeg and make sure 'ffmpeg' works in your terminal.")
        return

    video_path = input("Enter path to video: ").strip().strip('"')
    if not os.path.isfile(video_path):
        print("Error: file not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or fps <= 0:
        print("Error: invalid video metadata.")
        cap.release()
        return

    jump_small = max(1,  int(fps))       # ~1 second
    jump_large = max(10, int(fps * 10))  # ~10 seconds

    current_frame = 0
    regions       = []   # list of confirmed (start, end) tuples
    pending_start = None # start frame waiting for an end mark
    digit_buf     = ""
    g_mode        = False

    window = "Video Clip Picker"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Frame", window, 0, max(1, total_frames - 1), _on_trackbar)
    last_trackbar = 0

    print("\nControls:")
    print(f"  S         mark start of a clip")
    print(f"  E         mark end of clip (after marking start)")
    print(f"  x         remove last clip (or cancel open start)")
    print(f"  d / a     next / previous frame")
    print(f"  D / A     jump ~1s ({jump_small} frames)")
    print(f"  Ctrl+D/A  jump ~10s ({jump_large} frames)")
    print(f"  g         go to frame number")
    print(f"  q         finish and export")
    print(f"  Esc       quit without exporting\n")

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
        fsize   = max(0.5, display.shape[1] / 1280)
        draw_hud(display, current_frame, total_frames, fps,
                 regions, pending_start, g_mode, digit_buf, fsize)

        cv2.imshow(window, display)
        key = cv2.waitKeyEx(30)

        if key == -1:
            continue

        # ── quit / finish ──────────────────────────────────────────────────
        if key == 27:  # Esc
            print("Exited without exporting.")
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord('q'):
            if not regions:
                print("No clips selected. Mark at least one clip before finishing.")
                continue
            if pending_start is not None:
                print("You have an open start mark — press E to close it first.")
                continue
            break

        # ── go-to mode ─────────────────────────────────────────────────────
        if key == ord('g'):
            g_mode    = True
            digit_buf = ""
            continue

        if g_mode:
            if key == 13:  # Enter
                try:
                    current_frame = max(0, min(int(digit_buf), total_frames - 1))
                    print(f"Jumped to frame {current_frame}")
                except ValueError:
                    print("Invalid frame number.")
                g_mode    = False
                digit_buf = ""
            elif key == 8:  # Backspace
                digit_buf = digit_buf[:-1]
            elif ord('0') <= key <= ord('9'):
                digit_buf += chr(key)
            continue

        # ── navigation ─────────────────────────────────────────────────────
        if key == ord('d'):
            current_frame += 1
        elif key == ord('a'):
            current_frame -= 1
        elif key == ord('D'):
            current_frame += jump_small
        elif key == ord('A'):
            current_frame -= jump_small
        elif key == 4:   # Ctrl+D
            current_frame += jump_large
        elif key == 1:   # Ctrl+A
            current_frame -= jump_large

        # ── region marking ─────────────────────────────────────────────────
        elif key == ord('s') or key == ord('S'):
            if pending_start is not None:
                print("Already have an open start — press E to set the end, or X to cancel.")
            else:
                pending_start = current_frame
                print(f"Start marked @ frame {current_frame}  ({format_time(current_frame, fps)})")

        elif key == ord('e') or key == ord('E'):
            if pending_start is None:
                print("No start marked yet — press S first.")
            elif current_frame <= pending_start:
                print("End must be after start.")
            elif overlaps(regions, pending_start, current_frame):
                print("Region overlaps an existing clip — adjust start or end.")
            else:
                regions.append((pending_start, current_frame))
                regions.sort()
                print(
                    f"Clip {len(regions)} added: "
                    f"frame {pending_start} -> {current_frame}  "
                    f"({(current_frame - pending_start) / fps:.1f}s)"
                )
                pending_start = None

        elif key == ord('x'):
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

    # ── confirmation summary ───────────────────────────────────────────────
    print(f"\n{len(regions)} clip(s) to export:")
    for i, (s, e) in enumerate(regions):
        dur = (e - s) / fps
        print(f"  Clip {i + 1}: {format_time(s, fps)}  ->  {format_time(e, fps)}  ({dur:.1f}s)")

    confirm = input("\nProceed with export? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Cancelled.")
        return

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(video_path)),
        f"{Path(video_path).stem}_clips",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    export_regions(
        video_path = video_path,
        regions    = regions,
        output_dir = output_dir,
        fps        = fps,
    )


if __name__ == "__main__":
    main()