"""Video input/output helpers and ffmpeg/NVENC capability probing."""

import shutil
import subprocess
from pathlib import Path

import cv2

VIZ_PLAYBACK_SLOWDOWN = 4.0
_NVENC_AVAILABLE: bool | None = None


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _visualization_fps(source_fps: float):
    fps = float(source_fps) if float(source_fps) > 0 else 30.0
    return max(fps / VIZ_PLAYBACK_SLOWDOWN, 1.0)


def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _has_nvenc():
    """Return True only if ffmpeg can actually initialize h264_nvenc."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    if not _has_ffmpeg():
        _NVENC_AVAILABLE = False
        return _NVENC_AVAILABLE

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=size=16x16:rate=1",
        "-frames:v", "1",
        "-c:v", "h264_nvenc",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[VIS] NVENC probe failed ({exc}); falling back to mp4v.")
        _NVENC_AVAILABLE = False
        return _NVENC_AVAILABLE

    _NVENC_AVAILABLE = result.returncode == 0
    if not _NVENC_AVAILABLE:
        err = " ".join((result.stderr or "").strip().split())
        if err:
            print(f"[VIS] NVENC unavailable; falling back to mp4v. ffmpeg said: {err}")
        else:
            print("[VIS] NVENC unavailable; falling back to mp4v.")
    return _NVENC_AVAILABLE


def _open_writer(path, fps, w, h, encoder):
    """Returns either a cv2.VideoWriter or a dict wrapping an ffmpeg pipe."""
    path = str(path)
    if encoder == "nvenc" and _has_nvenc():
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
            "-i", "-",
            "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
            path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return {"proc": proc, "kind": "ffmpeg"}
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"[VIS] Could not open video writer: {path}")
    return {"writer": vw, "kind": "cv2"}


def _writer_write(w, img):
    if w["kind"] == "ffmpeg":
        if w["proc"].poll() is not None:
            raise RuntimeError("[VIS] ffmpeg encoder exited before all frames were written.")
        w["proc"].stdin.write(img.tobytes())
    else:
        w["writer"].write(img)


def _writer_close(w):
    if w["kind"] == "ffmpeg":
        w["proc"].stdin.close()
        ret = w["proc"].wait()
        if ret != 0:
            raise RuntimeError(f"[VIS] ffmpeg encoder exited with status {ret}.")
    else:
        w["writer"].release()


def _concat_segments(segments, out_path):
    """Lossless concat of mp4 segments via ffmpeg."""
    if len(segments) == 1:
        shutil.move(segments[0], out_path)
        return
    list_file = Path(out_path).with_suffix(".list.txt")
    list_file.write_text("\n".join(f"file '{Path(s).resolve()}'" for s in segments))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)
    for s in segments:
        Path(s).unlink(missing_ok=True)
