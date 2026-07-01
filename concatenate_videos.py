#!/usr/bin/env python3
"""Concatenate two compatible videos without re-encoding or dropping frames."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


STREAM_FIELDS = (
    "codec_type",
    "codec_name",
    "codec_tag_string",
    "profile",
    "level",
    "width",
    "height",
    "pix_fmt",
    "field_order",
    "r_frame_rate",
    "avg_frame_rate",
    "time_base",
    "sample_fmt",
    "sample_rate",
    "channels",
    "channel_layout",
)


def _prompt_path(label: str) -> Path:
    while True:
        try:
            raw = input(f"{label}: ").strip()
        except EOFError as exc:
            raise SystemExit("Input canceled.") from exc
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'\"', "'"}:
            raw = raw[1:-1]
        if raw:
            return Path(raw).expanduser()
        print("Enter a path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate two videos end-to-end using FFmpeg stream copy. "
            "The inputs must have matching stream formats."
        )
    )
    parser.add_argument("first", type=Path, nargs="?", help="Video that appears first")
    parser.add_argument("second", type=Path, nargs="?", help="Video that appears second")
    parser.add_argument("output", type=Path, nargs="?", help="Combined output video")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    args = parser.parse_args()
    supplied = (args.first, args.second, args.output)
    if all(value is None for value in supplied) and len(sys.argv) == 1:
        print("Enter the two videos in playback order.")
        args.first = _prompt_path("First video path")
        args.second = _prompt_path("Second video path")
        args.output = _prompt_path("Output video name or path")
        if not args.output.suffix:
            args.output = args.output.with_suffix(".mp4")
    elif any(value is None for value in supplied):
        parser.error("first, second, and output must be provided together")
    return args


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
    ]
    command.append(str(path))

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"Could not inspect {path}: {detail}")
    return json.loads(result.stdout)


def stream_signature(stream: dict[str, Any]) -> dict[str, Any]:
    return {field: stream.get(field) for field in STREAM_FIELDS}


def validate_compatibility(
    first_info: dict[str, Any], second_info: dict[str, Any]
) -> None:
    supported_types = {"video", "audio"}
    first_streams = [
        stream
        for stream in first_info.get("streams", [])
        if stream.get("codec_type") in supported_types
    ]
    second_streams = [
        stream
        for stream in second_info.get("streams", [])
        if stream.get("codec_type") in supported_types
    ]

    if not any(stream.get("codec_type") == "video" for stream in first_streams):
        raise RuntimeError("The first input has no video stream.")
    if not any(stream.get("codec_type") == "video" for stream in second_streams):
        raise RuntimeError("The second input has no video stream.")
    if len(first_streams) != len(second_streams):
        raise RuntimeError(
            "The inputs have different stream counts "
            f"({len(first_streams)} versus {len(second_streams)})."
        )

    differences: list[str] = []
    for index, (first_stream, second_stream) in enumerate(
        zip(first_streams, second_streams)
    ):
        first_sig = stream_signature(first_stream)
        second_sig = stream_signature(second_stream)
        for field in STREAM_FIELDS:
            if first_sig[field] != second_sig[field]:
                differences.append(
                    f"stream {index} {field}: "
                    f"{first_sig[field]!r} != {second_sig[field]!r}"
                )

    if differences:
        rendered = "\n  ".join(differences)
        raise RuntimeError(
            "The videos cannot be concatenated accurately with stream copy. "
            "Their stream formats differ:\n  " + rendered
        )


def concat_path_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def video_stream(info: dict[str, Any], path: Path) -> dict[str, Any]:
    video_streams = [
        stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise RuntimeError(
            f"Expected exactly one video stream in {path}, found {len(video_streams)}."
        )
    return video_streams[0]


def container_frame_count(info: dict[str, Any], path: Path) -> int | None:
    value = video_stream(info, path).get("nb_frames")
    if value in (None, "N/A"):
        return None
    return int(value)


def main() -> int:
    args = parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            print(f"Error: {executable} was not found in PATH.", file=sys.stderr)
            return 2

    first = args.first.expanduser().resolve()
    second = args.second.expanduser().resolve()
    output = args.output.expanduser().resolve()

    for path in (first, second):
        if not path.is_file():
            print(f"Error: input video does not exist: {path}", file=sys.stderr)
            return 2
    if first == second:
        print("Error: the two input paths refer to the same file.", file=sys.stderr)
        return 2
    if output in (first, second):
        print("Error: output must not overwrite either input video.", file=sys.stderr)
        return 2
    if output.exists() and not args.overwrite:
        print(
            f"Error: output already exists: {output}\nUse --overwrite to replace it.",
            file=sys.stderr,
        )
        return 2

    try:
        print("[INFO] Inspecting input streams...")
        first_info = probe(first)
        second_info = probe(second)
        validate_compatibility(first_info, second_info)
        first_frames = container_frame_count(first_info, first)
        second_frames = container_frame_count(second_info, second)
        expected_frames = (
            first_frames + second_frames
            if first_frames is not None and second_frames is not None
            else None
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="video_concat_") as temp_dir:
            concat_list = Path(temp_dir) / "inputs.txt"
            concat_list.write_text(
                concat_path_line(first) + concat_path_line(second), encoding="utf-8"
            )
            command = [
                "ffmpeg",
                "-hide_banner",
                "-y" if args.overwrite else "-n",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ]
            print(f"[INFO] Concatenating {first.name} + {second.name}...")
            result = subprocess.run(command)
            if result.returncode != 0:
                output.unlink(missing_ok=True)
                raise RuntimeError(f"FFmpeg exited with status {result.returncode}.")

        output_info = probe(output)
        actual_frames = container_frame_count(output_info, output)
        if expected_frames is not None and actual_frames is not None:
            if actual_frames != expected_frames:
                output.unlink(missing_ok=True)
                raise RuntimeError(
                    "Frame verification failed: expected "
                    f"{expected_frames}, found {actual_frames}. The output was removed."
                )
            verification = (
                f"[OK] Verified {actual_frames} frames "
                "(exact sum reported by both MP4 containers)."
            )
        else:
            verification = (
                "[WARN] This container does not report frame counts; "
                "the streams were copied without re-encoding."
            )

        print(f"[OK] Wrote {output}\n{verification}")
        return 0
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
