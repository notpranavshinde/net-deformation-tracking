#!/usr/bin/env python3
"""Extract and plot GoPro accelerometer/gyroscope data from GPMF metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Klv:
    key: str
    type_code: int
    structure_size: int
    repeat: int
    payload: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accelerometer and gyroscope data stored in a GoPro MP4."
    )
    parser.add_argument("video", type=Path, help="GoPro MP4 containing a gpmd stream")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG (default: imu_analysis/<video-stem>_imu.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional output CSV containing the extracted samples",
    )
    return parser.parse_args()


def iter_klv(data: bytes) -> Iterator[Klv]:
    offset = 0
    while offset + 8 <= len(data):
        key_bytes = data[offset : offset + 4]
        if key_bytes == b"\0\0\0\0":
            break
        type_code = data[offset + 4]
        structure_size = data[offset + 5]
        repeat = struct.unpack_from(">H", data, offset + 6)[0]
        data_size = structure_size * repeat
        payload_start = offset + 8
        payload_end = payload_start + data_size
        if payload_end > len(data):
            break
        try:
            key = key_bytes.decode("ascii")
        except UnicodeDecodeError:
            break
        yield Klv(
            key=key,
            type_code=type_code,
            structure_size=structure_size,
            repeat=repeat,
            payload=data[payload_start:payload_end],
        )
        offset = payload_start + ((data_size + 3) & ~3)


def stream_payloads(data: bytes) -> Iterator[bytes]:
    for item in iter_klv(data):
        if item.key == "STRM":
            yield item.payload
        elif item.type_code == 0:
            yield from stream_payloads(item.payload)


def decode_text(item: Klv) -> str:
    return item.payload.rstrip(b"\0").decode("ascii", errors="replace")


def decode_scale(item: Klv) -> list[float]:
    type_char = chr(item.type_code)
    formats = {"s": ">h", "S": ">H", "l": ">i", "L": ">I", "f": ">f"}
    value_format = formats.get(type_char)
    if value_format is None:
        return []
    value_size = struct.calcsize(value_format)
    return [
        float(struct.unpack_from(value_format, item.payload, offset)[0])
        for offset in range(0, len(item.payload), value_size)
    ]


def decode_xyz(item: Klv, scales: list[float], orientation: str):
    import numpy as np

    type_char = chr(item.type_code)
    formats = {"s": ">i2", "S": ">u2", "l": ">i4", "L": ">u4", "f": ">f4"}
    dtype = formats.get(type_char)
    if dtype is None or item.structure_size <= 0:
        raise RuntimeError(f"Unsupported {item.key} GPMF type {type_char!r}.")
    values_per_sample = item.structure_size // np.dtype(dtype).itemsize
    if values_per_sample != 3:
        raise RuntimeError(
            f"Expected three values per {item.key} sample, found {values_per_sample}."
        )
    values = np.frombuffer(item.payload, dtype=dtype).astype(np.float64).reshape(-1, 3)
    if not scales:
        scales = [1.0]
    scale_array = np.asarray(scales, dtype=np.float64)
    if scale_array.size == 1:
        values /= scale_array[0]
    elif scale_array.size == 3:
        values /= scale_array
    else:
        raise RuntimeError(f"Unexpected {item.key} scale count: {scale_array.size}.")

    clean_orientation = "".join(char for char in orientation.upper() if char in "XYZ")
    if len(clean_orientation) != 3 or len(set(clean_orientation)) != 3:
        clean_orientation = "XYZ"
    reordered = np.empty_like(values)
    for source_index, axis in enumerate(clean_orientation):
        reordered[:, "XYZ".index(axis)] = values[:, source_index]
    return reordered


def packet_bytes(hex_dump: str) -> bytes:
    chunks: list[str] = []
    for line in hex_dump.splitlines():
        _, separator, remainder = line.partition(":")
        if not separator:
            continue
        hex_part = remainder.split("  ", 1)[0].strip().replace(" ", "")
        if hex_part:
            chunks.append(hex_part)
    return bytes.fromhex("".join(chunks))


def ffprobe_packets(video: Path) -> list[dict[str, str]]:
    stream_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_tag_string",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if stream_probe.returncode != 0:
        raise RuntimeError(stream_probe.stderr.strip() or "ffprobe failed")
    streams = json.loads(stream_probe.stdout).get("streams", [])
    gpmd_indices = [
        stream.get("index")
        for stream in streams
        if stream.get("codec_tag_string", "").lower() == "gpmd"
    ]
    if not gpmd_indices:
        raise RuntimeError("No gpmd telemetry stream was found in the video.")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        str(gpmd_indices[0]),
        "-show_packets",
        "-show_data",
        "-show_entries",
        "packet=pts_time,duration_time,data",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    packets = json.loads(result.stdout).get("packets", [])
    if not packets:
        raise RuntimeError("No GoPro GPMF packets were found.")
    return packets


def extract_imu(video: Path):
    import numpy as np

    accel_times: list[np.ndarray] = []
    accel_values: list[np.ndarray] = []
    gyro_times: list[np.ndarray] = []
    gyro_values: list[np.ndarray] = []

    packets = ffprobe_packets(video)
    for packet in packets:
        start = float(packet.get("pts_time", 0.0))
        duration = float(packet.get("duration_time", 0.0))
        raw_packet = packet_bytes(packet.get("data", ""))
        for stream in stream_payloads(raw_packet):
            fields = list(iter_klv(stream))
            scales: list[float] = []
            orientation = "XYZ"
            sensor: Klv | None = None
            for field in fields:
                if field.key == "SCAL":
                    scales = decode_scale(field)
                elif field.key == "ORIN":
                    orientation = decode_text(field)
                elif field.key in {"ACCL", "GYRO"}:
                    sensor = field
            if sensor is None:
                continue
            values = decode_xyz(sensor, scales, orientation)
            sample_times = start + np.arange(len(values), dtype=np.float64) * (
                duration / len(values)
            )
            if sensor.key == "ACCL":
                accel_times.append(sample_times)
                accel_values.append(values)
            else:
                gyro_times.append(sample_times)
                gyro_values.append(values)

    if not accel_values or not gyro_values:
        raise RuntimeError("The GPMF stream did not contain both ACCL and GYRO samples.")
    return (
        np.concatenate(accel_times),
        np.concatenate(accel_values),
        np.concatenate(gyro_times),
        np.concatenate(gyro_values),
    )


def moving_average(values, samples: int):
    import numpy as np

    samples = max(1, int(samples))
    left = samples // 2
    right = samples - 1 - left
    padded = np.pad(values, ((left, right), (0, 0)), mode="edge")
    kernel = np.full(samples, 1.0 / samples)
    return np.column_stack(
        [np.convolve(padded[:, index], kernel, mode="valid") for index in range(values.shape[1])]
    )


def write_csv(path: Path, accel_t, accel, gyro_t, gyro) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor", "time_s", "x", "y", "z", "magnitude"])
        for sensor, times, values in (
            ("accelerometer", accel_t, accel),
            ("gyroscope", gyro_t, gyro),
        ):
            magnitudes = (values * values).sum(axis=1) ** 0.5
            for time_s, xyz, magnitude in zip(times, values, magnitudes):
                writer.writerow([sensor, time_s, *xyz, magnitude])


def plot_imu(output: Path, video: Path, accel_t, accel, gyro_t, gyro) -> dict[str, float]:
    import numpy as np

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accel_rate = len(accel_t) / max(accel_t[-1] - accel_t[0], 1e-9)
    gyro_rate = len(gyro_t) / max(gyro_t[-1] - gyro_t[0], 1e-9)
    gravity_estimate = moving_average(accel, round(accel_rate))
    dynamic_accel = np.linalg.norm(accel - gravity_estimate, axis=1)
    accel_magnitude = np.linalg.norm(accel, axis=1)
    gyro_magnitude = np.linalg.norm(gyro, axis=1)

    accel_smooth = moving_average(dynamic_accel[:, None], round(accel_rate * 0.1))[:, 0]
    gyro_smooth = moving_average(gyro_magnitude[:, None], round(gyro_rate * 0.1))[:, 0]

    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True, constrained_layout=True)
    colors = ("#1677b8", "#d1495b", "#2a9d55")
    for index, label in enumerate("XYZ"):
        axes[0].plot(accel_t, accel[:, index], color=colors[index], lw=0.55, label=label)
    axes[0].set_ylabel("Acceleration\n(m/s²)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].set_title(f"GoPro IMU: {video.name}")

    axes[1].plot(accel_t, accel_magnitude, color="#555555", lw=0.55, label="|a|")
    axes[1].plot(accel_t, accel_smooth, color="#e76f00", lw=0.9, label="dynamic acceleration")
    axes[1].axhline(9.80665, color="#777777", ls="--", lw=0.7, label="1 g")
    axes[1].set_ylabel("Acceleration\nactivity (m/s²)")
    axes[1].legend(loc="upper right", ncol=3)

    for index, label in enumerate("XYZ"):
        axes[2].plot(gyro_t, gyro[:, index], color=colors[index], lw=0.55, label=label)
    axes[2].set_ylabel("Angular rate\n(rad/s)")
    axes[2].legend(loc="upper right", ncol=3)

    axes[3].plot(gyro_t, gyro_smooth, color="#6a3d9a", lw=0.9)
    axes[3].set_ylabel("Angular activity\n(rad/s)")
    axes[3].set_xlabel("Video time (seconds)")

    for axis in axes:
        axis.grid(True, color="#d9d9d9", lw=0.5)
        axis.set_facecolor("#fafafa")
        axis.set_xlim(0, max(accel_t[-1], gyro_t[-1]))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {
        "accel_rate_hz": accel_rate,
        "gyro_rate_hz": gyro_rate,
        "dynamic_accel_median": float(np.median(accel_smooth)),
        "dynamic_accel_p95": float(np.percentile(accel_smooth, 95)),
        "dynamic_accel_max": float(np.max(accel_smooth)),
        "gyro_median": float(np.median(gyro_smooth)),
        "gyro_p95": float(np.percentile(gyro_smooth, 95)),
        "gyro_max": float(np.max(gyro_smooth)),
    }


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"Error: video does not exist: {video}", file=sys.stderr)
        return 2
    if shutil.which("ffprobe") is None:
        print("Error: ffprobe was not found in PATH.", file=sys.stderr)
        return 2

    output = (
        args.output.expanduser().resolve()
        if args.output
        else Path.cwd() / "imu_analysis" / f"{video.stem}_imu.png"
    )
    try:
        print(f"[INFO] Extracting GPMF IMU data from {video.name}...")
        accel_t, accel, gyro_t, gyro = extract_imu(video)
        stats = plot_imu(output, video, accel_t, accel, gyro_t, gyro)
        if args.csv:
            csv_path = args.csv.expanduser().resolve()
            write_csv(csv_path, accel_t, accel, gyro_t, gyro)
            print(f"[OK] Wrote {csv_path}")
        print(f"[OK] Wrote {output}")
        print(
            "[INFO] samples: "
            f"accelerometer={len(accel_t)} ({stats['accel_rate_hz']:.1f} Hz), "
            f"gyroscope={len(gyro_t)} ({stats['gyro_rate_hz']:.1f} Hz)"
        )
        print(
            "[INFO] dynamic acceleration median/p95/max: "
            f"{stats['dynamic_accel_median']:.3f} / "
            f"{stats['dynamic_accel_p95']:.3f} / "
            f"{stats['dynamic_accel_max']:.3f} m/s²"
        )
        print(
            "[INFO] angular activity median/p95/max: "
            f"{stats['gyro_median']:.4f} / {stats['gyro_p95']:.4f} / "
            f"{stats['gyro_max']:.4f} rad/s"
        )
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
