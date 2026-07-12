"""Deterministic deforming-net stereo renderer for tracking tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import CameraCalibration, StereoCalibration, project_points


def _look_at(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return rotation, -rotation @ position


class SyntheticNetScene:
    """Animated marker grid, calibrated stereo rig, and in-memory renderer."""

    def __init__(
        self,
        grid_cols: int,
        grid_rows: int,
        *,
        image_size: tuple[int, int] = (1600, 1200),
        seed: int = 1,
        width_m: float = 1.25,
        height_m: float = 0.90,
        depth_m: float = 3.0,
        baseline_m: float = 0.24,
        billow_amplitude_m: float = 0.075,
        billow_frequency_hz: float = 0.32,
        sway_amplitude_m: float = 0.035,
        sway_frequency_hz: float = 0.11,
        marker_sigma_px: float = 3.2,
        dropout_probability: float = 0.0,
        motion_blur_px: float = 0.0,
    ) -> None:
        if grid_cols < 1 or grid_rows < 1:
            raise ValueError("Grid dimensions must be positive")
        self.grid_cols = int(grid_cols)
        self.grid_rows = int(grid_rows)
        self.image_size = tuple(int(v) for v in image_size)
        self.seed = int(seed)
        self.billow_amplitude_m = float(billow_amplitude_m)
        self.billow_frequency_hz = float(billow_frequency_hz)
        self.sway_amplitude_m = float(sway_amplitude_m)
        self.sway_frequency_hz = float(sway_frequency_hz)
        self.marker_sigma_px = float(marker_sigma_px)
        self.dropout_probability = float(dropout_probability)
        self.motion_blur_px = float(motion_blur_px)
        xs = np.linspace(-width_m / 2.0, width_m / 2.0, grid_cols)
        ys = np.linspace(-height_m / 2.0, height_m / 2.0, grid_rows)
        gx, gy = np.meshgrid(xs, ys)
        self._base_points = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, depth_m)))
        rng = np.random.default_rng(seed)
        self._phases = rng.uniform(0.0, 2.0 * np.pi, gx.size)
        self._color_jitter = rng.normal(0.0, 4.0, (gx.size, 3))
        self.rig = self._make_rig(baseline_m)

    def _make_rig(self, baseline_m: float) -> StereoCalibration:
        width, height = self.image_size
        focal = 0.91 * width
        K = np.array([[focal, 0.0, width / 2.0], [0.0, focal * 1.01, height / 2.0], [0.0, 0.0, 1.0]])
        distortion_left = np.array([-0.075, 0.020, 0.0008, -0.0006, -0.004])
        distortion_right = np.array([-0.068, 0.017, -0.0007, 0.0005, -0.003])
        target = np.array([0.0, 0.0, 3.0])
        left_position = np.array([-baseline_m / 2.0, 0.006, 0.0])
        right_position = np.array([baseline_m / 2.0, -0.004, 0.002])
        left_R, left_t = _look_at(left_position, target)
        right_R, right_t = _look_at(right_position, target)
        left = CameraCalibration("left", K.copy(), distortion_left, left_R, left_t, self.image_size)
        right = CameraCalibration("right", K.copy(), distortion_right, right_R, right_t, self.image_size)
        return StereoCalibration(left, right)

    @property
    def cameras(self) -> tuple[CameraCalibration, CameraCalibration]:
        """Return the left and right synthetic cameras."""
        return self.rig.left, self.rig.right

    def points_3d(self, t: float) -> np.ndarray:
        """Return deformed world-space marker positions at time ``t`` seconds."""
        points = self._base_points.copy()
        x_norm = points[:, 0] / max(np.ptp(self._base_points[:, 0]), 1e-9)
        y_norm = points[:, 1] / max(np.ptp(self._base_points[:, 1]), 1e-9)
        wave = np.sin(2.0 * np.pi * (0.85 * x_norm + 0.35 * y_norm - self.billow_frequency_hz * t) + 0.12 * self._phases)
        envelope = 0.75 + 0.25 * np.cos(np.pi * y_norm)
        points[:, 2] += self.billow_amplitude_m * envelope * wave
        points[:, 0] += self.sway_amplitude_m * np.sin(2.0 * np.pi * self.sway_frequency_hz * t)
        points[:, 1] += 0.012 * np.sin(2.0 * np.pi * 0.17 * t + 1.3 * x_norm)
        return points

    def _dropout_mask(self, camera: CameraCalibration, t: float) -> np.ndarray:
        if self.dropout_probability <= 0.0:
            return np.zeros(len(self._base_points), dtype=bool)
        time_key = int(round(float(t) * 1000.0))
        camera_key = 0 if camera.name == "left" else 1
        rng = np.random.default_rng(self.seed * 1_000_003 + time_key * 97 + camera_key * 7_919)
        return rng.random(len(self._base_points)) < self.dropout_probability

    def ground_truth(self, camera: CameraCalibration, t: float) -> list[dict[str, Any]]:
        """Return marker positions, projections, and visibility for one camera/time."""
        xyz = self.points_3d(t)
        uv = project_points(xyz, camera)
        width, height = self.image_size
        dropped = self._dropout_mask(camera, t)
        visible = (
            (uv[:, 0] >= 8.0) & (uv[:, 0] < width - 8.0)
            & (uv[:, 1] >= 8.0) & (uv[:, 1] < height - 8.0) & ~dropped
        )
        return [
            {
                "marker_id": i,
                "xyz": xyz[i].tolist(),
                "u": float(uv[i, 0]),
                "v": float(uv[i, 1]),
                "visible": bool(visible[i]),
            }
            for i in range(len(xyz))
        ]

    def render_frame(self, cam: CameraCalibration | str, t: float) -> np.ndarray:
        """Render one deterministic BGR frame directly in memory."""
        camera = self.rig.left if cam == "left" else self.rig.right if cam == "right" else cam
        if not isinstance(camera, CameraCalibration):
            raise TypeError("cam must be 'left', 'right', or CameraCalibration")
        width, height = self.image_size
        time_key = int(round(float(t) * 1000.0))
        camera_key = 0 if camera.name == "left" else 1
        rng = np.random.default_rng(self.seed * 2_000_003 + time_key * 193 + camera_key * 65_537)
        small_h, small_w = max(2, height // 24), max(2, width // 24)
        texture = rng.normal(0.0, 1.0, (small_h, small_w)).astype(np.float32)
        texture = cv2.resize(texture, (width, height), interpolation=cv2.INTER_CUBIC)
        fine = rng.normal(0.0, 2.0, (height, width)).astype(np.float32)
        base = np.empty((height, width, 3), dtype=np.float32)
        base[..., 0] = 36.0 + 7.0 * texture + fine
        base[..., 1] = 43.0 + 6.0 * texture + fine
        base[..., 2] = 35.0 + 4.0 * texture + fine
        vertical = np.linspace(4.0, -4.0, height, dtype=np.float32)[:, None]
        base += vertical[..., None]

        truth = self.ground_truth(camera, t)
        radius = int(np.ceil(self.marker_sigma_px * 4.0 + self.motion_blur_px))
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        for item in truth:
            if not item["visible"]:
                continue
            marker_id = int(item["marker_id"])
            u, v = float(item["u"]), float(item["v"])
            ix, iy = int(np.floor(u)), int(np.floor(v))
            dx = xx + ix - u
            dy = yy + iy - v
            if self.motion_blur_px > 0.0:
                gaussian = sum(
                    np.exp(-0.5 * (((dx - shift) / self.marker_sigma_px) ** 2 + (dy / self.marker_sigma_px) ** 2))
                    for shift in np.linspace(-self.motion_blur_px / 2.0, self.motion_blur_px / 2.0, 5)
                ) / 5.0
            else:
                gaussian = np.exp(-0.5 * ((dx / self.marker_sigma_px) ** 2 + (dy / self.marker_sigma_px) ** 2))
            x0, x1 = ix - radius, ix + radius + 1
            y0, y1 = iy - radius, iy + radius + 1
            px0, py0 = max(0, -x0), max(0, -y0)
            px1, py1 = gaussian.shape[1] - max(0, x1 - width), gaussian.shape[0] - max(0, y1 - height)
            x0, x1, y0, y1 = max(0, x0), min(width, x1), max(0, y0), min(height, y1)
            color = np.array([24.0, 118.0, 244.0]) + self._color_jitter[marker_id]
            alpha = (0.97 * gaussian[py0:py1, px0:px1])[..., None]
            base[y0:y1, x0:x1] = base[y0:y1, x0:x1] * (1.0 - alpha) + color * alpha
        return np.clip(base, 0.0, 255.0).astype(np.uint8)

    def write_videos(self, out_dir: str | Path, n_frames: int, fps: float) -> tuple[Path, Path, Path]:
        """Write stereo MP4s and complete rig/per-marker ground truth JSON."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        width, height = self.image_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        paths = (out_dir / "left.mp4", out_dir / "right.mp4")
        writers = [cv2.VideoWriter(str(path), fourcc, float(fps), (width, height)) for path in paths]
        if not all(writer.isOpened() for writer in writers):
            for writer in writers:
                writer.release()
            raise RuntimeError("Could not open synthetic MP4 writers")
        frames = []
        try:
            for frame_index in range(int(n_frames)):
                t = frame_index / float(fps)
                camera_truth = {}
                for camera, writer in zip(self.cameras, writers):
                    writer.write(self.render_frame(camera, t))
                    camera_truth[camera.name] = self.ground_truth(camera, t)
                frames.append({"frame": frame_index, "time_s": t, "cameras": camera_truth})
        finally:
            for writer in writers:
                writer.release()
        truth_path = out_dir / "ground_truth.json"
        payload = {
            "schema_version": 1,
            "grid": {"cols": self.grid_cols, "rows": self.grid_rows},
            "rig": self.rig.to_dict(),
            "fps": float(fps),
            "frames": frames,
        }
        truth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return paths[0], paths[1], truth_path
