"""Small stereo geometry helpers shared by real and synthetic tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsic and world-to-camera extrinsic camera parameters."""

    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "K": np.asarray(self.K, dtype=float).tolist(),
            "dist": np.asarray(self.dist, dtype=float).reshape(-1).tolist(),
            "R": np.asarray(self.R, dtype=float).tolist(),
            "t": np.asarray(self.t, dtype=float).reshape(3).tolist(),
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_name: str = "camera") -> "CameraCalibration":
        """Build a camera from a serialized dictionary."""
        return cls(
            name=str(data.get("name", default_name)),
            K=np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
            dist=np.asarray(data.get("dist", data.get("D", [])), dtype=np.float64).reshape(-1),
            R=np.asarray(data.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3),
            t=np.asarray(data.get("t", data.get("T", np.zeros(3))), dtype=np.float64).reshape(3),
            image_size=tuple(int(v) for v in data["image_size"]),
        )


@dataclass(frozen=True)
class StereoCalibration:
    """A pair of calibrated cameras in one world coordinate system."""

    left: CameraCalibration
    right: CameraCalibration
    scale: float = 1.0

    @property
    def image_size(self) -> tuple[int, int]:
        return self.left.image_size

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable synthetic-rig schema."""
        return {"left": self.left.to_dict(), "right": self.right.to_dict(), "scale": self.scale}


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    """Load pipeline ``stereo.npz`` or a synthetic rig/ground-truth JSON file."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            size = tuple(int(v) for v in data["image_size"].tolist())
            left = CameraCalibration(
                "left", data["K1"].astype(np.float64), data["D1"].astype(np.float64).reshape(-1),
                np.eye(3), np.zeros(3), size,
            )
            right = CameraCalibration(
                "right", data["K2"].astype(np.float64), data["D2"].astype(np.float64).reshape(-1),
                data["R"].astype(np.float64), data["T"].astype(np.float64).reshape(3), size,
            )
            scale = float(np.asarray(data["scale"]).reshape(-1)[0]) if "scale" in data else 1.0
        return StereoCalibration(left, right, scale)

    payload = json.loads(path.read_text(encoding="utf-8"))
    rig = payload.get("rig", payload)
    cameras = rig.get("cameras", rig)
    left = CameraCalibration.from_dict(cameras["left"], "left")
    right = CameraCalibration.from_dict(cameras["right"], "right")
    return StereoCalibration(left, right, float(rig.get("scale", 1.0)))


def project_points(points_xyz: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Project world points through a camera, including lens distortion."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(np.asarray(camera.R, dtype=np.float64))
    pixels, _ = cv2.projectPoints(
        points, rvec, np.asarray(camera.t, dtype=np.float64).reshape(3, 1),
        np.asarray(camera.K, dtype=np.float64), np.asarray(camera.dist, dtype=np.float64),
    )
    return pixels.reshape(-1, 2)


def project_point(point_xyz: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Project one world point through a camera with distortion."""
    return project_points(np.asarray(point_xyz).reshape(1, 3), camera)[0]


def undistort_points(
    points_xy: np.ndarray, camera: CameraCalibration, *, normalized: bool = False
) -> np.ndarray:
    """Undistort image points into pixel or normalized camera coordinates."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    projection = None if normalized else np.asarray(camera.K, dtype=np.float64)
    out = cv2.undistortPoints(points, camera.K, camera.dist, P=projection)
    return out.reshape(-1, 2)


def triangulate_point(
    left_xy: np.ndarray, right_xy: np.ndarray, calibration: StereoCalibration
) -> np.ndarray:
    """Linearly triangulate one distorted left/right point pair."""
    left_norm = undistort_points(np.asarray(left_xy).reshape(1, 2), calibration.left, normalized=True)[0]
    right_norm = undistort_points(np.asarray(right_xy).reshape(1, 2), calibration.right, normalized=True)[0]
    p_left = np.column_stack((calibration.left.R, calibration.left.t.reshape(3, 1)))
    p_right = np.column_stack((calibration.right.R, calibration.right.t.reshape(3, 1)))
    homogeneous = cv2.triangulatePoints(
        p_left, p_right, left_norm.reshape(2, 1), right_norm.reshape(2, 1)
    ).reshape(4)
    if abs(float(homogeneous[3])) < 1e-12:
        raise ValueError("Point pair triangulates at infinity")
    return homogeneous[:3] / homogeneous[3]
