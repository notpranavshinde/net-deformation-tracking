"""Load and normalize stereo calibration arrays from ``stereo.npz`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    """Numerical stereo model in the shapes expected by OpenCV."""

    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple[int, ...] | None
    scale: float | None

    @property
    def coordinate_scale(self) -> float:
        """Scale from full-frame track pixels to calibration image pixels."""

        return float(self.scale) if self.scale is not None else 1.0


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    """Load a stereo calibration without changing its numeric precision."""

    stereo = np.load(path, allow_pickle=True)
    image_size = (
        tuple(int(x) for x in stereo["image_size"].tolist())
        if "image_size" in stereo
        else None
    )
    scale = float(stereo["scale"][0]) if "scale" in stereo else None
    return StereoCalibration(
        K1=stereo["K1"].astype(np.float64),
        D1=stereo["D1"].astype(np.float64).reshape(-1, 1),
        K2=stereo["K2"].astype(np.float64),
        D2=stereo["D2"].astype(np.float64).reshape(-1, 1),
        R=stereo["R"].astype(np.float64),
        T=stereo["T"].astype(np.float64).reshape(3, 1),
        image_size=image_size,
        scale=scale,
    )

