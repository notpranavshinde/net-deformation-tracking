"""Shared OpenCV colors and drawing primitives for triangulation visualizations."""

from __future__ import annotations

import cv2
import numpy as np


def color_for_obj(obj_id: int) -> tuple[int, int, int]:
    """Return the deterministic legacy BGR color for an object identifier."""

    rng = np.random.RandomState(int(obj_id))
    return tuple(int(value) for value in rng.randint(50, 255, size=3))


def depth_to_bgr(z: float, z_min: float, z_max: float) -> tuple[int, int, int]:
    """Map a depth value to a BGR color using a viridis-like ramp."""

    if z_max <= z_min:
        t = 0.5
    else:
        t = (z - z_min) / (z_max - z_min)
    t = float(np.clip(t, 0.0, 1.0))
    stops = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 144, 141],
        [94, 201, 98],
        [253, 231, 37],
    ], dtype=np.float32)
    position = t * (len(stops) - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(stops) - 1)
    fraction = position - lower
    rgb = (1 - fraction) * stops[lower] + fraction * stops[upper]
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def draw_value_colorbar(
    image: np.ndarray,
    value_min: float,
    value_max: float,
    x: int,
    y: int,
    w: int = 18,
    h: int = 180,
    label: str = "value",
) -> None:
    """Draw a labeled scalar color bar onto an image."""

    for index in range(h):
        fraction = 1.0 - (index / max(h - 1, 1))
        value = value_min + fraction * (value_max - value_min)
        color = depth_to_bgr(value, value_min, value_max)
        cv2.rectangle(image, (x, y + index), (x + w, y + index + 1), color, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(image, f"{value_max:.3f}", (x + w + 6, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(image, f"{value_min:.3f}", (x + w + 6, y + h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(image, label, (x - 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
