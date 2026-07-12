"""Deterministic coarse-to-fine painted-marker detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .colormodel import MarkerColorModel


@dataclass(frozen=True)
class Detection:
    """One full-resolution marker detection."""

    u: float
    v: float
    area_px: int
    solidity: float
    quality: float


def _parse_roi(roi: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    if isinstance(roi, dict):
        if not roi.get("crop_applied", True) and not any(k in roi for k in ("x", "left", "x0")):
            return 0, 0, width, height
        x = roi.get("x", roi.get("left", roi.get("x0", 0)))
        y = roi.get("y", roi.get("top", roi.get("y0", 0)))
        w = roi.get("width", roi.get("w"))
        h = roi.get("height", roi.get("h"))
        if w is None:
            w = roi.get("x1", width) - x
        if h is None:
            h = roi.get("y1", height) - y
    else:
        x, y, w, h = roi
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1 = min(width, x0 + max(0, int(round(w))))
    y1 = min(height, y0 + max(0, int(round(h))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI is empty or outside the image")
    return x0, y0, x1 - x0, y1 - y0


def _coarse_components(score: np.ndarray, threshold: float) -> list[tuple[int, int, int, int]]:
    mask = (score >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= 1:
            components.append((x, y, w, h))
    return components


def detect_markers(
    image_bgr: np.ndarray,
    model: MarkerColorModel,
    roi: Any = None,
    expected_count: int | None = None,
) -> list[Detection]:
    """Detect markers with downscaled proposals and full-resolution refinement."""
    if model.ab_mean is None or model.ab_cov is None:
        raise RuntimeError("Color model has not been fitted")
    height, width = image_bgr.shape[:2]
    roi_x, roi_y, roi_w, roi_h = _parse_roi(roi, width, height)
    image = image_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    scale = 0.25
    coarse_size = (max(1, int(round(roi_w * scale))), max(1, int(round(roi_h * scale))))
    coarse_image = cv2.resize(image, coarse_size, interpolation=cv2.INTER_AREA)
    coarse_score = model.likelihood(coarse_image)
    thresholds = (0.32, 0.24, 0.17, 0.11, 0.07, 0.05)
    candidates_by_threshold = [(threshold, _coarse_components(coarse_score, threshold)) for threshold in thresholds]
    if expected_count is None:
        coarse_threshold, candidates = candidates_by_threshold[1]
    else:
        coarse_threshold, candidates = min(
            candidates_by_threshold,
            key=lambda item: (abs(len(item[1]) - expected_count), -item[0]),
        )
    fine_threshold = max(0.035, min(0.10, coarse_threshold * 0.70))

    area_gate_min = max(3.0, min(model.area_min * 0.55, model.area_median * 0.45))
    area_gate_max = max(model.area_max * 1.8, model.area_median * 2.2)
    solidity_gate = max(0.55, min(0.80, model.solidity_min * 0.85))
    radius = max(7, int(np.ceil(np.sqrt(max(area_gate_max, 9.0) / np.pi) * 2.2)))
    detections: list[Detection] = []
    for cx, cy, cw, ch in candidates:
        center_x = (cx + 0.5 * cw) / scale
        center_y = (cy + 0.5 * ch) / scale
        x0 = max(0, int(np.floor(center_x - radius)))
        y0 = max(0, int(np.floor(center_y - radius)))
        x1 = min(roi_w, int(np.ceil(center_x + radius + 1)))
        y1 = min(roi_h, int(np.ceil(center_y + radius + 1)))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        score = model.likelihood(crop)
        mask = (score >= fine_threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            continue
        crop_center = np.array([(crop.shape[1] - 1) * 0.5, (crop.shape[0] - 1) * 0.5])
        choices = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            ys, xs = np.where(labels == label)
            if area == 0:
                continue
            distance = float(np.hypot(xs.mean() - crop_center[0], ys.mean() - crop_center[1]))
            choices.append((distance, -area, label))
        if not choices:
            continue
        label = min(choices)[2]
        component = labels == label
        area = int(np.count_nonzero(component))
        if area < area_gate_min or area > area_gate_max:
            continue
        ys, xs = np.where(component)
        weights = score[component].astype(np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            continue
        u = float(x0 + np.dot(xs, weights) / weight_sum + roi_x)
        v = float(y0 + np.dot(ys, weights) / weight_sum + roi_y)
        contour_data = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_data[-2]
        solidity = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = float(np.clip(contour_area / max(hull_area, 1.0), 0.0, 1.0))
        if solidity < solidity_gate:
            continue
        chroma_quality = float(np.mean(weights))
        area_ratio = area / max(model.area_median, 1.0)
        area_quality = float(np.exp(-0.5 * (np.log(max(area_ratio, 1e-6)) / 0.7) ** 2))
        quality = float(np.clip(0.55 * chroma_quality + 0.25 * solidity + 0.20 * area_quality, 0.0, 1.0))
        detections.append(Detection(u, v, area, solidity, quality))

    detections.sort(key=lambda item: (item.v, item.u))
    deduplicated: list[Detection] = []
    min_separation = max(2.0, np.sqrt(max(area_gate_min, 1.0) / np.pi))
    for detection in detections:
        duplicate_index = next(
            (i for i, old in enumerate(deduplicated) if np.hypot(detection.u - old.u, detection.v - old.v) < min_separation),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(detection)
        elif detection.quality > deduplicated[duplicate_index].quality:
            deduplicated[duplicate_index] = detection
    return sorted(deduplicated, key=lambda item: (item.v, item.u))
