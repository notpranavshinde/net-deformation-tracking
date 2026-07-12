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
        w = roi.get("x1", width) - x if w is None else w
        h = roi.get("y1", height) - y if h is None else h
    else:
        x, y, w, h = roi
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1 = min(width, x0 + max(0, int(round(w))))
    y1 = min(height, y0 + max(0, int(round(h))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI is empty or outside the image")
    return x0, y0, x1 - x0, y1 - y0


def _coarse_components(
    score: np.ndarray, threshold: float, max_area: float | None = None
) -> list[tuple[int, int, int, int, float]]:
    mask = (score >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 1 or (max_area is not None and area > max_area):
            continue
        peak = float(np.max(score[labels == label]))
        result.append((x, y, w, h, peak))
    return result


def _gates(model: MarkerColorModel, tier: str) -> tuple[float, float, float, float, int]:
    if tier == "window":
        area_min = max(3.0, min(model.area_min * 0.25, model.area_median * 0.20))
        area_max = max(model.area_max * 2.5, model.area_median * 4.5)
        solidity = max(0.30, min(0.50, model.solidity_min * 0.60))
        aspect_max = 4.0
    elif tier == "rescue":
        area_min = max(3.0, min(model.area_min * 0.25, model.area_median * 0.20))
        area_max = max(model.area_max * 1.8, model.area_median * 2.2)
        solidity = max(0.50, min(0.70, model.solidity_min * 0.75))
        aspect_max = 2.5
    else:
        area_min = max(3.0, min(model.area_min * 0.55, model.area_median * 0.45))
        area_max = max(model.area_max * 1.8, model.area_median * 2.2)
        solidity = max(0.55, min(0.80, model.solidity_min * 0.85))
        aspect_max = float("inf")
    radius = max(7, int(np.ceil(np.sqrt(max(area_max, 9.0) / np.pi) * 2.2)))
    return area_min, area_max, solidity, aspect_max, radius


def _fine_detection(
    crop: np.ndarray,
    model: MarkerColorModel,
    center_xy: tuple[float, float],
    offset_xy: tuple[int, int],
    threshold: float,
    tier: str,
) -> Detection | None:
    if crop.size == 0:
        return None
    score, chroma, core = model.likelihood_components(crop)
    # The core is admitted only through the model's chroma-adjacency gate.  The
    # explicit union keeps centroiding from following only the orange rim.
    mask = (chroma >= threshold) | (core >= max(0.08, threshold))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    cx, cy = center_xy
    choices = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        lx = stats[label, cv2.CC_STAT_LEFT] + 0.5 * stats[label, cv2.CC_STAT_WIDTH]
        ly = stats[label, cv2.CC_STAT_TOP] + 0.5 * stats[label, cv2.CC_STAT_HEIGHT]
        choices.append((float(np.hypot(lx - cx, ly - cy)), -area, label))
    label = min(choices)[2]
    component = labels == label
    area = int(stats[label, cv2.CC_STAT_AREA])
    area_min, area_max, solidity_gate, aspect_max, _ = _gates(model, tier)
    if area < area_min or area > area_max:
        return None

    contours = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = float(np.clip(contour_area / max(hull_area, 1.0), 0.0, 1.0))
    _, _, bw, bh = cv2.boundingRect(contour)
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if solidity < solidity_gate or aspect > aspect_max:
        return None

    ys, xs = np.where(component)
    # Give every accepted union pixel a floor weight so a saturated center is
    # represented geometrically even when its chroma probability is modest.
    weights = np.maximum(score[component].astype(np.float64), 0.22)
    weight_sum = float(weights.sum())
    ox, oy = offset_xy
    u = float(ox + np.dot(xs, weights) / weight_sum)
    v = float(oy + np.dot(ys, weights) / weight_sum)
    response_quality = float(np.mean(score[component]))
    area_ratio = area / max(model.area_median, 1.0)
    area_quality = float(np.exp(-0.5 * (np.log(max(area_ratio, 1e-6)) / 0.7) ** 2))
    quality = float(np.clip(0.55 * response_quality + 0.25 * solidity + 0.20 * area_quality, 0.0, 1.0))
    if tier != "standard":
        quality *= 0.65
    return Detection(u, v, area, solidity, quality)


def _deduplicate(detections: list[Detection], min_separation: float) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: (item.v, item.u)):
        duplicate = next(
            (i for i, old in enumerate(kept) if np.hypot(detection.u - old.u, detection.v - old.v) < min_separation),
            None,
        )
        if duplicate is None:
            kept.append(detection)
        elif detection.quality > kept[duplicate].quality:
            kept[duplicate] = detection
    return sorted(kept, key=lambda item: (item.v, item.u))


def detect_in_window(
    image_bgr: np.ndarray,
    model: MarkerColorModel,
    center_uv: tuple[float, float] | np.ndarray,
    radius: int | float,
    relaxed: bool = True,
) -> Detection | None:
    """Refine one marker in a window centered on a tracker prediction."""
    height, width = image_bgr.shape[:2]
    u, v = (float(value) for value in center_uv)
    r = max(2, int(np.ceil(radius)))
    x0, y0 = max(0, int(np.floor(u - r))), max(0, int(np.floor(v - r)))
    x1, y1 = min(width, int(np.ceil(u + r + 1))), min(height, int(np.ceil(v + r + 1)))
    crop = image_bgr[y0:y1, x0:x1]
    threshold = 0.018 if relaxed else 0.075
    tier = "window" if relaxed else "standard"
    return _fine_detection(crop, model, (u - x0, v - y0), (x0, y0), threshold, tier)


def detect_markers(
    image_bgr: np.ndarray,
    model: MarkerColorModel,
    roi: Any = None,
    expected_count: int | None = None,
) -> list[Detection]:
    """Detect markers, with an expected-count-only low-response rescue pass."""
    if model.ab_mean is None or model.ab_cov is None:
        raise RuntimeError("Color model has not been fitted")
    height, width = image_bgr.shape[:2]
    roi_x, roi_y, roi_w, roi_h = _parse_roi(roi, width, height)
    image = image_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    scale = 0.25
    coarse_size = (max(1, int(round(roi_w * scale))), max(1, int(round(roi_h * scale))))
    coarse_score = model.likelihood(cv2.resize(image, coarse_size, interpolation=cv2.INTER_AREA))
    coarse_threshold = 0.24 if expected_count is None else 0.17
    candidates = _coarse_components(coarse_score, coarse_threshold)
    fine_threshold = max(0.035, min(0.10, coarse_threshold * 0.70))
    _, _, _, _, radius = _gates(model, "standard")
    detections: list[Detection] = []
    failed: list[tuple[int, int, int, int, float]] = []

    def refine(candidate, threshold, tier):
        cx, cy, cw, ch, _ = candidate
        center_x, center_y = (cx + 0.5 * cw) / scale, (cy + 0.5 * ch) / scale
        x0, y0 = max(0, int(np.floor(center_x - radius))), max(0, int(np.floor(center_y - radius)))
        x1, y1 = min(roi_w, int(np.ceil(center_x + radius + 1))), min(roi_h, int(np.ceil(center_y + radius + 1)))
        return _fine_detection(
            image[y0:y1, x0:x1], model, (center_x - x0, center_y - y0),
            (x0 + roi_x, y0 + roi_y), threshold, tier,
        )

    for candidate in candidates:
        detection = refine(candidate, fine_threshold, "standard")
        (detections if detection is not None else failed).append(detection if detection is not None else candidate)

    if expected_count is not None and len(detections) < expected_count:
        local_max = cv2.dilate(coarse_score, np.ones((7, 7), np.uint8))
        low_y, low_x = np.where((coarse_score >= 0.002) & (coarse_score >= local_max - 1e-7))
        low = [(int(x), int(y), 1, 1, float(coarse_score[y, x])) for y, x in zip(low_y, low_x)]
        standard_centers = np.asarray([((x + 0.5 * w), (y + 0.5 * h)) for x, y, w, h, _ in candidates])
        rescue = list(failed)
        for candidate in sorted(low, key=lambda item: -item[4]):
            center = np.array([candidate[0] + 0.5 * candidate[2], candidate[1] + 0.5 * candidate[3]])
            if len(standard_centers) and np.min(np.linalg.norm(standard_centers - center, axis=1)) < 3.0:
                continue
            rescue.append(candidate)
        for candidate in rescue:
            detection = refine(candidate, 0.018, "rescue")
            if detection is not None:
                detections.append(detection)

    area_min = _gates(model, "rescue" if expected_count is not None else "standard")[0]
    result = _deduplicate(detections, max(2.0, np.sqrt(max(area_min, 1.0) / np.pi)))
    if expected_count is not None and len(result) > expected_count:
        result = sorted(result, key=lambda item: item.quality, reverse=True)[:expected_count]
    return sorted(result, key=lambda item: (item.v, item.u))
