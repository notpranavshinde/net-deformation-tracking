"""Session-adaptive CIELab color model for painted markers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _mad(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - median), axis=axis) * 1.4826


@dataclass
class MarkerColorModel:
    """Robust Gaussian marker chroma model with brightness and area bounds."""

    ab_mean: np.ndarray | None = None
    ab_cov: np.ndarray | None = None
    l_min: float = 0.0
    l_max: float = 255.0
    area_median: float = 0.0
    area_min: float = 0.0
    area_max: float = 0.0
    solidity_median: float = 0.0
    solidity_min: float = 0.0
    patch_radius: int = 0

    def fit(
        self, image_bgr: np.ndarray, points_xy: np.ndarray, patch_radius: int | None = None
    ) -> "MarkerColorModel":
        """Fit marker chroma, luminance, and full-resolution blob-area statistics."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be an HxWx3 BGR image")
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if len(points) == 0:
            raise ValueError("At least one marker point is required")
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]

        center_samples = []
        for x, y in points:
            ix, iy = int(round(x)), int(round(y))
            x0, x1 = max(0, ix - 2), min(w, ix + 3)
            y0, y1 = max(0, iy - 2), min(h, iy + 3)
            if x0 < x1 and y0 < y1:
                center_samples.append(lab[y0:y1, x0:x1].reshape(-1, 3))
        if not center_samples:
            raise ValueError("No marker points fall inside the image")
        centers = np.concatenate(center_samples)
        seed_ab = np.median(centers[:, 1:3], axis=0)
        seed_scale = np.maximum(_mad(centers[:, 1:3], axis=0), 6.0)

        generous = int(patch_radius) if patch_radius is not None else 16
        generous = max(4, generous)
        selected = []
        components: list[np.ndarray] = []
        component_labs: list[np.ndarray] = []
        for x, y in points:
            ix, iy = int(round(x)), int(round(y))
            x0, x1 = max(0, ix - generous), min(w, ix + generous + 1)
            y0, y1 = max(0, iy - generous), min(h, iy + generous + 1)
            patch = lab[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            delta = (patch[..., 1:3] - seed_ab) / seed_scale
            initial = (np.sum(delta * delta, axis=2) <= 12.0).astype(np.uint8)
            n, labels = cv2.connectedComponents(initial, connectivity=8)
            sx = int(np.clip(ix - x0, 0, patch.shape[1] - 1))
            sy = int(np.clip(iy - y0, 0, patch.shape[0] - 1))
            label = int(labels[sy, sx])
            if label == 0:
                nearest = labels[max(0, sy - 2):sy + 3, max(0, sx - 2):sx + 3]
                nonzero = nearest[nearest > 0]
                label = int(np.bincount(nonzero).argmax()) if nonzero.size else 0
            mask = labels == label if label > 0 else initial.astype(bool)
            if np.count_nonzero(mask) >= 3:
                pixels = patch[mask]
                selected.append(pixels)
                components.append(mask)
                component_labs.append(patch)
        if not selected:
            raise ValueError("Could not isolate marker-colored pixels around setup points")

        pixels = np.concatenate(selected)
        ab_median = np.median(pixels[:, 1:3], axis=0)
        ab_scale = np.maximum(_mad(pixels[:, 1:3], axis=0), 4.0)
        robust_distance = np.sum(((pixels[:, 1:3] - ab_median) / ab_scale) ** 2, axis=1)
        kept = pixels[robust_distance <= 9.0]
        if len(kept) < 6:
            kept = pixels
        self.ab_mean = np.median(kept[:, 1:3], axis=0).astype(np.float64)
        centered = kept[:, 1:3].astype(np.float64) - self.ab_mean
        cov = (centered.T @ centered) / max(len(centered) - 1, 1)
        cov += np.eye(2) * 16.0
        self.ab_cov = cov
        l_values = kept[:, 0]
        l_med = float(np.median(l_values))
        l_sigma = max(float(_mad(l_values)), 8.0)
        self.l_min = max(0.0, l_med - 5.0 * l_sigma - 18.0)
        self.l_max = min(255.0, l_med + 5.0 * l_sigma + 18.0)

        areas = []
        solidities = []
        for patch, old_mask in zip(component_labs, components):
            score = self._likelihood_lab(patch)
            mask = (score >= 0.12).astype(np.uint8)
            n, labels = cv2.connectedComponents(mask, connectivity=8)
            cy, cx = np.array(mask.shape) // 2
            label = int(labels[min(cy, labels.shape[0] - 1), min(cx, labels.shape[1] - 1)])
            if label == 0 and np.any(old_mask):
                ys, xs = np.where(old_mask)
                label = int(labels[int(np.median(ys)), int(np.median(xs))])
            component = labels == label if label > 0 else old_mask
            area = int(np.count_nonzero(component))
            if area > 0:
                areas.append(area)
                contours = cv2.findContours(
                    component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[-2]
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    contour_area = float(cv2.contourArea(contour))
                    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
                    solidities.append(float(np.clip(contour_area / max(hull_area, 1.0), 0.0, 1.0)))
        if not areas:
            areas = [int(np.median([np.count_nonzero(m) for m in components]))]
        area_values = np.asarray(areas, dtype=np.float64)
        self.area_median = float(np.median(area_values))
        self.area_min = float(np.min(area_values))
        self.area_max = float(np.max(area_values))
        solidity_values = np.asarray(solidities or [1.0], dtype=np.float64)
        self.solidity_median = float(np.median(solidity_values))
        self.solidity_min = float(np.min(solidity_values))
        derived_radius = int(np.ceil(np.sqrt(self.area_median / np.pi) * 1.8))
        self.patch_radius = int(patch_radius) if patch_radius is not None else max(4, derived_radius)
        return self

    def _likelihood_lab(self, lab: np.ndarray) -> np.ndarray:
        if self.ab_mean is None or self.ab_cov is None:
            raise RuntimeError("Color model has not been fitted")
        ab = lab[..., 1:3].astype(np.float32)
        delta = ab - self.ab_mean.astype(np.float32)
        inverse = np.linalg.inv(self.ab_cov).astype(np.float32)
        distance2 = np.einsum("...i,ij,...j->...", delta, inverse, delta, optimize=True)
        score = np.exp(-0.5 * distance2).astype(np.float32)
        luminance = lab[..., 0]
        score[(luminance < self.l_min) | (luminance > self.l_max)] = 0.0
        return np.clip(score, 0.0, 1.0)

    def likelihood(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a vectorized per-pixel marker likelihood map in ``[0, 1]``."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        return self._likelihood_lab(lab)

    def to_json(self) -> str:
        """Serialize the fitted model to JSON text."""
        if self.ab_mean is None or self.ab_cov is None:
            raise RuntimeError("Color model has not been fitted")
        payload = {
            "version": 1,
            "ab_mean": self.ab_mean.tolist(),
            "ab_cov": self.ab_cov.tolist(),
            "l_range": [self.l_min, self.l_max],
            "area_px": {"median": self.area_median, "min": self.area_min, "max": self.area_max},
            "solidity": {"median": self.solidity_median, "min": self.solidity_min},
            "patch_radius": self.patch_radius,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, value: str | bytes | dict[str, Any]) -> "MarkerColorModel":
        """Restore a model from JSON text or an already-decoded dictionary."""
        payload = value if isinstance(value, dict) else json.loads(value)
        area = payload["area_px"]
        solidity = payload.get("solidity", {"median": 1.0, "min": 0.7})
        return cls(
            ab_mean=np.asarray(payload["ab_mean"], dtype=np.float64),
            ab_cov=np.asarray(payload["ab_cov"], dtype=np.float64),
            l_min=float(payload["l_range"][0]),
            l_max=float(payload["l_range"][1]),
            area_median=float(area["median"]),
            area_min=float(area["min"]),
            area_max=float(area["max"]),
            solidity_median=float(solidity["median"]),
            solidity_min=float(solidity["min"]),
            patch_radius=int(payload["patch_radius"]),
        )
