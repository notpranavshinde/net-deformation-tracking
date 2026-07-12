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
    core_l_min: float = 220.0
    core_l_max: float = 255.0
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
        core_samples = []
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
                # A bloomed marker has an orange component surrounding a bright,
                # weak-chroma core.  Only admit bright pixels close enough to the
                # chroma component to be part of the same physical marker.
                neighborhood = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                local_l = patch[..., 0]
                bright_cutoff = max(205.0, float(np.percentile(local_l[mask], 75)) + 18.0)
                core = neighborhood & (local_l >= bright_cutoff)
                union = mask | core
                pixels = patch[mask]
                selected.append(pixels)
                if np.any(core):
                    core_samples.append(patch[core])
                components.append(union)
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
        if core_samples:
            core_l = np.concatenate(core_samples)[:, 0]
            self.core_l_min = float(np.clip(np.percentile(core_l, 10) - 8.0, 190.0, 245.0))
            self.core_l_max = float(np.clip(np.percentile(core_l, 99) + 3.0, self.core_l_min, 255.0))
        else:
            self.core_l_min = float(np.clip(np.percentile(l_values, 95) + 22.0, 205.0, 238.0))
            self.core_l_max = 255.0

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

    def _score_components_lab(self, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.ab_mean is None or self.ab_cov is None:
            raise RuntimeError("Color model has not been fitted")
        ab = lab[..., 1:3].astype(np.float32)
        delta = ab - self.ab_mean.astype(np.float32)
        inverse = np.linalg.inv(self.ab_cov).astype(np.float32)
        distance2 = np.einsum("...i,ij,...j->...", delta, inverse, delta, optimize=True)
        chroma = np.exp(-0.5 * distance2).astype(np.float32)
        luminance = lab[..., 0].astype(np.float32)
        # Preserve a small, smoothly decaying response below the fitted rim
        # range so the expected-count rescue pass can recover attenuated marks.
        low_gap = np.maximum(self.l_min - luminance, 0.0)
        chroma *= np.exp(-0.5 * (low_gap / 32.0) ** 2).astype(np.float32)
        chroma[luminance > self.l_max + 8.0] *= 0.35

        bright = np.clip(
            (luminance - self.core_l_min) / max(self.core_l_max - self.core_l_min, 12.0) + 0.35,
            0.0,
            1.0,
        ).astype(np.float32)
        near_chroma = cv2.dilate((chroma >= 0.075).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        core = bright * near_chroma
        combined = np.maximum(chroma, core)
        return np.clip(combined, 0.0, 1.0), np.clip(chroma, 0.0, 1.0), core

    def _likelihood_lab(self, lab: np.ndarray) -> np.ndarray:
        return self._score_components_lab(lab)[0]

    def likelihood(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a vectorized per-pixel marker likelihood map in ``[0, 1]``."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        return self._likelihood_lab(lab)

    def likelihood_components(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return combined, orange-rim, and adjacency-gated bright-core scores."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        return self._score_components_lab(lab)

    def to_json(self) -> str:
        """Serialize the fitted model to JSON text."""
        if self.ab_mean is None or self.ab_cov is None:
            raise RuntimeError("Color model has not been fitted")
        payload = {
            "version": 2,
            "ab_mean": self.ab_mean.tolist(),
            "ab_cov": self.ab_cov.tolist(),
            "l_range": [self.l_min, self.l_max],
            "core_l_range": [self.core_l_min, self.core_l_max],
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
        core_l = payload.get("core_l_range", [max(205.0, float(payload["l_range"][1]) - 20.0), 255.0])
        return cls(
            ab_mean=np.asarray(payload["ab_mean"], dtype=np.float64),
            ab_cov=np.asarray(payload["ab_cov"], dtype=np.float64),
            l_min=float(payload["l_range"][0]),
            l_max=float(payload["l_range"][1]),
            core_l_min=float(core_l[0]),
            core_l_max=float(core_l[1]),
            area_median=float(area["median"]),
            area_min=float(area["min"]),
            area_max=float(area["max"]),
            solidity_median=float(solidity["median"]),
            solidity_min=float(solidity["min"]),
            patch_radius=int(payload["patch_radius"]),
        )
