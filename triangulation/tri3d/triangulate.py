"""Stereo undistortion, DLT triangulation, reprojection, and point filtering."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .calib import StereoCalibration
from .matching import Observation, ObservationKey


def undist_norm_points(K: np.ndarray, D: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    """
    pts_uv: (N,2) float in pixel coords
    returns: (N,2) float normalized coords (x,y) on z=1 plane
    """
    pts = pts_uv.reshape(-1, 1, 2).astype(np.float64)
    # undistortPoints returns normalized coords if P=None
    und = cv2.undistortPoints(pts, K, D, P=None)
    return und.reshape(-1, 2)


def project_points(K: np.ndarray, D: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """
    pts_xyz: (N,3) in camera coordinates
    returns: (N,2) pixel coords
    """
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    img, _ = cv2.projectPoints(pts_xyz.astype(np.float64), rvec, tvec, K, D)
    return img.reshape(-1, 2)


def triangulate_observations(
    calibration: StereoCalibration,
    keys: list[ObservationKey],
    left: dict[ObservationKey, Observation],
    right: dict[ObservationKey, Observation],
    *,
    source_start_frame: int,
    frame_step: int,
    max_reproj: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Triangulate matched observations and retain the legacy output schema."""

    K1, D1 = calibration.K1, calibration.D1
    K2, D2 = calibration.K2, calibration.D2
    R, T = calibration.R, calibration.T
    calib_scale = calibration.coordinate_scale
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
    P2 = np.hstack([R, T]).astype(np.float64)
    out_rows: list[dict[str, Any]] = []
    reproj_errs: list[float] = []

    for gframe, obj_id in keys:
        u1, v1, q1, lframe = left[(gframe, obj_id)]
        u2, v2, q2, rframe = right[(gframe, obj_id)]
        pts1 = np.array([[u1 * calib_scale, v1 * calib_scale]], dtype=np.float64)
        pts2 = np.array([[u2 * calib_scale, v2 * calib_scale]], dtype=np.float64)
        x1 = undist_norm_points(K1, D1, pts1)
        x2 = undist_norm_points(K2, D2, pts2)
        Xh = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
        X = (Xh[:3] / Xh[3]).reshape(3)
        ZL = float(X[2])
        XR = (R @ X.reshape(3, 1) + T).reshape(3)
        ZR = float(XR[2])
        p1_hat = project_points(K1, D1, X.reshape(1, 3))[0]
        p2_hat = project_points(K2, D2, XR.reshape(1, 3))[0]
        p1_hat_out = p1_hat / calib_scale
        p2_hat_out = p2_hat / calib_scale
        errL = float(np.linalg.norm(np.array([u1, v1]) - p1_hat_out))
        errR = float(np.linalg.norm(np.array([u2, v2]) - p2_hat_out))
        err = 0.5 * (errL + errR)
        ok = 1
        if ZL <= 0 or ZR <= 0:
            ok = 0
        if err > max_reproj:
            ok = 0
        if ok:
            reproj_errs.append(err)
        out_rows.append({
            "gframe": gframe,
            "frame_L": lframe,
            "frame_R": rframe,
            "source_frame_L": int(source_start_frame) + int(lframe) * int(frame_step),
            "source_frame_R": int(source_start_frame) + int(rframe) * int(frame_step),
            "obj_id": obj_id,
            "uL": float(u1),
            "vL": float(v1),
            "uR": float(u2),
            "vR": float(v2),
            "uL_hat": float(p1_hat_out[0]),
            "vL_hat": float(p1_hat_out[1]),
            "uR_hat": float(p2_hat_out[0]),
            "vR_hat": float(p2_hat_out[1]),
            "X": float(X[0]),
            "Y": float(X[1]),
            "Z": float(X[2]),
            "Z_right": ZR,
            "errL_px": errL,
            "errR_px": errR,
            "err_px": err,
            "qL": q1,
            "qR": q2,
            "valid_3d": ok,
        })
    return out_rows, reproj_errs
