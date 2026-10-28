"""Pinhole camera helpers for the OBP-CV OptiTrack rig.

The published provisional calibration has zero lens distortion for all eight cameras, so
projection is a plain pinhole model. Positions are in millimetres in the camera-19-relative
reference frame of ``optitrack_rig_provisional.json``.
"""
import json
from pathlib import Path

import numpy as np

CAMERAS = tuple(range(15, 23))


def load_cameras(cal_dir, cams=CAMERAS):
    """Return dicts ``K`` (3x3 intrinsics) and ``P`` (3x4 reference-to-camera) per camera."""
    cal_dir = Path(cal_dir)
    intr = json.loads((cal_dir / "calibration_results.json").read_text())["calibrations"]
    pose = json.loads((cal_dir / "optitrack_rig_provisional.json").read_text())["camera_poses"]
    K, P = {}, {}
    for c in cams:
        k = intr[f"optitrack_{c}"]
        if any(k["distortion"]):
            raise ValueError(f"Camera {c} has lens distortion; undistort before triangulating")
        K[c] = np.array([[k["fx_px"], 0.0, k["cx_px"]], [0.0, k["fy_px"], k["cy_px"]], [0.0, 0.0, 1.0]])
        p = pose[str(c)]
        P[c] = np.column_stack((p["reference_to_camera_rotation_matrix"], p["reference_to_camera_translation_mm"]))
    return K, P


def project(point, K, P):
    """Pixel coordinates (u, v) of a 3D point for one camera."""
    q = K @ (P @ np.r_[np.asarray(point, float), 1.0])
    return q[:2] / q[2]


def triangulate(uvs, K, P):
    """Linear (DLT) triangulation from ``{camera: (u, v)}`` with at least two cameras.

    Returns the 3D point and the reprojection error in pixels per camera.
    """
    if len(uvs) < 2:
        raise ValueError("Need at least two views")
    rows = []
    for c, uv in uvs.items():
        q = np.linalg.solve(K[c], np.r_[np.asarray(uv, float), 1.0])
        rows += [q[0] * P[c][2] - P[c][0], q[1] * P[c][2] - P[c][1]]
    X = np.linalg.svd(np.array(rows))[2][-1]
    X = X[:3] / X[3]
    residual = {c: float(np.linalg.norm(project(X, K[c], P[c]) - np.asarray(uv, float))) for c, uv in uvs.items()}
    return X, residual
