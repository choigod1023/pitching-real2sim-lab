"""Synthetic checks for the video_track helpers (no OBP data needed)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video_track.camera import load_cameras, project, triangulate
from video_track.detect import detect_ball
from video_track.error_budget import proxy_speed
from video_track.hand_ball_distance import speed_profile


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def synthetic_rig():
    K = np.array([[1300.0, 0, 640.0], [0, 1300.0, 360.0], [0, 0, 1.0]])
    Ks, Ps = {}, {}
    for cam, angle in zip((15, 16, 17), (-0.4, 0.0, 0.5)):
        R = rotation_y(angle)
        Ks[cam] = K
        Ps[cam] = np.column_stack((R, [0.0, 0.0, 6000.0]))
    return Ks, Ps


class CameraTests(unittest.TestCase):
    def test_triangulation_round_trip(self):
        K, P = synthetic_rig()
        X = np.array([120.0, -80.0, 350.0])
        uvs = {c: project(X, K[c], P[c]) for c in K}
        got, residual = triangulate(uvs, K, P)
        np.testing.assert_allclose(got, X, atol=1e-6)
        self.assertLess(max(residual.values()), 1e-6)

    def test_triangulation_needs_two_views(self):
        K, P = synthetic_rig()
        with self.assertRaises(ValueError):
            triangulate({15: np.array([640.0, 360.0])}, K, P)

    def test_distortion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calibration = {"optitrack_15": {"fx_px": 1300, "fy_px": 1300, "cx_px": 640, "cy_px": 360, "distortion": [0.1, 0, 0, 0, 0]}}
            pose = {"15": {"reference_to_camera_rotation_matrix": np.eye(3).tolist(), "reference_to_camera_translation_mm": [0, 0, 1000]}}
            (tmp / "calibration_results.json").write_text(json.dumps({"calibrations": calibration}))
            (tmp / "optitrack_rig_provisional.json").write_text(json.dumps({"camera_poses": pose}))
            with self.assertRaises(ValueError):
                load_cameras(tmp, cams=(15,))


class DetectionTests(unittest.TestCase):
    def blob(self, center, peak=220.0):
        yy, xx = np.mgrid[0:200, 0:300]
        image = 20 + peak * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * 3.0 ** 2))
        return np.clip(image, 0, 255).astype(np.uint8)

    def test_finds_blob_center_with_subpixel_accuracy(self):
        found = detect_ball(self.blob((151.3, 99.6)), expect=(148.0, 103.0))
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found["uv"], (151.3, 99.6), atol=0.4)

    def test_rejects_dim_windows_and_borders(self):
        self.assertIsNone(detect_ball(self.blob((150, 100), peak=40.0), expect=(150, 100)))
        self.assertIsNone(detect_ball(self.blob((10, 10)), expect=(10, 10)))


class SpeedTests(unittest.TestCase):
    def test_speed_profile_of_linear_motion(self):
        v = 100.0 / 3.6  # m/s
        track = {str(f): {"X_mm": [v * 1000 * f / 360.0, 5.0, -3.0]} for f in range(20)}
        profile = speed_profile(track)
        self.assertTrue(profile)
        for value in profile.values():
            self.assertAlmostEqual(value, 100.0, places=6)

    def test_proxy_speed_of_rigid_translation(self):
        rate, n, v = 360.0, 400, 10.0
        t = np.arange(n) / rate
        base = np.column_stack((v * t, np.zeros(n), np.zeros(n)))
        markers = {"RWRA": base + [0, 0.02, 0], "RWRB": base - [0, 0.02, 0], "RFIN": base + [0.1, 0, 0]}
        speed = proxy_speed(markers, rate, release_s=0.5, offset_m=0.2, cutoff_hz=20.0)
        self.assertAlmostEqual(speed, v * 3.6, delta=0.05)


if __name__ == "__main__":
    unittest.main()
