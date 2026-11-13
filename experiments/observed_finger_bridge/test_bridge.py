"""Geometric and missing-data checks, without physiological claims."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .bridge import CAMERA_IDS, HAND_AXES, TRIAL, audit_multiview, contact_input_adapter, contiguous_velocity, longest_streak, main, project, read_inference, read_prime_inputs, summary, triangulate_pair


class BridgeTests(unittest.TestCase):
    def setUp(self):
        k = np.array([[1000., 0, 500], [0, 1000., 300], [0, 0, 1]])
        self.cameras = [(k.copy(), np.eye(3), np.array([x, 0., 0.])) for x in (0., -.7, .6)]
        self.frames = np.array([973, 974, 975, 976])
        hand = np.column_stack((np.linspace(-.1, .1, 21), np.sin(np.arange(21)) * .04, np.full(21, 3.)))
        self.xyz = np.stack([hand + [0, .01 * i, .02 * i] for i in range(4)])
        self.pixels = np.array([[project(points, camera)[0] for points in self.xyz] for camera in self.cameras])
        self.wrist = self.xyz[:, 0]

    def test_projection_and_pair_recover_metric_points(self):
        reconstructed = triangulate_pair(self.pixels[:2, 0], self.cameras[:2])
        np.testing.assert_allclose(reconstructed, self.xyz[0], atol=1e-12)
        self.assertTrue((project(reconstructed, self.cameras[2])[1] > 0).all())

    def test_projection_preserves_missing_and_rejects_shapes(self):
        uv, depth = project(np.array([[np.nan, 0, 1], [0, 0, 0]]), self.cameras[0])
        self.assertTrue(np.isnan(uv).all())
        self.assertEqual(depth[1], 0)
        with self.assertRaises(ValueError):
            project(np.ones(3), self.cameras[0])

    def test_pair_requires_baseline(self):
        with self.assertRaises(ValueError):
            triangulate_pair(self.pixels[:2, 0], [self.cameras[0]] * 2)

    def test_pair_missing_not_interpolated(self):
        pixel = self.pixels[:2, 0].copy()
        pixel[0, 8] = np.nan
        xyz = triangulate_pair(pixel, self.cameras[:2])
        self.assertTrue(np.isnan(xyz[8]).all())
        self.assertEqual(np.isfinite(xyz).all(axis=1).sum(), 20)

    def test_exact_observations_pass_but_not_body_dynamics(self):
        report, xyz = audit_multiview(self.frames, self.pixels, self.cameras, self.wrist)
        self.assertTrue(report["all_pairs_image_geometry_candidates"])
        self.assertFalse(report["body_attachment_validated"])
        self.assertFalse(report["dynamics_executed"])
        self.assertIsNone(report["prediction"]["speed_kmh"])
        self.assertIsNone(report["same_pitch_record"]["rpm"])
        self.assertEqual(report["pairs"][0]["consecutive_pass_frames"], 4)
        np.testing.assert_allclose(xyz[0], self.xyz, atol=1e-12)

    def test_held_view_changes_never_move_fit_points(self):
        original, xyz = audit_multiview(self.frames, self.pixels, self.cameras, self.wrist)
        pixels = self.pixels.copy()
        pixels[2, :, :, 1] += 30.
        result, changed = audit_multiview(self.frames, pixels, self.cameras, self.wrist)
        np.testing.assert_array_equal(xyz[0], changed[0])
        self.assertFalse(result["pairs"][0]["image_geometry_candidate"])
        self.assertAlmostEqual(result["pairs"][0]["heldout_error_px"]["median"], 30.)
        self.assertTrue(original["pairs"][0]["image_geometry_candidate"])

    def test_one_bad_fingertip_blocks_full_frame_gate(self):
        pixels = self.pixels.copy()
        pixels[2, :, 8, 1] += 11.
        report, _ = audit_multiview(self.frames, pixels, self.cameras, self.wrist)
        self.assertFalse(report["pairs"][0]["image_geometry_candidate"])
        self.assertLess(report["pairs"][0]["heldout_error_px"]["median"], 5.)

    def test_missing_held_points_increase_missing_not_accuracy(self):
        pixels = self.pixels.copy()
        pixels[2, :, 8] = np.nan
        report, _ = audit_multiview(self.frames, pixels, self.cameras, self.wrist)
        self.assertEqual(report["pairs"][0]["frames"][0]["heldout_points"], 20)
        self.assertFalse(report["pairs"][0]["image_geometry_candidate"])

    def test_negative_depth_rejected(self):
        behind = self.xyz.copy()
        behind[:, :, 2] = -3.
        pixels = np.array([[project(points, camera)[0] for points in behind] for camera in self.cameras])
        report, _ = audit_multiview(self.frames, pixels, self.cameras, behind[:, 0])
        self.assertFalse(report["any_image_geometry_candidate"])
        self.assertEqual(report["pairs"][0]["frames"][0]["positive_depth_points_all_three_views"], 0)

    def test_source_wrist_difference_does_not_snap_candidate(self):
        report, xyz = audit_multiview(self.frames, self.pixels, self.cameras, self.wrist + [.02, 0, 0])
        self.assertAlmostEqual(report["pairs"][0]["source_wrist_difference_mm"]["median"], 20.)
        np.testing.assert_allclose(xyz[0], self.xyz, atol=1e-12)
        self.assertFalse(report["body_attachment_validated"])

    def test_streak_requires_original_adjacent_frames(self):
        self.assertEqual(longest_streak([1, 2, 4, 5], [True] * 4), 2)
        self.assertEqual(longest_streak([1, 2, 3, 4], [True, False, True, True]), 2)
        with self.assertRaises(ValueError):
            longest_streak([1, 1], [True, True])

    def test_audit_rejects_schema_and_invented_frames(self):
        with self.assertRaises(ValueError):
            audit_multiview(self.frames, self.pixels[:, :, :3], self.cameras, self.wrist)
        with self.assertRaises(ValueError):
            audit_multiview(self.frames.astype(float) + .2, self.pixels, self.cameras, self.wrist)

    def test_summary_absent_is_null(self):
        self.assertEqual(summary([np.nan]), {"n": 0, "median": None, "max": None})
        self.assertEqual(summary([1, 3, np.nan]), {"n": 2, "median": 2., "max": 3.})

    def test_inference_adapter_preserves_absent_view_and_ignores_world(self):
        clips = []
        for c, camera in enumerate(CAMERA_IDS):
            rows = [{"frame": int(frame), "selected_index": 0,
                     "candidates": [{"eligible": True, "pixel21": self.pixels[c, i].tolist(),
                                     "world21": [[1000., 1000., 1000.]] * 21}]}
                    for i, frame in enumerate(self.frames)]
            if c == 2:
                rows = rows[1:]
            clips.append({"clip_id": f"prime{camera}", "methods": {"raw": rows}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.json"
            path.write_text(json.dumps({"clips": clips}))
            frames, pixels = read_inference(path, "raw")
            np.testing.assert_array_equal(frames, self.frames)
            self.assertTrue(np.isnan(pixels[2, 0]).all())
            np.testing.assert_array_equal(pixels[0], self.pixels[0])
            clips[0]["methods"]["raw"][0]["candidates"][0]["eligible"] = False
            path.write_text(json.dumps({"clips": clips}))
            with self.assertRaisesRegex(ValueError, "Ineligible"):
                read_inference(path, "raw")

    def test_inference_adapter_rejects_missing_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.json"
            path.write_text('{"clips": []}')
            with self.assertRaisesRegex(ValueError, "Missing same-pitch"):
                read_inference(path, "raw")

    def test_local_frame_roundtrip_preserves_rotated_body(self):
        theta = np.array([0., .2, .4, .6])
        rotations = np.array([[[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]] for t in theta])
        local = self.xyz[0] - self.xyz[0, 0]
        world = self.wrist[:, None] + np.einsum("nij,kj->nki", rotations, local)
        contract, arrays = contact_input_adapter(self.frames, world, self.wrist, rotations)
        np.testing.assert_allclose(arrays["landmarks_local_m"], np.broadcast_to(local, (4, 21, 3)), atol=1e-12)
        np.testing.assert_allclose(arrays["fingertip_local_velocity_m_s"], 0., atol=1e-10)
        model_world = self.wrist[:, None] + np.einsum("nij,nkj->nki", rotations @ HAND_AXES,
                                                                     arrays["fingertip_candidates_model_local_m"])
        np.testing.assert_allclose(model_world, world[:, [8, 12]], atol=1e-12)
        self.assertLess(contract["world_to_local_to_world_difference_mm"]["max"], 1e-10)
        self.assertIsNone(contract["contact_pad_radius_m"])
        self.assertFalse(contract["executed"])

    def test_local_frame_rejects_reflection(self):
        rotations = np.broadcast_to(np.diag([-1., 1., 1.]), (4, 3, 3))
        with self.assertRaisesRegex(ValueError, "Proper source"):
            contact_input_adapter(self.frames, self.xyz, self.wrist, rotations)

    def test_velocity_does_not_cross_missing_frame_or_missing_tip(self):
        position = np.zeros((5, 2, 3))
        position[:, :, 0] = np.array([0., 1., 300., 400., 500.])[:, None]
        position[1, 1] = np.nan
        velocity = contiguous_velocity(np.array([1, 2, 4, 5, 7]), position, 360.)
        np.testing.assert_allclose(velocity[:2, 0, 0], 360.)
        np.testing.assert_allclose(velocity[2:4, 0, 0], 36000.)
        self.assertTrue(np.isnan(velocity[4]).all())
        self.assertTrue(np.isnan(velocity[:2, 1]).all())

    def test_unseen_body_frame_does_not_create_local_fingers(self):
        origins = self.wrist.copy()
        origins[1] = np.nan
        rotations = np.broadcast_to(np.eye(3), (4, 3, 3))
        _, arrays = contact_input_adapter(self.frames, self.xyz, origins, rotations)
        self.assertTrue(np.isnan(arrays["landmarks_local_m"][1]).all())
        self.assertTrue(np.isnan(arrays["fingertip_local_velocity_m_s"][:2]).all())
        self.assertTrue(np.isfinite(arrays["fingertip_local_velocity_m_s"][2:]).all())

    def test_cli_writes_actual_array_and_no_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, result = root / "input.npz", root / "bridge.json"
            np.savez(source, frames=self.frames, pixels=self.pixels, camera_ids=CAMERA_IDS)
            args = ["bridge", "--lab", directory, "--input", str(source), "--output", str(result)]
            with patch.object(sys, "argv", args), patch("experiments.observed_finger_bridge.bridge.read_prime_inputs",
                                                       return_value=(self.cameras, self.wrist, {"rig_status": "synthetic_test"},
                                                                     np.broadcast_to(np.eye(3), (4, 3, 3)))):
                main()
            output = json.loads(result.read_text())
            self.assertFalse(output["dynamics_executed"])
            self.assertEqual(output["input_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            with np.load(result.with_suffix(".private.npz")) as arrays:
                self.assertEqual(arrays["points"].shape, (3, 4, 21, 3))

    def test_source_hash_and_time_identity_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger_path = root / "hand-ball-link/artifacts/hand-ball/observation.json"
            ledger_path.parent.mkdir(parents=True)
            paths = {"body": root / "tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d",
                     "alignment": root / "openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json",
                     "ball": root / "speed-journey/artifacts/release-observation/candidate_audit.json"}
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            ledger = {"trial": TRIAL, "source_sha256": {key: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                                        for key, p in paths.items()}}
            ledger_path.write_text(json.dumps(ledger))
            paths["body"].write_text("changed")
            with self.assertRaisesRegex(ValueError, "source hash changed"):
                read_prime_inputs(root, [973])
            paths["body"].write_text("{}")
            alignment = {"trial": TRIAL, "c3d_frame_minus_video_frame": 0}
            paths["alignment"].write_text(json.dumps(alignment))
            ledger["source_sha256"]["alignment"]["sha256"] = hashlib.sha256(paths["alignment"].read_bytes()).hexdigest()
            ledger_path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "identity/time"):
                read_prime_inputs(root, [973])

    def test_cli_failed_geometry_cannot_enter_contact_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, result = root / "input.npz", root / "bridge.json"
            pixels = self.pixels.copy()
            pixels[2, :, :, 1] += 30.
            np.savez(source, frames=self.frames, pixels=pixels, camera_ids=CAMERA_IDS)
            args = ["bridge", "--lab", directory, "--input", str(source), "--output", str(result)]
            with patch.object(sys, "argv", args), patch("experiments.observed_finger_bridge.bridge.read_prime_inputs",
                                                       return_value=(self.cameras, self.wrist, {"rig_status": "synthetic_test"},
                                                                     np.broadcast_to(np.eye(3), (4, 3, 3)))):
                main()
            output = json.loads(result.read_text())
            self.assertEqual(output["contact_input_contracts"][0]["input_geometry_accepted_frames"], 0)
            self.assertEqual(output["contact_input_contracts"][0]["status"], "prepared_not_run_geometry_gate")
            with np.load(result.with_suffix(".private.npz")) as arrays:
                self.assertTrue(np.isfinite(arrays["points"][0]).all())
                self.assertTrue(np.isnan(arrays["fingertip_candidates_model_local_m"][0]).all())


if __name__ == "__main__":
    unittest.main()
