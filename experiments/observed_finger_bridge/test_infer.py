"""Coordinate and identity safeguards for hand-only model inference."""
import tempfile
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from . import infer


class InferenceTests(unittest.TestCase):
    def test_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bytes'
            path.write_bytes(b'abc')
            self.assertEqual(infer.sha256(path), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_roi_and_bounds(self):
        self.assertEqual(infer.square_roi([2, 3], 100, (200, 300)), [0, 0, 100, 100])
        self.assertEqual(infer.square_roi([290, 190], 100, (200, 300)), [200, 100, 100, 100])
        with self.assertRaises(ValueError):
            infer.square_roi([1, 2], 201, (200, 300))

    def test_rotation_points_up(self):
        for vector in ([2, 0], [1, 2], [-3, 2], [0, -1]):
            angle = infer.upright_rotation(vector)
            r = cv2.getRotationMatrix2D((0, 0), angle, 1)
            result = r[:, :2] @ vector
            self.assertAlmostEqual(result[0], 0.)
            self.assertLess(result[1], 0.)
        with self.assertRaises(ValueError):
            infer.upright_rotation([0, 0])

    def test_affine_roundtrip_nonzero_rotation_and_resize(self):
        frame = np.zeros((300, 400, 3), np.uint8)
        rgb, transform = infer.transform_input(frame, [55, 75, 144, 144], 73.)
        self.assertEqual(rgb.shape, (256, 256, 3))
        points = np.column_stack((np.linspace(80, 150, 21), np.linspace(90, 170, 21), np.ones(21)))
        processed = points @ transform.T
        normalized = np.column_stack((processed[:, :2]/256., np.zeros(21)))
        np.testing.assert_allclose(infer.restore_pixels(normalized, transform), points[:, :2], atol=1e-12)

    def test_bad_transform_or_landmarks(self):
        with self.assertRaises(ValueError):
            infer.transform_input(np.zeros((10, 10, 3), np.uint8), [0, 0, 11, 11], 0)
        with self.assertRaises(ValueError):
            infer.restore_pixels(np.full((21, 3), np.nan), np.eye(3))

    def test_rejected_candidates_preserved(self):
        candidates = [dict(pixel21=np.tile([x, 50.], (21, 1)).tolist()) for x in (95., 51., 70.)]
        sample = dict(wrist_anchor_px=[50., 50.], wrist_gate_px=10., role='target')
        self.assertEqual(infer.select_candidate(candidates, sample, (100, 100, 3)), 1)
        self.assertFalse(candidates[0]['eligible'])
        self.assertEqual(len(candidates), 3)
        self.assertFalse(candidates[1]['identity_verified'])

    def test_negative_control_and_invalid_wrist(self):
        candidate = dict(pixel21=np.zeros((21, 2)).tolist())
        sample = dict(wrist_anchor_px=[0., 0.], wrist_gate_px=10., role='no_throwing_hand_control')
        self.assertIsNone(infer.select_candidate([candidate], sample, (100, 100, 3)))
        self.assertIn('negative_control_no_throwing_hand', candidate['reasons'])
        candidate = dict(pixel21=[[float('nan'), 0.]]*21)
        self.assertIsNone(infer.select_candidate([candidate], sample, (100, 100, 3)))

    def test_longest_run_respects_frame_gaps(self):
        self.assertEqual(infer.longest_run([5, 5, 7, 8, 9, 12]), 3)
        self.assertEqual(infer.longest_run([]), 0)

    def test_decode_actual_frames_and_missing_required(self):
        class Capture:
            def isOpened(self): return True
            def read(self):
                self.index += 1
                return (True, np.full((2, 2, 3), self.index, np.uint8)) if self.index < 3 else (False, None)
            def release(self): pass
            index = -1
        with patch.object(infer.cv2, 'VideoCapture', side_effect=lambda _: Capture()):
            self.assertEqual([f for f, _ in infer.decode_selected('unused', [0, 2])], [0, 2])
            with self.assertRaises(ValueError):
                list(infer.decode_selected('unused', [4]))

    def test_summary_is_support_not_accuracy(self):
        row = dict(frame=1, role='target', selected_index=0,
                   candidates=[dict(pixel21=np.zeros((21, 2)).tolist())])
        control = dict(frame=30, role='no_throwing_hand_control', selected_index=None, candidates=[{}])
        clips = [dict(clip_id='test', methods={k:[row, control] for k in ('raw', 'rotated')})]
        result = infer.summarize(clips)[0]
        self.assertEqual(result['methods']['raw']['selected_frames'], 1)
        self.assertEqual(result['methods']['raw']['negative_control_detector_hits'], 1)
        self.assertEqual(result['methods']['raw']['negative_control_roi_samples'], 1)
        self.assertEqual(result['methods']['raw']['negative_control_unique_frames'], 1)
        self.assertEqual(result['two_input_agreement']['all21_max_px'], 0)
        self.assertFalse(result['two_input_agreement']['is_ground_truth_accuracy'])

    def test_gif_decode_and_invalid_required_frame(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'clip.gif'
            frames = [Image.fromarray(np.full((4, 4, 3), value, np.uint8)) for value in (30, 90)]
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=100)
            self.assertEqual([f for f, _ in infer.decode_selected(path, [0, 1])], [0, 1])
            with self.assertRaises(ValueError):
                list(infer.decode_selected(path, [2]))

    def test_prepare_and_run_refuse_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'input_manifest.private.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                infer.prepare(out, out)

    def test_run_preserves_all_landmarks_and_refuses_rerun(self):
        point = SimpleNamespace(x=.5, y=.5, z=.01)
        detection = SimpleNamespace(hand_landmarks=[[point]*21],
            hand_world_landmarks=[[point]*21],
            handedness=[[SimpleNamespace(category_name='Left', score=.9)]])
        class Detector:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def detect(self, image): return detection
        mp = SimpleNamespace(__version__='fake-test-only', Image=lambda **kw: kw,
            ImageFormat=SimpleNamespace(SRGB='RGB'), tasks=SimpleNamespace(
                BaseOptions=lambda **kw: kw, vision=SimpleNamespace(
                    RunningMode=SimpleNamespace(IMAGE='image'),
                    HandLandmarkerOptions=lambda **kw: kw,
                    HandLandmarker=SimpleNamespace(create_from_options=lambda options: Detector()))))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = dict(model_path='model', model_sha256=infer.MODEL_SHA256,
                num_hands=2, min_detection_confidence=.5, min_presence_confidence=.5,
                min_tracking_confidence=.5, input_side=256,
                clips=[dict(clip_id='mock', domain='test', release_near_first=0,
                    source_sha256=infer.MODEL_SHA256, path='clip', samples=[dict(
                        frame=0, roi_xywh=[0,0,16,16], wrist_anchor_px=[8,8],
                        wrist_gate_px=10, role='target', rotation_deg=45)])])
            path = out/'manifest.json'
            path.write_text(json.dumps(manifest))
            with patch.dict(sys.modules, {'mediapipe':mp}), patch.object(infer, 'sha256', return_value=infer.MODEL_SHA256), \
                 patch.object(infer, 'decode_selected', return_value=iter([(0, np.zeros((16,16,3), np.uint8))])):
                infer.run(path, out)
                with self.assertRaises(FileExistsError):
                    infer.run(path, out)
            result = json.loads((out/'inference.private.json').read_text())
            candidate = result['clips'][0]['methods']['raw'][0]['candidates'][0]
            self.assertEqual(len(candidate['normalized21']), 21)
            self.assertEqual(len(candidate['world21']), 21)
            self.assertEqual(candidate['handedness'][0]['label'], 'Left')

    def test_main_requires_source_argument(self):
        with patch.object(sys, 'argv', ['infer', 'prepare', '--out', 'unused']):
            with self.assertRaises(SystemExit):
                infer.main()


if __name__ == '__main__':
    unittest.main()
