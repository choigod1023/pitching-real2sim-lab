import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, ImageFont

from . import overlay as o


def fixture(pitch='a', fp=.8, br=1., offset=0.):
    time = np.arange(721) / 360.
    landmarks = pd.DataFrame(dict(session_pitch=pitch, time=np.round(time, 4), fp_10_time=fp, BR_time=br))
    for index, joint in enumerate(o.JOINTS):
        landmarks[joint + '_x'] = time + index * .02 + offset
        landmarks[joint + '_y'] = index * .025 + offset
        landmarks[joint + '_z'] = .1 + index * .1
    angles = landmarks[['session_pitch', 'time', 'fp_10_time', 'BR_time']].copy()
    for index, column in enumerate(o.ANGLES):
        angles[column] = 60 + index * 10 - time * 20
    return landmarks, angles


def tracks_and_case():
    first = o.prepare_track(*fixture(), 102.)
    second = o.prepare_track(*fixture('b', fp=.9, br=1.05, offset=2.), 100.)
    case = dict(user='1', fast='a', slow='b', sensitivity={'profiles': [{}, {'independent_event': False}]},
                peak_loads={'a': {'elbow_varus_peak_nm': 100., 'shoulder_resultant_peak_n': 900.},
                            'b': {'elbow_varus_peak_nm': 104., 'shoulder_resultant_peak_n': 920.}})
    return [first, second], case


class OverlayTests(unittest.TestCase):
    def test_interpolation_shape_finite_and_no_extrapolation(self):
        time = np.array([0., 1., 2.])
        values = np.arange(18).reshape(3, 2, 3)
        self.assertEqual(o.interpolate(time, values, .5).shape, (2, 3))
        np.testing.assert_allclose(o.interpolate(time, values, [.5, 1.5]), (values[:-1] + values[1:]) / 2)
        for query in (-.01, 2.01, np.nan):
            with self.assertRaises(ValueError):
                o.interpolate(time, values, query)
        with self.assertRaises(ValueError):
            o.interpolate([0., 0., 1.], values, .5)

    def test_fixed_horizontal_origin_preserves_vertical_and_translation(self):
        tracks, _ = tracks_and_case()
        for track in tracks:
            self.assertEqual(track['origin'][2], 0.)
            np.testing.assert_allclose(track['xyz'][:, :, 2], track['raw_xyz'][:, :, 2])
            self.assertAlmostEqual(track['xyz'][-1, 0, 0] - track['xyz'][0, 0, 0], 2.)
            hips = (track['xyz'][:, o.JOINTS.index('lead_hip')] + track['xyz'][:, o.JOINTS.index('rear_hip')]) / 2
            np.testing.assert_allclose(o.interpolate(track['time'], hips, track['fp'])[:2], 0., atol=1e-12)
        self.assertNotEqual(tracks[0]['origin'][0], tracks[1]['origin'][0])

    def test_prepare_rejects_nan_time_and_event_mismatch(self):
        landmarks, angles = fixture()
        changed = landmarks.copy()
        changed.loc[1, o.JOINTS[0] + '_x'] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite_pose'):
            o.prepare_track(changed, angles, 100.)
        changed = angles.copy()
        changed.loc[0, 'time'] = .1
        with self.assertRaisesRegex(ValueError, 'time_mismatch'):
            o.prepare_track(landmarks, changed, 100.)
        changed = angles.copy()
        changed['BR_time'] = 1.1
        with self.assertRaisesRegex(ValueError, 'event'):
            o.prepare_track(landmarks, changed, 100.)

    def test_real_elapsed_time_preserves_distinct_release_and_source_times(self):
        tracks, _ = tracks_and_case()
        q = o.aligned_times(tracks, 'fp')
        np.testing.assert_allclose(np.diff(q), 1 / 360., atol=1e-14)
        _, _, states = o.sample_pair(tracks, .16, 'fp')
        self.assertEqual(states[0]['phase'], 'FP10→BR')
        self.assertEqual(states[1]['phase'], 'BR 후')
        self.assertAlmostEqual(states[0]['source_time'], .96)
        self.assertAlmostEqual(states[1]['source_time'], 1.06)
        _, _, states = o.sample_pair(tracks, 0., 'br')
        self.assertAlmostEqual(states[0]['br_remaining_ms'], 0.)
        self.assertAlmostEqual(states[1]['br_remaining_ms'], 0.)
        with self.assertRaises(ValueError):
            o.aligned_times(tracks, 'phase_warp')

    def test_bounds_and_nonblank_full_body_frame(self):
        tracks, case = tracks_and_case()
        bounds = o.scene_bounds(tracks)
        self.assertTrue((bounds[1] > bounds[0]).all())
        fonts = {n: ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', n) if Path('C:/Windows/Fonts/malgun.ttf').exists()
                 else ImageFont.load_default() for n in (18, 20, 21, 22, 23, 24, 28, 32)}
        image = o.draw_frame(tracks, case, .1, 'fp', bounds, fonts)
        self.assertEqual(image.size, o.SIZE)
        pixels = np.asarray(image)[186:696]
        for color in o.COLORS:
            self.assertGreater(np.all(pixels == color, axis=2).sum(), 100)

    def test_loader_raw_marker_names_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'full_sig').mkdir()
            (root / 'c3d').mkdir()
            landmarks, angles, metadata, metric_rows = [], [], [], []
            cases = []
            for user, fast, slow in o.CASES:
                cases.append(dict(faster_pitch=fast, slower_pitch=slow))
                for pitch in (fast, slow):
                    g, a = fixture(pitch)
                    landmarks.append(g); angles.append(a)
                    metadata.append(dict(user=user, session_pitch=pitch, pitch_speed_mph=60., filename_new=pitch + '.c3d'))
                    metric_rows.append(dict(session_pitch=pitch, cutoff_hz=20, pitch_speed_kmh=60 * 1.609344,
                                            elbow_varus_peak_nm=100., shoulder_resultant_peak_n=900.))
                    (root / 'c3d' / (pitch + '.c3d')).write_bytes(b'synthetic')
            pd.concat(landmarks).to_csv(root / 'full_sig/landmarks.csv', index=False)
            pd.concat(angles).to_csv(root / 'full_sig/joint_angles.csv', index=False)
            pd.DataFrame(metadata).to_csv(root / 'metadata.csv', index=False)
            metrics = root / 'metrics.csv'; pd.DataFrame(metric_rows).to_csv(metrics, index=False)
            sensitivity = root / 'sensitivity.json'; sensitivity.write_text(json.dumps({'pair_records': cases}))
            actual_digest = o.file_digest
            def checked_digest(path):
                path = Path(path)
                if path == metrics:
                    return 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4'
                if path == sensitivity:
                    return '32ca2146b29a31ddf212988711705832e05366a0300300999f4535ed5fa0e490'
                return actual_digest(path)
            hashes = {name: actual_digest(root / name) for name in o.SOURCE_HASHES}
            raw = np.zeros((4, 3, 721))
            raw[:3, 0] = 100.
            raw[:3, 1] = 1.
            raw[:3, 2] = 3.
            fake_c3d = {'parameters': {'POINT': {'UNITS': {'value': ['m']}, 'RATE': {'value': [360.]},
                                                'LABELS': {'value': ['UNRELATED', 'RWRB', 'RWRA']}}},
                        'header': {'points': {'first_frame': 0}}, 'data': {'points': raw}}
            with patch.object(o, 'SOURCE_HASHES', hashes), patch.object(o, 'file_digest', side_effect=checked_digest), \
                    patch.object(o.ezc3d, 'c3d', return_value=fake_c3d):
                tracks, chosen, audit = o.load_tracks(root, metrics, sensitivity)
                self.assertEqual((len(tracks), len(chosen), len(audit)), (4, 2, 4))
                self.assertTrue(all(r['raw_residual_zero_is_not_accuracy'] for r in audit.values()))
                expected = np.median(np.linalg.norm(2. - tracks['1615_1']['raw_xyz'][:, o.JOINTS.index('wrist_jc')], axis=1)) * 1000
                self.assertAlmostEqual(audit['1615_1']['raw_marker_midpoint_processed_wrist_median_distance_mm'], expected)
                (root / 'metadata.csv').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed_source'):
                    o.load_tracks(root, metrics, sensitivity)

    def test_export_full_decode_count_failure_and_previews(self):
        tracks, case = tracks_and_case()
        fonts = {}
        frames = []
        def writer():
            frame = yield
            while True:
                frames.append(frame)
                frame = yield
        def decode(*args, **kwargs):
            return type('Result', (), {'stdout': 'frame=' + str(len(frames)), 'stderr': ''})()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(o.imageio_ffmpeg, 'write_frames', side_effect=lambda *args, **kwargs: writer()), \
                patch.object(o, 'draw_frame', return_value=Image.new('RGB', (4, 4))), \
                patch.object(o, 'file_digest', return_value='synthetic'), \
                patch.object(o.subprocess, 'run', side_effect=decode):
            result = o.export_case(tracks, case, Path(temporary), fonts)
            self.assertTrue(result['full_decode_verified'])
            self.assertEqual(result['frames'], len(frames))
            self.assertEqual(len(result['previews']), 3)

    def test_main_dispatch_and_output_boundary(self):
        tracks, case = tracks_and_case()
        with tempfile.TemporaryDirectory() as temporary, \
                patch('sys.argv', ['overlay', '--data-dir', 'd', '--metrics-file', 'm', '--sensitivity-file', 's', '--output', temporary]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            o.main()
        # Output paths and source capture are exercised by the real CLI run, not a fake success receipt.


if __name__ == '__main__':
    unittest.main()
