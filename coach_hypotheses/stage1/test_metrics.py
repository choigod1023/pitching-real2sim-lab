import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (COLUMNS, EVENTS, G, FEATURES, checked_event, prepare_pitch,
                      filtered_signals, at_event, window_peak, pitch_metrics, read_cohort)


def fixture_tables():
    t = np.round(np.arange(721) / 360, 4)
    events = dict(zip(EVENTS, (.4, 1., 1.5)))
    values = {'pelvis_velo_z': 100*np.exp(-((t-1.1)/.08)**2),
              'torso_velo_z': 150*np.exp(-((t-1.2)/.08)**2),
              'lead_knee_velo_x': np.full(len(t), -20.),
              'lead_knee_angle_x': 60-20*t, 'torso_pelvis_angle_z': 30+2*t,
              'elbow_moment_y': np.full(len(t), 60.),
              'shoulder_upper_arm_force_x': np.full(len(t), 30.),
              'shoulder_upper_arm_force_y': np.full(len(t), 40.),
              'shoulder_upper_arm_force_z': np.full(len(t), -120.),
              'shoulder_thorax_force_z': np.full(len(t), 50.),
              'wrist_jc_z': np.full(len(t), 1.8)}
    return {name: pd.DataFrame({'time': t, **{c: np.full(len(t), v) for c, v in events.items()},
                               'fp_100_time': 1.05, **{c: values[c] for c in columns}})
            for name, columns in COLUMNS.items()}


def fixture_metadata():
    return {'session_pitch': '10_1', 'user': 1, 'session': 10,
            'session_mass_kg': 80., 'session_height_m': 2., 'pitch_speed_mph': 80.}


def fixture_poi():
    return {'session_pitch': '10_1', 'pitch_speed_mph': 80., 'stride_length': .8,
            'arm_slot': 45., 'elbow_varus_moment': 60.,
            'rotation_hip_shoulder_separation_fp': 32., 'max_rotation_hip_shoulder_separation': 34.}


class MetricTests(unittest.TestCase):
    def test_event_constant_required(self):
        frame = fixture_tables()['joint_velos']
        self.assertEqual(checked_event(frame, 'BR_time'), 1.5)
        frame.loc[0, 'BR_time'] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonconstant_event'):
            checked_event(frame, 'BR_time')

    def test_known_timing_units_knee_and_force_signs(self):
        result = pitch_metrics(prepare_pitch(fixture_tables()), fixture_metadata(), fixture_poi(), 20)
        self.assertAlmostEqual(result['sequence_lag_ms'], 100., delta=.2)
        self.assertAlmostEqual(result['sequence_lag_ms'], result['torso_peak_time_fp10_ms']-result['pelvis_peak_time_fp10_ms'])
        self.assertAlmostEqual(result['sequence_lag_ms'], result['torso_peak_time_br_ms']-result['pelvis_peak_time_br_ms'])
        self.assertAlmostEqual(result['lead_knee_extension_deg'], 10., places=5)
        self.assertAlmostEqual(result['lead_knee_peak_extension_dps'], 20., places=5)
        self.assertAlmostEqual(result['elbow_varus_bw_height_pct'], 100*60/(80*G*2))
        self.assertAlmostEqual(result['shoulder_resultant_peak_n'], 130.)
        self.assertAlmostEqual(result['shoulder_distraction_peak_n'], 120.)
        self.assertAlmostEqual(result['pitch_speed_kmh'], 128.74752)
        self.assertIsNone(result['actual_ball_release_height_m'])

    def test_poi_not_refiltered_and_input_immutable(self):
        prepared = prepare_pitch(fixture_tables())
        original = prepared[1].copy()
        rows = [pitch_metrics(prepared, fixture_metadata(), fixture_poi(), f) for f in (10, 20, 30)]
        np.testing.assert_array_equal(prepared[1], original)
        self.assertEqual([r['stride_reported'] for r in rows], [.8]*3)
        self.assertEqual([r['arm_slot_poi_deg'] for r in rows], [45.]*3)
        with self.assertRaises(ValueError):
            filtered_signals(prepared[1], prepared[2], 99)

    def test_angle_unwrap(self):
        prepared = prepare_pitch(fixture_tables())
        matrix = prepared[1].copy()
        i = prepared[2].index('torso_pelvis_angle_z')
        angles = np.linspace(170, 190, len(matrix))
        matrix[:, i] = (angles+180) % 360 - 180
        signal = filtered_signals(matrix, prepared[2], 20)['torso_pelvis_angle_z']
        self.assertLess(np.max(np.abs(np.diff(signal))), 1.)
        self.assertGreater(signal[-1], 185.)

    def test_interpolation_and_no_extrapolation(self):
        self.assertEqual(at_event(np.array([0., 1.]), np.array([2., 4.]), .5), 3.)
        with self.assertRaisesRegex(ValueError, 'extrapolation'):
            at_event(np.array([0., 1.]), np.array([2., 4.]), 2.)

    def test_peak_ties_and_boundary(self):
        self.assertEqual(window_peak(np.arange(4), np.array([1., 3., 3., 1.]), 0, 3), (1., 3., False))
        self.assertEqual(window_peak(np.arange(4), np.array([3., 3., 1., 1.]), 0, 3), (0., 3., True))
        with self.assertRaisesRegex(ValueError, 'empty'):
            window_peak(np.arange(4), np.ones(4), 5, 6)

    def test_outer_nans_trimmed_inner_nan_rejected(self):
        tables = fixture_tables()
        tables['joint_velos'].loc[:5, 'pelvis_velo_z'] = np.nan
        tables['joint_velos'].loc[715:, 'pelvis_velo_z'] = np.nan
        self.assertEqual(len(prepare_pitch(tables)[0]), 709)
        tables['joint_velos'].loc[400, 'pelvis_velo_z'] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite_analysis_window'):
            prepare_pitch(tables)

    def test_time_cross_table_and_padding_checks(self):
        for problem, message in [('gap', 'sampling'), ('cross', 'cross_table_time'),
                                 ('event', 'cross_table_event'), ('order', 'event_order'),
                                 ('padding', 'padding')]:
            with self.subTest(problem=problem):
                tables = fixture_tables()
                if problem == 'gap':
                    tables['joint_velos'].loc[400:, 'time'] += .01
                elif problem == 'cross':
                    tables['landmarks']['time'] += .0001
                elif problem == 'event':
                    tables['landmarks']['BR_time'] = 1.4
                elif problem == 'order':
                    tables['joint_velos']['BR_time'] = .5
                else:
                    tables['joint_velos'].loc[:140, 'pelvis_velo_z'] = np.nan
                with self.assertRaisesRegex(ValueError, message):
                    prepare_pitch(tables)

    def test_bad_anthropometrics_and_poi(self):
        meta, poi = fixture_metadata(), fixture_poi()
        meta['session_mass_kg'] = -1
        with self.assertRaisesRegex(ValueError, 'anthropometrics'):
            pitch_metrics(prepare_pitch(fixture_tables()), meta, poi, 20)
        meta['session_mass_kg'] = 80
        poi['stride_length'] = np.nan
        with self.assertRaisesRegex(ValueError, 'poi_predictor'):
            pitch_metrics(prepare_pitch(fixture_tables()), meta, poi, 20)

    def test_read_cohort_identity_common_filters_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'poi').mkdir()
            (path/'full_sig').mkdir()
            meta, poi, tables = [], [], {k: [] for k in COLUMNS}
            for i in range(4):
                key = f'10_{i}'
                meta.append(dict(fixture_metadata(), session_pitch=key, user=1 if i < 3 else 2))
                poi.append(dict(fixture_poi(), session_pitch=key))
                for name, frame in fixture_tables().items():
                    frame['session_pitch'] = key
                    if i == 2 and name == 'landmarks':
                        frame.loc[400, 'wrist_jc_z'] = np.nan
                    tables[name].append(frame)
            pd.DataFrame(meta).to_csv(path/'metadata.csv', index=False)
            pd.DataFrame(poi).to_csv(path/'poi/poi_metrics.csv', index=False)
            for name, blocks in tables.items():
                pd.concat(blocks).to_csv(path/f'full_sig/{name}.csv', index=False)
            frame, excluded, total = read_cohort(path)
            self.assertEqual((total, len(frame), len(excluded)), (4, 6, 2))
            self.assertEqual(frame.cutoff_hz.value_counts().to_dict(), {10: 2, 20: 2, 30: 2})
            self.assertTrue(np.isfinite(frame[FEATURES]).all().all())
            poi[0]['pitch_speed_mph'] = 81
            pd.DataFrame(poi).to_csv(path/'poi/poi_metrics.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'identity_or_speed_conflict'):
                read_cohort(path)


if __name__ == '__main__':
    unittest.main()
