import contextlib
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import compare as c


class CurveTests(unittest.TestCase):
    def test_constant_triangle_and_zero_crossing_centers(self):
        self.assertAlmostEqual(c.temporal_centroid([0, 1, 2], [1, 1, 1], .2, 1.8), 1.)
        self.assertAlmostEqual(c.temporal_centroid([0, 1, 2], [0, 1, 0], 0, 2), 1.)
        self.assertAlmostEqual(c.temporal_centroid([0, 2], [-1, 1], 0, 2), 5 / 3)
        self.assertAlmostEqual(c.temporal_centroid([0, 1], [0, 1], 0, 1), 2 / 3)
        self.assertAlmostEqual(c.temporal_centroid([0, 2], [-1, 1], .5, 1.5), 4 / 3)

    def test_centroid_shift_amplitude_and_sampling_invariance(self):
        reference = c.temporal_centroid([0, 2], [-1, 1], .1, 1.8)
        self.assertAlmostEqual(c.temporal_centroid([100, 102], [-1, 1], 100.1, 101.8), reference + 100)
        self.assertAlmostEqual(c.temporal_centroid([0, 2], [-10, 10], .1, 1.8), reference)
        self.assertAlmostEqual(c.temporal_centroid([0, .5, 1, 1.5, 2], [-1, -.5, 0, .5, 1], .1, 1.8), reference)

    def test_no_invalid_or_zero_area_imputation(self):
        for time, values, begin, end in [([0, 1], [0, 0], 0, 1), ([0, 1], [-1, -2], 0, 1),
                                         ([0, 0], [1, 2], 0, 1), ([0, 1], [1, np.nan], 0, 1),
                                         ([0, 1], [1, 2], -.1, 1), ([0, 1], [1, 2], .5, .5)]:
            with self.assertRaises(ValueError): c.temporal_centroid(time, values, begin, end)

    def test_feature_cases_reference_and_mismatch(self):
        t = np.arange(361) / 360.
        signals = {'pelvis_velo_z': np.full(361, 10.), 'torso_velo_z': 10 + t}
        module = SimpleNamespace(filtered_signals=lambda *args: signals,
                                 window_peak=lambda time, y, a, b: (float(time[np.flatnonzero((time >= a) & (time <= b))[np.argmax(y[(time >= a) & (time <= b)])]]), 0, False))
        prepared = (t, None, None, dict(pkh_time=.2, fp_10_time=.4, BR_time=.9), None)
        expected = pd.DataFrame({'sequence_lag_ms': [700., 700., 700.]}, index=c.CUTOFFS)
        nominal, rows, widths = c.pitch_features(module, prepared, expected)
        self.assertEqual((len(nominal), len(rows)), (3, 27))
        self.assertGreater(nominal[0]['centroid_lag_ms'], 0)
        self.assertGreater(widths['peak_lag_ms_range_ms'], widths['centroid_lag_ms_range_ms'])
        with self.assertRaisesRegex(ValueError, 'not_reproduced'):
            c.pitch_features(module, prepared, expected + 1)

    def test_pair_ids_direction_and_duplicate_guard(self):
        frame = pd.DataFrame([dict(user='1', session_pitch='b', pitch_speed_kmh=102., peak_lag_ms=5., centroid_lag_ms=2.),
                              dict(user='1', session_pitch='a', pitch_speed_kmh=100., peak_lag_ms=1., centroid_lag_ms=1.)])
        pairs = c.make_pairs(frame)
        self.assertEqual((pairs.iloc[0].pitch_i, pairs.iloc[0].pitch_j), ('a', 'b'))
        self.assertEqual(pairs.iloc[0].truth_delta_kmh, 2.)
        self.assertEqual(pairs.iloc[0].peak_lag_ms, 4.)
        pd.testing.assert_frame_equal(pairs, c.make_pairs(frame.iloc[::-1]))
        with self.assertRaises(ValueError): c.make_pairs(pd.concat([frame, frame.iloc[:1]]))
        with self.assertRaises(ValueError): c.make_pairs(frame.iloc[:1])

    def test_heldout_outcomes_do_not_affect_own_prediction(self):
        pairs = pd.DataFrame([dict(user=str(user), pitch_i=str(i), pitch_j=str(i+1), truth_delta_kmh=float(user+i),
                                   peak_lag_ms=float(i+1), centroid_lag_ms=float(i+2)) for user in range(4) for i in range(3)])
        original, _ = c.leave_one_player_out(pairs, 'centroid_lag_ms')
        changed = pairs.copy(); changed.loc[changed.user.eq('0'), 'truth_delta_kmh'] += 10000
        altered, _ = c.leave_one_player_out(changed, 'centroid_lag_ms')
        np.testing.assert_allclose(original[original.user.eq('0')].prediction_delta_kmh, altered[altered.user.eq('0')].prediction_delta_kmh)
        reversed_pairs = pairs.copy()
        reversed_pairs[['truth_delta_kmh', *c.FEATURES]] *= -1
        reversed_predictions, _ = c.leave_one_player_out(reversed_pairs, 'centroid_lag_ms')
        np.testing.assert_allclose(original.prediction_delta_kmh, -reversed_predictions.prediction_delta_kmh)

    def test_equal_player_weights_and_zero_feature(self):
        pairs = pd.DataFrame([dict(user=user, pitch_i=str(i), pitch_j=str(i+1), truth_delta_kmh=y,
                                   peak_lag_ms=1., centroid_lag_ms=0.)
                              for user, count, y in [('1', 1, 1.), ('2', 3, 10.), ('3', 1, 0.)] for i in range(count)])
        result, splits = c.leave_one_player_out(pairs, 'peak_lag_ms')
        self.assertAlmostEqual(result[result.user.eq('3')].prediction_delta_kmh.iloc[0], 11 / 12)
        self.assertTrue(all(s['heldout_absent_from_train'] for s in splits))
        result, _ = c.leave_one_player_out(pairs, 'centroid_lag_ms')
        self.assertTrue(result.prediction_delta_kmh.eq(0).all())

    def test_zero_baseline_score_and_unchanged_gate_conditions(self):
        predictions = pd.DataFrame({'user': ['1', '1', '2'], 'truth_delta_kmh': [1., 3., 1.], 'prediction_delta_kmh': [0., 0., 0.]})
        score = c.score(predictions)
        self.assertEqual(score['model_mae_kmh'], 1.5)
        self.assertEqual(score['zero_delta_mae_kmh'], 1.5)
        self.assertEqual(score['player_bootstrap_difference_ci95'], [0., 0.])
        values = {str(x): score for x in c.CUTOFFS}
        self.assertFalse(c.development_checks(values, 98, 399, True)['development_prediction_criteria_pass'])
        good = {str(x): dict(model_mae_kmh=.5, zero_delta_mae_kmh=1., player_bootstrap_difference_ci95=[-.6, -.4]) for x in c.CUTOFFS}
        self.assertTrue(c.development_checks(good, 98, 399, True)['development_prediction_criteria_pass'])
        self.assertFalse(c.development_checks(good, 98, 399, False)['development_prediction_criteria_pass'])
        self.assertFalse(c.development_checks(good, 1, 399, True)['development_prediction_criteria_pass'])

    def test_changed_source_and_cli_boundary(self):
        with patch.object(c, 'file_digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'changed_frozen'): c.run('data', 'metrics', 'out')
        with patch('sys.argv', ['compare', '--data-dir', 'd', '--metrics-dir', 'm']), patch.object(c, 'run') as mocked:
            c.main()
            self.assertEqual(mocked.call_args.args[:2], (c.Path('d'), c.Path('m')))
        with tempfile.TemporaryDirectory() as temporary, patch('sys.argv', ['compare', '--data-dir', 'd', '--metrics-dir', 'm', '--output', temporary]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            c.main()


if __name__ == '__main__': unittest.main()
