import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from .metrics import FEATURES, LOADS
from .modeling import (demean, player_weights, fit_ridge, predict_ridge, choose_alpha,
                       nested_predictions, bootstrap_mean, prediction_summary, associations,
                       filter_sensitivity, scientific_gate, ASSOCIATION_METRICS)


def fixture_features():
    rows = []
    for player in range(6):
        for trial, delta in enumerate([-1., 0., 1.]):
            row = {c: player*5 + delta for c in set(ASSOCIATION_METRICS + LOADS)}
            row.update(user=str(player), session_pitch=f'{player}_{trial}',
                       pitch_speed_kmh=100+player*5-delta)
            rows.append(row)
    return pd.DataFrame(rows)


class ModelingTests(unittest.TestCase):
    def test_demean_and_equal_player_weights(self):
        groups = np.array(['a', 'a', 'b'])
        x = np.array([[1., 5.], [3., 7.], [10., 9.]])
        np.testing.assert_allclose(demean(x, groups), [[-1., -1.], [1., 1.], [0., 0.]])
        np.testing.assert_allclose(player_weights(groups), [.5, .5, 1.])

    def test_ridge_constant_rank_deficient_and_train_only_scale(self):
        x = np.array([[1., 1., 0.], [2., 1., 0.], [3., 1., 0.]])
        model = fit_ridge(x, np.array([2., 4., 6.]), np.array(['a','b','c']), .01)
        self.assertTrue(np.isfinite(predict_ridge(model, x)).all())
        self.assertEqual(model[2].tolist(), [False, True, True])
        before = model[1].copy()
        predict_ridge(model, np.array([[1e8, 100., 5.]]))
        np.testing.assert_array_equal(model[1], before)
        self.assertLess(np.max(np.abs(predict_ridge(model, x)-[2,4,6])), .02)

    def test_alpha_tie_prefers_larger_and_small_group_error(self):
        x = np.ones((6, 2))
        self.assertEqual(choose_alpha(x, np.zeros(6), np.array(list('abcdef'))), 100.)
        with self.assertRaisesRegex(ValueError, 'too_few'):
            choose_alpha(x, np.zeros(6), np.array(['a']*6))

    def test_outer_held_outcomes_cannot_change_own_prediction(self):
        frame = fixture_features()
        first, audits = nested_predictions(frame)
        changed = frame.copy()
        changed.loc[changed.user == '2', 'pitch_speed_kmh'] += [20., -40., 100.]
        second, _ = nested_predictions(changed)
        np.testing.assert_allclose(first[first.user == '2'].prediction, second[second.user == '2'].prediction)
        for audit in audits:
            self.assertFalse(audit['overlap'])
            self.assertNotIn(audit['held_player'], audit['training_players'])
        self.assertTrue(np.allclose(first.groupby('user').prediction.mean(), 0))
        summary, errors = prediction_summary(first)
        self.assertLess(summary['player_balanced_model_mae_kmh'], summary['player_balanced_baseline_mae_kmh'])
        self.assertEqual(len(errors), 6)

    def test_between_predictions_use_train_mean_baseline(self):
        frame = fixture_features()
        predictions, _ = nested_predictions(frame, within=False)
        self.assertEqual(len(predictions), 6)
        expected = frame[frame.user != '0'].groupby('user').pitch_speed_kmh.mean().mean()
        self.assertAlmostEqual(predictions.loc[predictions.user == '0', 'baseline'].iloc[0], expected)

    def test_bootstrap_reproducible_cluster_and_invalid(self):
        self.assertEqual(bootstrap_mean([2.,2.,2.]), [2.,2.])
        self.assertEqual(bootstrap_mean([1.,2.,5.]), bootstrap_mean([1.,2.,5.]))
        with self.assertRaises(ValueError):
            bootstrap_mean([np.nan])

    def test_between_within_opposite_sign_and_constant_not_zero(self):
        frame = fixture_features()
        frame['arm_slot_poi_deg'] = 45.
        result = associations(frame, repetitions=100)
        selected = result[(result.metric == FEATURES[0]) & (result.target == 'pitch_speed_kmh')].set_index('level')
        self.assertAlmostEqual(selected.loc['between', 'r'], 1.)
        self.assertAlmostEqual(selected.loc['within', 'r'], -1.)
        self.assertTrue(result[result.metric == 'arm_slot_poi_deg'].r.isna().all())

    def test_filter_sensitivity_common_support_and_sign(self):
        frame = fixture_features()
        blocks = []
        for cutoff, delta in [(10, -1), (20, 0), (30, 1)]:
            block = frame.copy()
            block['cutoff_hz'] = cutoff
            block['sequence_lag_ms'] = delta
            for column in ['elbow_varus_peak_nm', 'shoulder_compression_peak_n', 'shoulder_distraction_peak_n']:
                block[column] = 5.
            blocks.append(block)
        all_rows = pd.concat(blocks)
        result, flips = filter_sensitivity(all_rows)
        self.assertEqual(flips, len(frame))
        self.assertEqual(result.set_index('metric').loc['sequence_lag_ms', 'max_range'], 2.)
        with self.assertRaisesRegex(ValueError, 'noncommon'):
            filter_sensitivity(all_rows.iloc[1:])

    def test_gate_requires_all_preregistered_conditions(self):
        within = {'players': 60, 'pitches': 180, 'paired_mae_difference_kmh': -.1,
                  'paired_difference_cluster_ci95': [-.2, -.01]}
        summaries = {str(c): {'within': dict(within)} for c in (10,20,30)}
        self.assertEqual(scientific_gate(summaries, True)['verdict'], 'PASS')
        summaries['30']['within']['paired_mae_difference_kmh'] = .01
        self.assertEqual(scientific_gate(summaries, True)['verdict'], 'FAIL')
        self.assertFalse(scientific_gate(summaries, False)['criteria']['split_and_preprocessing_integrity'])


if __name__ == '__main__':
    unittest.main()
