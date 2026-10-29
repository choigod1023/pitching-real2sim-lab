"""Regression checks for every comparison helper, including real execution wiring."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import compare as c


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pairs = pd.DataFrame([
            dict(user='1', pitch_i='a', pitch_j='b', faster_pitch='a', slower_pitch='b',
                 recorded_speed_gap_kmh=2., independent_event_stable=True),
            dict(user='1', pitch_i='a', pitch_j='c', faster_pitch='a', slower_pitch='c',
                 recorded_speed_gap_kmh=.5, independent_event_stable=True),
            dict(user='2', pitch_i='e', pitch_j='d', faster_pitch='d', slower_pitch='e',
                 recorded_speed_gap_kmh=2., independent_event_stable=True)])
        rows = []
        for pitch, user, speed, value in [('a', '1', 100., 5.), ('b', '1', 98., 1.),
                                           ('c', '1', 99.5, 3.), ('d', '2', 100., 1.), ('e', '2', 98., 3.)]:
            for cutoff in c.CUTOFFS:
                rows.append(dict(user=user, session_pitch=pitch, cutoff_hz=cutoff,
                                 pitch_speed_kmh=speed, **{metric: value for metric in c.METRICS}))
        self.metrics = pd.DataFrame(rows)
        self.summary_path, self.metrics_path = self.root / 'source.json', self.root / 'metrics.csv'
        self.summary_path.write_text(json.dumps(dict(pair_records=self.pairs.to_dict('records'))), encoding='utf-8')
        self.metrics.to_csv(self.metrics_path, index=False)

    def test_digest(self):
        self.assertEqual(c.digest(self.metrics_path), hashlib.sha256(self.metrics_path.read_bytes()).hexdigest())
        with self.assertRaises(FileNotFoundError):
            c.digest(self.root / 'absent')

    def test_load_frozen_hash_and_cohort(self):
        with self.assertRaisesRegex(ValueError, 'changed_frozen_source'):
            c.load_inputs(self.summary_path, self.metrics_path)
        with patch.object(c, 'SUMMARY_SHA256', c.digest(self.summary_path)), patch.object(c, 'METRICS_SHA256', c.digest(self.metrics_path)):
            with self.assertRaisesRegex(ValueError, 'changed_frozen_cohort'):
                c.load_inputs(self.summary_path, self.metrics_path)
        pairs = []
        for i in range(35):
            pairs.append(dict(self.pairs.to_dict('records')[0], user=str(i % 23), pitch_i='p' + str(i),
                              recorded_speed_gap_kmh=2 if i < 14 else .5))
        self.summary_path.write_text(json.dumps(dict(pair_records=pairs)), encoding='utf-8')
        with patch.object(c, 'SUMMARY_SHA256', c.digest(self.summary_path)), patch.object(c, 'METRICS_SHA256', c.digest(self.metrics_path)):
            loaded, metrics = c.load_inputs(self.summary_path, self.metrics_path)
            self.assertEqual((len(loaded), loaded.user.nunique(), len(metrics)), (35, 23, 15))
        pairs[-1] = dict(pairs[0], recorded_speed_gap_kmh=.5)
        self.summary_path.write_text(json.dumps(dict(pair_records=pairs)), encoding='utf-8')
        with patch.object(c, 'SUMMARY_SHA256', c.digest(self.summary_path)), patch.object(c, 'METRICS_SHA256', c.digest(self.metrics_path)):
            with self.assertRaisesRegex(ValueError, 'duplicate_frozen_pair'):
                c.load_inputs(self.summary_path, self.metrics_path)

    def test_pair_subtraction_uses_recorded_speed_and_all_filters(self):
        rows = c.pair_differences(self.pairs, self.metrics)
        self.assertEqual(len(rows), 9)
        self.assertEqual(set(rows.cutoff_hz), {10, 20, 30})
        self.assertTrue(rows.loc[rows.faster_pitch == 'a', 'sequence_lag_ms'].isin([4., 2.]).all())
        self.assertTrue(rows.loc[rows.faster_pitch == 'd', 'sequence_lag_ms'].eq(-2).all())
        self.assertEqual(set(rows.loc[rows.faster_pitch == 'd', 'pitch_i']), {'e'})

    def test_pair_input_guards(self):
        for field, value, message in [('faster_pitch', 'b', 'invalid_pair_identity'),
                                      ('recorded_speed_gap_kmh', 0., 'invalid_speed_order'),
                                      ('recorded_speed_gap_kmh', 3., 'recorded_speed_mismatch'),
                                      ('user', '99', 'cross_player_pair')]:
            with self.subTest(field=field):
                pairs = self.pairs.copy()
                pairs.loc[0, field] = value
                with self.assertRaisesRegex(ValueError, message):
                    c.pair_differences(pairs, self.metrics)
        for change, message in [('nan', 'nonfinite_selected_metric'), ('speed_nan', 'nonfinite_selected_metric'),
                                ('duplicate', 'invalid_metric_schema_or_duplicate'), ('missing', 'missing_pitch_filter'),
                                ('schema', 'invalid_metric_schema_or_duplicate')]:
            with self.subTest(change=change):
                table = self.metrics.copy()
                if change == 'nan': table.loc[0, c.PRIMARY[0]] = np.nan
                if change == 'speed_nan': table.loc[0, 'pitch_speed_kmh'] = np.nan
                if change == 'duplicate': table = pd.concat([table, table.iloc[:1]])
                if change == 'missing': table = table.iloc[1:]
                if change == 'schema': table = table.drop(columns=c.PRIMARY[0])
                with self.assertRaisesRegex(ValueError, message):
                    c.pair_differences(self.pairs, table)
        with self.assertRaisesRegex(ValueError, 'empty_pair_cohort'):
            c.pair_differences(self.pairs.iloc[:0], self.metrics)

    def test_player_equal_mean_not_pair_equal(self):
        players = c.player_differences(c.pair_differences(self.pairs, self.metrics))
        nominal = players[players.cutoff_hz == 20]
        self.assertEqual(nominal.pairs.tolist(), [2, 1])
        self.assertEqual(nominal.sequence_lag_ms.tolist(), [3., -2.])
        self.assertEqual(c.summarize_filter(nominal.sequence_lag_ms)['mean'], .5)
        with self.assertRaisesRegex(ValueError, 'empty_or_incomplete_filters'):
            c.player_differences(pd.DataFrame())
        with self.assertRaisesRegex(ValueError, 'invalid_player_filter_support'):
            c.player_differences(c.pair_differences(self.pairs, self.metrics).iloc[1:])

    def test_bootstrap_statistics_and_reproducibility(self):
        value = c.summarize_filter([-2., 0., 5.])
        self.assertEqual([value[k] for k in ['mean', 'median', 'minimum', 'maximum']], [1., 0., -2., 5.])
        self.assertEqual([value[k] for k in ['positive_players', 'negative_players', 'zero_players']], [1, 1, 1])
        self.assertEqual(value, c.summarize_filter([-2., 0., 5.]))
        indices = np.random.default_rng(c.SEED).integers(0, 3, (2000, 3))
        means = sorted(sum([-2., 0., 5.][index] for index in row) / 3 for row in indices)
        self.assertEqual(value['bootstrap_mean_percentile_95'], np.quantile(means, [.025, .975]).tolist())
        self.assertEqual(c.summarize_filter([3.])['bootstrap_mean_percentile_95'], [3., 3.])
        for values in [[], [np.nan], [[1.]]]:
            with self.assertRaisesRegex(ValueError, 'invalid_player_values'):
                c.summarize_filter(values)
        with self.assertRaises(ValueError): c.summarize_filter([1.], bootstraps=0)

    def test_flag_needs_all_filters_ci_and_75_percent_with_zeros_in_denominator(self):
        players = pd.DataFrame([dict(user=str(user), cutoff_hz=cutoff, pairs=1,
                                     sequence_lag_ms=value, pelvis_peak_dps=value)
                                for user, value in enumerate([1., 1., 1., 0.]) for cutoff in c.CUTOFFS])
        result = c.summarize_metric(players, 'sequence_lag_ms')
        self.assertTrue(result['exploratory_followup_flag'])
        self.assertEqual(result['fraction_players_in_shared_mean_direction_all_filters'], .75)
        self.assertFalse(c.summarize_metric(players, 'pelvis_peak_dps')['exploratory_followup_flag'])
        negative = players.copy()
        negative['sequence_lag_ms'] *= -1
        self.assertTrue(c.summarize_metric(negative, 'sequence_lag_ms')['exploratory_followup_flag'])
        players.loc[(players.user == '0') & (players.cutoff_hz == 10), 'sequence_lag_ms'] = -1.
        result = c.summarize_metric(players, 'sequence_lag_ms')
        self.assertFalse(result['exploratory_followup_flag'])
        self.assertEqual(result['player_positive_negative_reversal_count'], 1)
        self.assertEqual(result['player_any_sign_change_including_zero_count'], 1)
        players.loc[(players.user == '1') & (players.cutoff_hz == 10), 'sequence_lag_ms'] = 0.
        self.assertEqual(c.summarize_metric(players, 'sequence_lag_ms')['player_any_sign_change_including_zero_count'], 2)
        with self.assertRaisesRegex(ValueError, 'unknown_metric'): c.summarize_metric(players, 'unknown')
        with self.assertRaisesRegex(ValueError, 'incomplete_metric_support'):
            c.summarize_metric(players.iloc[1:], 'sequence_lag_ms')

    def test_compare_preserves_subset_and_primary_secondary_metrics(self):
        result = c.compare(c.pair_differences(self.pairs, self.metrics))
        self.assertEqual([(v['pairs'], v['players']) for v in result.values()], [(3, 2), (2, 2)])
        self.assertEqual(result['independent_event_stable']['metrics']['sequence_lag_ms']['filters']['20']['mean'], .5)
        self.assertEqual(result['original_gap_at_least_1_kmh']['metrics']['sequence_lag_ms']['filters']['20']['mean'], 1.)
        self.assertEqual(set(result['independent_event_stable']['metrics']), set(c.METRICS))

    def test_flag_is_per_filter_not_intersection_and_rejects_conflicting_means(self):
        players = pd.DataFrame([dict(user=str(user), cutoff_hz=cutoff, pairs=1,
                                     sequence_lag_ms=-.001 if user // 3 == index else 1.)
                                for user in range(12) for index, cutoff in enumerate(c.CUTOFFS)])
        result = c.summarize_metric(players, 'sequence_lag_ms')
        self.assertTrue(result['exploratory_followup_flag'])
        self.assertEqual(result['fraction_players_in_shared_mean_direction_all_filters'], .25)
        self.assertEqual(set(result['fraction_players_in_shared_mean_direction_by_filter'].values()), {.75})
        players.loc[players.cutoff_hz == 10, 'sequence_lag_ms'] = -1.
        result = c.summarize_metric(players, 'sequence_lag_ms')
        self.assertFalse(result['exploratory_followup_flag'])
        self.assertEqual(result['shared_mean_direction'], 0)
        players['sequence_lag_ms'] = 0.
        result = c.summarize_metric(players, 'sequence_lag_ms')
        self.assertFalse(result['exploratory_followup_flag'])
        self.assertFalse(result['all_intervals_same_direction'])
        self.assertEqual(result['filters']['10']['zero_players'], 12)

    def test_independent_check_and_corruption(self):
        result = c.compare(c.pair_differences(self.pairs, self.metrics))
        check = c.independent_check(self.summary_path, self.metrics_path, result)
        self.assertTrue(check['passed'])
        self.assertEqual(check['maximum_absolute_difference'], 0.)
        result['independent_event_stable']['metrics']['sequence_lag_ms']['filters']['20']['mean'] += 1
        with self.assertRaisesRegex(AssertionError, 'independent_arithmetic_mismatch'):
            c.independent_check(self.summary_path, self.metrics_path, result)

    def test_run_writes_artifacts_and_local_csv_without_gate_change(self):
        with patch.object(c, 'load_inputs', return_value=(self.pairs, self.metrics)):
            result, check = c.run(self.summary_path, self.metrics_path, self.root / 'out')
        self.assertFalse(result['speed_prediction_gate_changed'])
        self.assertTrue(check['passed'])
        files = {p.name for p in (self.root / 'out').iterdir()}
        self.assertEqual(files, {'comparison.json', 'comparison_manifest.json', 'comparison_verification.json',
                                 'comparison_pairs.csv', 'comparison_players.csv'})
        manifest = json.loads((self.root / 'out/comparison_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['code_sha256'], c.digest(c.__file__))
        self.assertEqual(len(pd.read_csv(self.root / 'out/comparison_pairs.csv')), 9)
        self.assertEqual(len(pd.read_csv(self.root / 'out/comparison_players.csv')), 12)


if __name__ == '__main__':
    unittest.main()
