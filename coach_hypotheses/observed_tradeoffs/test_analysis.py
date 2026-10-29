import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import analysis as a


def synthetic_frames(repeats=(3, 2)):
    metrics, metadata, poi = [], [], []
    for player, count in enumerate(repeats):
        user, session = str(player + 101), str(player + 1001)
        for pitch in range(count):
            key = session + '_' + str(pitch + 1)
            mph = 60. + pitch if player == 0 else 62.
            source = dict(user=user, session=session, session_pitch=key, pitch_speed_mph=mph)
            metadata.append(source)
            poi.append(dict(session=session, session_pitch=key, pitch_speed_mph=mph, pitch_type='FF'))
            load = (100., 90., 110., 95., 105.)[pitch]
            for cutoff in a.CUTOFFS:
                row = dict(user=user, session=session, session_pitch=key,
                           cutoff_hz=cutoff, pitch_speed_kmh=mph * 1.609344)
                row.update(dict(zip(a.LOADS, [load, load * 4, load * 2])))
                row.update({name: float(pitch + 1) for name in a.MOTION})
                metrics.append(row)
    return pd.DataFrame(metrics), pd.DataFrame(metadata), pd.DataFrame(poi)


def write_fixture(root, frames):
    frame, metadata, poi = frames
    metrics, data = root / 'metrics', root / 'data'
    (metrics / 'results').mkdir(parents=True)
    (data / 'poi').mkdir(parents=True)
    frame.to_csv(metrics / 'results/pitch_metrics.csv', index=False)
    metadata.to_csv(data / 'metadata.csv', index=False)
    poi.to_csv(data / 'poi/poi_metrics.csv', index=False)
    manifest = {'sources': {name: {'sha256': a.file_digest(data / name)}
                            for name in ('metadata.csv', 'poi/poi_metrics.csv')}}
    (metrics / 'results/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (metrics / 'results/exclusions.json').write_text('[]', encoding='utf-8')
    (metrics / 'PROTOCOL.md').write_text('Synthetic test protocol', encoding='utf-8')
    (metrics / 'AMENDMENT_01_KNEE_SIGN.md').write_text('Synthetic test amendment', encoding='utf-8')
    (root / 'README.md').write_text('Synthetic source force conventions', encoding='utf-8')
    return metrics, data


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.metadata, self.poi = synthetic_frames()

    def test_validation_and_pair_count(self):
        frame = a.validate_frame(self.frame, self.metadata, self.poi)
        comparisons = a.make_comparisons(frame)
        self.assertEqual(len(comparisons), 12)
        self.assertEqual(comparisons[comparisons.cutoff_hz == 20].tied_speed.sum(), 1)
        good = comparisons[(comparisons.pitch_i == '1001_1') & (comparisons.pitch_j == '1001_2')]
        self.assertTrue(good.status.eq('observed_nonincrease').all())
        self.assertTrue(good.faster_pitch.eq('1001_2').all())
        self.assertAlmostEqual(good.iloc[0].elbow_varus_peak_nm_delta, -10.)
        self.assertAlmostEqual(good.iloc[0].recorded_speed_gap_kmh, 1.609344)

    def test_classification_and_epsilon_scope(self):
        self.assertEqual(a.classify_loads([-1, -2, 0]), 'observed_nonincrease')
        self.assertEqual(a.classify_loads([a.EPSILON / 2, 0, 0]), 'observed_nonincrease')
        self.assertEqual(a.classify_loads([1, 2, 3]), 'all_loads_increase')
        self.assertEqual(a.classify_loads([-1, 2, 0]), 'mixed_load_tradeoff')
        self.assertEqual(a.classify_loads([-1, -2, -3], True), 'tied_speed')
        for invalid in ([], [1, 2], [1, np.nan, 3]):
            with self.assertRaises(ValueError):
                a.classify_loads(invalid)

    def test_input_order_invariance(self):
        one = a.make_comparisons(self.frame)
        two = a.make_comparisons(self.frame.sample(frac=1., random_state=3))
        pd.testing.assert_frame_equal(one, two)

    def test_missing_nonfinite_and_inconsistent_filters_rejected(self):
        variants = []
        variants.append(self.frame.drop(columns='session'))
        variants.append(self.frame.iloc[:-1])
        variants.append(pd.concat([self.frame, self.frame.iloc[:1]]))
        for column, value in [('user', ''), ('user', 'not_numeric'),
                              ('elbow_varus_peak_nm', np.nan), ('shoulder_resultant_peak_n', -1.),
                              ('cutoff_hz', 11), ('pitch_speed_kmh', 99.), ('session', 'other')]:
            changed = self.frame.copy()
            changed.loc[0, column] = value
            variants.append(changed)
        for frame in variants:
            with self.subTest(columns=list(frame.columns), rows=len(frame)):
                with self.assertRaises(ValueError):
                    a.validate_frame(frame, self.metadata, self.poi)
        with self.assertRaises(ValueError):
            a.make_comparisons(self.frame.iloc[:0])

    def test_source_speed_identity_and_type_rejected(self):
        for table, column, value in [('metadata', 'pitch_speed_mph', 0.),
                                      ('metadata', 'user', '999'), ('metadata', 'session', 'x'),
                                      ('poi', 'pitch_type', 'CU'), ('poi', 'session', 'x')]:
            meta, poi = self.metadata.copy(), self.poi.copy()
            (meta if table == 'metadata' else poi).loc[0, column] = value
            with self.assertRaises(ValueError):
                a.validate_frame(self.frame, meta, poi)
        for meta in (self.metadata.iloc[:-1], pd.concat([self.metadata, self.metadata.iloc[:1]])):
            with self.assertRaises(ValueError):
                a.validate_frame(self.frame, meta, self.poi)

    def test_ambiguous_session_and_insufficient_repeats_rejected(self):
        changed = self.frame.copy()
        changed.loc[changed.session_pitch.eq('1001_1'), 'session'] = 'other'
        with self.assertRaisesRegex(ValueError, 'ambiguous_player_session'):
            a.validate_frame(changed, self.metadata, self.poi)
        changed = self.frame[~self.frame.session_pitch.eq('1002_1')]
        with self.assertRaisesRegex(ValueError, 'insufficient_player_repeats'):
            a.validate_frame(changed, self.metadata, self.poi)

    def test_filter_change_prevents_stable_selection(self):
        changed = self.frame.copy()
        mask = changed.session_pitch.eq('1001_2') & changed.cutoff_hz.eq(30)
        changed.loc[mask, a.LOADS[0]] = 101.
        summary, pairs = a.summarize_comparisons(a.make_comparisons(changed))
        self.assertEqual(summary['filter_stable_nonincrease_pairs'], 0)
        self.assertEqual(summary['observational_finding'], 'ABSENT')
        self.assertEqual(summary['status_disagreement_pairs'], 1)
        self.assertEqual(pairs.qualifying_filters.max(), 2)
        with self.assertRaises(ValueError):
            a.summarize_comparisons(a.make_comparisons(changed).iloc[:-1])

    def test_signed_tension_is_not_negative_magnitude(self):
        for signal in ([10., 20., 30.], [-10., -20., -30.], [-10., 0., 20.]):
            raw = max(-np.asarray(signal))
            self.assertEqual(max(0., raw), max(np.maximum(0., -np.asarray(signal))))
        changed = self.frame.copy()
        changed.loc[changed.session_pitch.eq('1001_1'), a.LOADS[2]] = -10.
        changed.loc[changed.session_pitch.eq('1001_2'), a.LOADS[2]] = -5.
        original = changed.copy(deep=True)
        valid = a.validate_frame(changed, self.metadata, self.poi)
        records = a.make_comparisons(valid)
        selected = records[records.pitch_i.eq('1001_1') & records.pitch_j.eq('1001_2')]
        self.assertTrue(selected.raw_signed_status.eq('mixed_load_tradeoff').all())
        self.assertTrue(selected.status.eq('observed_nonincrease').all())
        self.assertTrue(selected[a.TENSION + '_delta'].eq(0).all())
        self.assertTrue(selected[a.LOADS[2] + '_delta'].eq(5).all())
        pd.testing.assert_frame_equal(changed, original)
        summary, _ = a.summarize_comparisons(records)
        self.assertGreater(summary['amended_vs_raw_stable_disagreement_pairs'], 0)

    def test_cluster_rates_ties_examples_and_flags(self):
        summary, _ = a.summarize_comparisons(a.make_comparisons(self.frame))
        self.assertEqual(summary['players_without_non_tied_pairs'], ['102'])
        self.assertEqual(summary['unordered_pairs'], 4)
        self.assertEqual(summary['non_tied_pairs'], 3)
        self.assertEqual(summary['filter_stable_nonincrease_pairs'], 1)
        self.assertAlmostEqual(summary['player_balanced_stable_fraction'], 1 / 3)
        self.assertEqual(summary['examples'][0]['pitch_i'], '1001_1')
        for flag in ('measurement_robustness_validated', 'same_capacity_validated',
                     'intervention_effect_validated', 'coaching_recommendation_ready',
                     'speed_prediction_gate_changed', 'model_fitted'):
            self.assertIs(summary[flag], False)
        ties = self.frame.copy()
        ties['pitch_speed_kmh'] = 100.
        summary, _ = a.summarize_comparisons(a.make_comparisons(ties))
        self.assertIsNone(summary['player_balanced_stable_fraction'])
        self.assertIsNone(summary['player_cluster_sampling_ci95'])
        self.assertEqual(summary['examples'], [])

    def test_player_equal_weight_is_not_pair_weight(self):
        f, _, _ = synthetic_frames((2, 3))
        f.loc[f.user.eq('102'), 'pitch_speed_kmh'] = f.loc[f.user.eq('102'), 'session_pitch'].str[-1].astype(float) + 100.
        # First player's only pair qualifies. All three pairs of second player fail.
        for column in a.LOADS:
            f.loc[f.user.eq('102'), column] = f.loc[f.user.eq('102'), 'pitch_speed_kmh']
        summary, _ = a.summarize_comparisons(a.make_comparisons(f))
        self.assertEqual(summary['player_balanced_stable_fraction'], .5)
        self.assertEqual(summary['filter_stable_nonincrease_pairs'], 1)
        self.assertEqual(summary['non_tied_pairs'], 4)

    def test_bootstrap_seed_and_input_checks(self):
        self.assertEqual(a.bootstrap_player_mean([.1, .5, .9]), a.bootstrap_player_mean([.1, .5, .9]))
        self.assertEqual(a.bootstrap_player_mean([.4, .4]), [.4, .4])
        for values in ([], [np.nan], [[.1, .5]]):
            with self.assertRaises(ValueError):
                a.bootstrap_player_mean(values)

    def test_digest_loader_run_and_source_preservation(self):
        repeats = (2,) * 9 + (3,) * 23 + (4,) * 18 + (5,) * 48
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, data = write_fixture(root, synthetic_frames(repeats))
            digest = a.file_digest(metrics / 'results/pitch_metrics.csv')
            with patch.object(a, 'METRICS_SHA256', digest):
                frame, manifest = a.load_inputs(metrics, data)
                self.assertEqual(len(frame), 1197)
                summary = a.run_analysis(metrics, data, root / 'output')
                self.assertEqual(summary['unordered_pairs'], 666)
                self.assertEqual(summary['filter_pair_rows'], 1998)
                self.assertEqual(a.file_digest(metrics / 'results/pitch_metrics.csv'), digest)
                self.assertEqual(json.loads((root / 'output/manifest.json').read_text())['protocol_commit'], a.PROTOCOL_COMMIT)
                self.assertEqual(manifest['new_exclusions'], [])
                with (data / 'metadata.csv').open('a') as handle:
                    handle.write('\n')
                with self.assertRaisesRegex(ValueError, 'changed_upstream_source'):
                    a.load_inputs(metrics, data)
            with self.assertRaisesRegex(ValueError, 'changed_frozen_metrics'):
                a.load_inputs(metrics, data)

    def test_loader_rejects_cohort_and_reintroduced_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics, data = write_fixture(Path(temporary), synthetic_frames())
            with patch.object(a, 'METRICS_SHA256', a.file_digest(metrics / 'results/pitch_metrics.csv')):
                with self.assertRaisesRegex(ValueError, 'unexpected_frozen_cohort'):
                    a.load_inputs(metrics, data)
        with tempfile.TemporaryDirectory() as temporary:
            repeats = (2,) * 9 + (3,) * 23 + (4,) * 18 + (5,) * 48
            metrics, data = write_fixture(Path(temporary), synthetic_frames(repeats))
            (metrics / 'results/exclusions.json').write_text('[{"session_pitch":"1001_1"}]')
            with patch.object(a, 'METRICS_SHA256', a.file_digest(metrics / 'results/pitch_metrics.csv')):
                with self.assertRaisesRegex(ValueError, 'excluded_pitch_reintroduced'):
                    a.load_inputs(metrics, data)

    def test_main_dispatch_and_output_boundary(self):
        package = Path(a.__file__).parent
        summary = dict(players=98, unordered_pairs=666, filter_stable_nonincrease_pairs=0,
                       stable_gap_at_least_1kmh_pairs=0, observational_finding='ABSENT')
        with patch('sys.argv', ['analysis', '--metrics-dir', 'metrics', '--data-dir', 'data']), \
                patch.object(a, 'run_analysis', return_value=summary) as mocked, \
                contextlib.redirect_stdout(io.StringIO()):
            a.main()
            self.assertEqual(mocked.call_args.args[2], package / 'results')
        with tempfile.TemporaryDirectory() as temporary, \
                patch('sys.argv', ['analysis', '--metrics-dir', 'metrics', '--data-dir', 'data', '--output', temporary]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            a.main()


if __name__ == '__main__':
    unittest.main()
