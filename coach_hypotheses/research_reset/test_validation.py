"""Policy/algorithm fixtures are software checks, not athlete validation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from .validation import (CUTOFFS, FEATURES, GRF, IDS, audit_c3d, audit_grf,
                         development_verdict, file_digest, fit_pair_ridge, make_pairs,
                         pair_loao, pair_summary, predict_pairs, validate_metrics, verify_sources)
from .run import main, run


def fixture_tables(players=4):
    metadata, metrics, force = [], [], []
    for player in range(players):
        for trial in range(3):
            pitch = f'{player}_{trial}'
            speed = 70.+player+trial
            metadata.append(dict(session_pitch=pitch, user=str(player), session=str(player),
                                 filename_new=pitch+'.c3d', pitch_speed_mph=speed))
            for cutoff in CUTOFFS:
                row = dict(session_pitch=pitch, user=str(player), session=str(player),
                           pitch_speed_kmh=speed*1.609344, cutoff_hz=cutoff)
                row.update({name: float(trial) if index == 0 else 0.
                            for index, name in enumerate(FEATURES)})
                metrics.append(row)
            for time in (-.1, 0., .5, 1., 1.1):
                row = dict(session_pitch=pitch, time=time, pkh_time=0., BR_time=1.)
                row.update({name: 1. for name in GRF})
                force.append(row)
    return pd.DataFrame(metadata), pd.DataFrame(metrics), pd.DataFrame(force)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.metadata, self.metrics, self.force = fixture_tables()
        self.primary = self.metrics[self.metrics.cutoff_hz == 20]
        self.pairs = make_pairs(self.primary)

    def test_hash_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'source.txt'
            data.write_text('frozen', encoding='utf-8')
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'sources': {'source.txt': {'sha256': file_digest(data)}}}))
            self.assertTrue(verify_sources(root, manifest)['source.txt']['verified'])
            data.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed_frozen_source'):
                verify_sources(root, manifest)
            manifest.write_text(json.dumps({'sources': {'../outside': {'sha256': 'irrelevant'}}}))
            with self.assertRaisesRegex(ValueError, 'source_outside_data'):
                verify_sources(root, manifest)
            manifest.write_text('{"sources": {}}')
            with self.assertRaisesRegex(ValueError, 'empty_source_manifest'):
                verify_sources(root, manifest)

    def test_metric_keys_units_and_cutoffs(self):
        cohort = validate_metrics(self.metrics, self.metadata)
        self.assertEqual(cohort['pitches'], 12)
        self.assertEqual(cohort['players'], 4)
        bad = self.metrics.copy()
        bad.loc[0, 'pitch_speed_kmh'] += 1.
        with self.assertRaisesRegex(ValueError, 'speed_unit_or_link'):
            validate_metrics(bad, self.metadata)
        bad = self.metrics.drop(index=0)
        with self.assertRaisesRegex(ValueError, 'cohort_mismatch'):
            validate_metrics(bad, self.metadata)
        with self.assertRaisesRegex(ValueError, 'duplicate_metric_key'):
            validate_metrics(pd.concat([self.metrics, self.metrics.iloc[:1]]), self.metadata)

    def test_nonfinite_identity_and_metadata_duplicates_rejected(self):
        for column, value, message in [(FEATURES[0], np.nan, 'nonfinite'),
                                        ('user', 'other', 'player_mismatch'),
                                        ('session', 'other', 'session_mismatch'),
                                        ('session_pitch', 'absent', 'unlinked')]:
            bad = self.metrics.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                validate_metrics(bad, self.metadata)
        with self.assertRaisesRegex(ValueError, 'duplicate_metadata'):
            validate_metrics(self.metrics, pd.concat([self.metadata, self.metadata.iloc[:1]]))

    def test_c3d_availability_requires_unique_filename_not_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'c3d').mkdir()
            for name in self.metadata.filename_new:
                (root/'c3d'/name).write_bytes(b'fixture availability only')
            self.assertEqual(audit_c3d(root, self.metadata)['matched_dynamic_files'], 12)
            (root/'c3d/duplicate').mkdir()
            (root/'c3d/duplicate'/self.metadata.filename_new.iloc[0]).write_bytes(b'duplicate')
            with self.assertRaisesRegex(ValueError, 'ambiguous_c3d'):
                audit_c3d(root, self.metadata)

    def test_grf_window_missingness_does_not_become_contact_validation(self):
        force = self.force.copy()
        force.loc[0, GRF[0]] = np.nan
        audit = audit_grf(force, self.metadata, self.primary.session_pitch)
        self.assertEqual(audit['all_rows_finite_pitches'], 11)
        self.assertEqual(audit['pkh_br_finite_pitches'], 12)
        self.assertEqual(audit['missing_rows'], 1)
        self.assertEqual(audit['selected_not_all_finite'], ['0_0'])
        self.assertFalse(audit['physical_contact_validated'])
        force.loc[1, GRF[0]] = np.nan
        self.assertEqual(audit_grf(force, self.metadata, [])['pkh_br_finite_pitches'], 11)
        absent = force[force.session_pitch != '0_0']
        self.assertEqual(audit_grf(absent, self.metadata, [])['missing_metadata_pitches'], ['0_0'])

    def test_grf_duplicate_time_and_invalid_events_fail(self):
        for field, value in [('pkh_time', np.nan), ('BR_time', -.5), ('time', np.nan)]:
            bad = self.force.copy()
            bad.loc[bad.session_pitch == '0_0', field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_grf(bad, self.metadata, [])
        with self.assertRaisesRegex(ValueError, 'duplicate_grf'):
            audit_grf(pd.concat([self.force, self.force.iloc[:1]]), self.metadata, [])

    def test_pair_counts_exact_differences_and_order(self):
        self.assertEqual(len(self.pairs), 12)
        row = self.pairs.iloc[0]
        self.assertEqual((row.pitch_i, row.pitch_j), ('0_0', '0_1'))
        self.assertAlmostEqual(row.truth_delta_kmh, 1.609344)
        self.assertEqual(row[FEATURES[0]], 1.)
        pd.testing.assert_frame_equal(self.pairs, make_pairs(self.primary.sample(frac=1, random_state=1)))

    def test_bad_pair_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_pair_pitch'):
            make_pairs(pd.concat([self.primary, self.primary.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, 'insufficient_pair_repeats'):
            make_pairs(self.primary.iloc[:1])
        bad = self.primary.copy()
        bad.loc[bad.index[0], FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite_pair'):
            make_pairs(bad)

    def test_training_rms_zero_columns_and_prediction_antisymmetry(self):
        model = fit_pair_ridge(self.pairs)
        self.assertAlmostEqual(model['scale'][0], np.sqrt(2.))
        np.testing.assert_array_equal(model['scale'][1:], np.ones(6))
        reverse = self.pairs.copy()
        reverse[FEATURES] *= -1
        np.testing.assert_array_equal(predict_pairs(reverse, model), -predict_pairs(self.pairs, model))
        self.assertLess(np.abs(predict_pairs(self.pairs, model)-self.pairs.truth_delta_kmh).mean(),
                        np.abs(self.pairs.truth_delta_kmh).mean())

    def test_all_heldout_outcomes_can_change_without_changing_own_predictions(self):
        before, splits = pair_loao(self.pairs)
        changed = self.pairs.copy()
        changed.loc[changed.user == '0', 'truth_delta_kmh'] = [1e6, -1e6, 42.]
        after, _ = pair_loao(changed)
        np.testing.assert_array_equal(before[before.user == '0'].predicted_pair_delta_kmh,
                                      after[after.user == '0'].predicted_pair_delta_kmh)
        train = self.pairs[self.pairs.user != '0']
        np.testing.assert_array_equal(fit_pair_ridge(train)['scale'],
                                      fit_pair_ridge(changed[changed.user != '0'])['scale'])
        self.assertTrue(all(s['players_disjoint'] for s in splits))
        self.assertEqual(set(before.user), set(self.pairs.user))

    def test_ridge_and_prediction_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fit_pair_ridge(self.pairs, alpha=0)
        with self.assertRaises(ValueError):
            pair_loao(self.pairs[self.pairs.user == '0'])
        bad = self.pairs.copy()
        bad.loc[0, FEATURES[0]] = np.inf
        with self.assertRaises(ValueError):
            fit_pair_ridge(bad)
        with self.assertRaises(ValueError):
            predict_pairs(bad, fit_pair_ridge(self.pairs))

    def test_summary_clusters_players_instead_of_pairs(self):
        predictions, _ = pair_loao(self.pairs)
        result = pair_summary(predictions)
        duplicated_player = pd.concat([predictions, predictions[predictions.user == '0']]*2)
        repeat = pair_summary(duplicated_player)
        for key in ('model_mae_kmh', 'zero_delta_mae_kmh', 'paired_mae_difference_kmh'):
            self.assertAlmostEqual(result[key], repeat[key])
        self.assertEqual(result['player_bootstrap_95_difference_kmh'],
                         pair_summary(predictions)['player_bootstrap_95_difference_kmh'])
        self.assertFalse(result['prospective_validated'])
        with self.assertRaises(ValueError):
            pair_summary(predictions, repetitions=1)

    def test_all_development_checks_required_and_no_recommendation(self):
        predictions, _ = pair_loao(self.pairs)
        summaries = {str(c): pair_summary(predictions) for c in CUTOFFS}
        cohort = dict(players=98, pitches=399)
        result = development_verdict(summaries, cohort, True)
        self.assertTrue(result['development_pairwise_criteria_pass'])
        self.assertFalse(result['coaching_recommendation_ready'])
        self.assertFalse(result['original_scientific_gate_changed'])
        bad = copy.deepcopy(summaries)
        bad['10']['model_mae_kmh'] = 10.
        self.assertFalse(development_verdict(bad, cohort, True)['development_pairwise_criteria_pass'])
        self.assertFalse(development_verdict(summaries, cohort, False)['development_pairwise_criteria_pass'])
        self.assertFalse(development_verdict(summaries, dict(players=4, pitches=12), True)['development_pairwise_criteria_pass'])

    def test_main_and_end_to_end_audit_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, output = root/'data', root/'results'
            (data/'c3d').mkdir(parents=True)
            (data/'full_sig').mkdir()
            self.metadata.to_csv(data/'metadata.csv', index=False)
            self.force.to_csv(data/'full_sig/force_plate.csv', index=False)
            metrics = root/'metrics.csv'
            self.metrics.to_csv(metrics, index=False)
            for name in self.metadata.filename_new:
                (data/'c3d'/name).write_bytes(b'fixture')
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'sources': {
                'metadata.csv': {'sha256': file_digest(data/'metadata.csv')}}}))
            args = ['run', '--data', str(data), '--metrics', str(metrics),
                    '--source-manifest', str(manifest), '--output', str(output)]
            with patch('coach_hypotheses.research_reset.run.METRICS_SHA256', file_digest(metrics)), \
                    patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((output/'results.json').read_text())
            self.assertEqual(result['cohort']['pitches'], 12)
            self.assertTrue(all(all(x.values()) for x in result['invariance_checks'].values()))
            self.assertFalse(result['verdict']['development_pairwise_criteria_pass'])
            self.assertEqual(len(pd.read_csv(output/'pair_predictions_20.csv')), 12)
            with self.assertRaisesRegex(ValueError, 'changed_frozen_metrics'):
                run(data, metrics, manifest, output)


if __name__ == '__main__':
    unittest.main()
