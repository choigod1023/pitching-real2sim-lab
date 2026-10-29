"""Synthetic algorithm checks, separate from all athlete-level conclusions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ..research_reset.validation import FEATURES, GRF, file_digest
from .signals import (ADDED, NEW_FEATURES, POWER, SOURCE_HASHES, event_window,
                      force_representation, load_features, phase_samples)
from .models import FAMILIES, choose_family, fit_predict, nested_loao, pair_data, pitch_map
from .guard import guard_indices, guard_verdict, guarded_case, load_pilot, run_guard
from . import guard as guard_module
from . import run as run_module


def sample_pitches(players=6):
    rows = []
    for player in range(players):
        for trial in range(3):
            row = dict(user=str(player), session=str(player), session_pitch=f'{player}_{trial}',
                       pitch_speed_kmh=110.+player+trial)
            row.update({name: float(trial)+index*.01+player*.1 for index, name in enumerate(NEW_FEATURES)})
            rows.append(row)
    return pd.DataFrame(rows)


def signal_frame(rate, names):
    time = np.arange(int(rate)+1)/rate
    frame = pd.DataFrame(dict(time=time, fp_10_time=.4, BR_time=.8))
    for i, name in enumerate(names):
        frame[name] = np.sin(2*np.pi*3*time)+i+2
    return frame


class NewSignalTests(unittest.TestCase):
    def test_events_and_filtered_phase_values(self):
        frame = signal_frame(360, POWER)
        self.assertEqual(event_window(frame), (.4, .8))
        values, support = phase_samples(frame, POWER, (.4, .8), 20, 360)
        self.assertEqual(values.shape, (5, 5))
        expected = np.sin(2*np.pi*3*np.linspace(.4, .8, 5))+2
        np.testing.assert_allclose(values[:, 0], expected, atol=.001)
        self.assertEqual(support['filter_samples'], 361)
        bad = frame.copy(); bad.loc[0, 'BR_time'] = .9
        with self.assertRaises(ValueError):
            event_window(bad)

    def test_no_interpolation_across_missing_signal_or_bad_grid(self):
        frame = signal_frame(360, POWER)
        frame.loc[180, POWER[0]] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing_window'):
            phase_samples(frame, POWER, (.4, .8), 20, 360)
        frame.loc[180, POWER[0]] = 1.
        with self.assertRaisesRegex(ValueError, 'duplicate_time'):
            phase_samples(pd.concat([frame, frame.iloc[:1]]), POWER, (.4, .8), 20, 360)
        with self.assertRaisesRegex(ValueError, 'unbracketed'):
            phase_samples(frame, POWER, (-.1, .8), 20, 360)
        frame.loc[100, 'time'] += .001
        with self.assertRaisesRegex(ValueError, 'sampling_grid'):
            phase_samples(frame, POWER, (.4, .8), 20, 360)

    def test_contiguous_finite_block_excludes_missing_padding(self):
        frame = signal_frame(360, POWER)
        frame.loc[:30, POWER] = np.nan
        _, support = phase_samples(frame, POWER, (.4, .8), 20, 360)
        self.assertAlmostEqual(support['left_time_s'], 31/360)
        small = frame.iloc[150:164].copy()
        with self.assertRaisesRegex(ValueError, 'insufficient_filter_support'):
            phase_samples(small, POWER, (small.time.iloc[2], small.time.iloc[-2]), 20, 360)

    def test_force_norm_rotation_invariance_and_zero_support(self):
        x = np.array([[3., 4, 0, 0, 0, 10], [0., 0, 0, 0, 0, 0]])
        values, zero = force_representation(x)
        np.testing.assert_allclose(values[0], [5, 10, 2/3])
        self.assertEqual(zero, 1)
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        rotated = np.column_stack([x[:, :3]@rotation, x[:, 3:]@rotation])
        np.testing.assert_allclose(force_representation(rotated)[0], values)
        with self.assertRaises(ValueError):
            force_representation(np.zeros((2, 5)))

    def test_load_features_frozen_sources_and_common_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp); (data/'full_sig').mkdir()
            base = sample_pitches(2)
            metadata = base[['user', 'session', 'session_pitch']].copy()
            metadata['filename_new'] = metadata.session_pitch+'.c3d'
            metadata['pitch_speed_mph'] = base.pitch_speed_kmh/1.609344
            metadata.to_csv(data/'metadata.csv', index=False)
            metrics = pd.concat([base.assign(cutoff_hz=c) for c in (10, 20, 30)])
            metrics.to_csv(data/'metrics.csv', index=False)
            energy = pd.concat([signal_frame(360, POWER).assign(session_pitch=p) for p in base.session_pitch])
            force = pd.concat([signal_frame(1080, GRF).assign(session_pitch=p) for p in base.session_pitch])
            energy.to_csv(data/'full_sig/energy_flow.csv', index=False)
            force.to_csv(data/'full_sig/force_plate.csv', index=False)
            manifest = data/'manifest.json'
            manifest.write_text(json.dumps({'sources': {'metadata.csv': {'sha256': file_digest(data/'metadata.csv')}}}))
            hashes = {name: file_digest(data/'full_sig'/name) for name in SOURCE_HASHES}
            with patch('coach_hypotheses.new_signal.signals.METRICS_SHA256', file_digest(data/'metrics.csv')), \
                    patch.dict(SOURCE_HASHES, hashes, clear=True):
                tables, audit = load_features(data, data/'metrics.csv', manifest)
            self.assertEqual(audit['pitches'], 6)
            self.assertEqual(len(NEW_FEATURES), 48)
            self.assertEqual(len(ADDED), 41)
            self.assertEqual(tables[20][NEW_FEATURES].shape, (6, 48))
            self.assertFalse(audit['physical_contact_or_work_budget_validated'])
            # numpy searchsorted indices must not leak into JSON metadata.
            self.assertEqual(json.loads(json.dumps(audit, allow_nan=False))['pitches'], 6)
            with self.assertRaisesRegex(ValueError, 'changed_metrics'):
                load_features(data, data/'metrics.csv', manifest)

    def test_pair_data_exact_target_and_differences(self):
        frame = sample_pitches(2)
        pairs, dx = pair_data(frame, frame[NEW_FEATURES].to_numpy())
        self.assertEqual(len(pairs), 6)
        np.testing.assert_allclose(dx[0], np.ones(48))
        self.assertEqual(pairs.truth_delta_kmh.iloc[0], 1)
        with self.assertRaisesRegex(ValueError, 'duplicate_pitch'):
            pair_data(pd.concat([frame, frame.iloc[:1]]), np.zeros((7, 2)))

    def test_nonlinear_mapping_uses_only_training_statistics(self):
        frame = sample_pitches()
        train, test = frame[frame.user != '0'], frame[frame.user == '0']
        x, y, model = pitch_map(train, test, 'new48_rff')
        changed = test.copy(); changed[NEW_FEATURES] += 1e3
        other_x, _, other = pitch_map(train, changed, 'new48_rff')
        np.testing.assert_array_equal(x, other_x)
        np.testing.assert_array_equal(model['mean'], other['mean'])
        self.assertEqual(y.shape, (3, 64))
        self.assertEqual(pitch_map(train, test, 'stable7')[0].shape[1], 7)
        with self.assertRaises(ValueError):
            pitch_map(train, test, 'unknown')

    def test_families_ignore_heldout_labels_and_reverse_pair_direction(self):
        frame = sample_pitches()
        train, test = frame[frame.user != '0'], frame[frame.user == '0']
        for family in FAMILIES:
            before, _ = fit_predict(train, test, family)
            changed = test.copy(); changed['pitch_speed_kmh'] = [1e6, -1e6, 42.]
            after, _ = fit_predict(train, changed, family)
            np.testing.assert_array_equal(before.predicted_pair_delta_kmh, after.predicted_pair_delta_kmh)
            reverse = test.iloc[:2].copy()
            reverse['session_pitch'] = reverse.session_pitch.to_numpy()[::-1]
            reversed_result, _ = fit_predict(train, reverse, family)
            self.assertAlmostEqual(reversed_result.predicted_pair_delta_kmh.iloc[0], -before.predicted_pair_delta_kmh.iloc[0])

    def test_family_selection_tie_prefers_null_and_splits_are_disjoint(self):
        frame = sample_pitches(4)
        frame['pitch_speed_kmh'] = 100.
        self.assertEqual(choose_family(frame)[0], 'null')
        predictions, summaries, splits = nested_loao(frame)
        self.assertEqual(set(predictions), set(FAMILIES+('selector',)))
        self.assertTrue(all(s['players_disjoint'] for s in splits))
        self.assertEqual(summaries['selector']['model_mae_kmh'], 0.)
        with self.assertRaises(ValueError):
            nested_loao(frame[frame.user == '0'])

    def test_new_signal_main_smoke_and_output_flags(self):
        frame = sample_pitches(4)
        tables = {c: frame for c in (10, 20, 30)}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(run_module, 'load_features', return_value=(tables, dict(players=4, pitches=12))), \
                patch('sys.argv', ['run', '--data', 'unused', '--metrics', 'unused', '--source-manifest', 'unused', '--output', temp]), \
                contextlib.redirect_stdout(io.StringIO()):
            run_module.main()
            result = json.loads((Path(temp)/'new_signal_results.json').read_text())
            self.assertFalse(result['verdict']['development_new_signal_pass'])
            self.assertFalse(result['verdict']['coaching_recommendation_ready'])
            self.assertTrue(all(all(v.values()) for v in result['invariance'].values()))
            self.assertTrue((Path(temp)/'checkpoint_20.json').exists())


class GuardTests(unittest.TestCase):
    def test_old_overlap_and_disjoint_new_stencil(self):
        time = np.arange(60)/200-.15
        train, test, support, audit = guard_indices(time)
        self.assertEqual(audit['old_train_derivative_shared_samples'], 1)
        self.assertEqual(audit['strict_train_derivative_shared_samples'], 0)
        self.assertEqual(len(np.intersect1d(train, support)), 0)
        self.assertEqual(len(test), 7)
        with self.assertRaises(ValueError):
            guard_indices(time[:5])

    def test_pinned_loader_rejects_changed_code(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'pilot.py'; source.write_text('TOKEN=1\n')
            with self.assertRaisesRegex(ValueError, 'changed_pilot'):
                load_pilot(source)
            with patch.object(guard_module, 'PILOT_SHA256', file_digest(source)):
                self.assertEqual(load_pilot(source).TOKEN, 1)
            self.assertFalse((Path(temp)/'__pycache__').exists())

    def test_guarded_case_checks_window_and_metric_units(self):
        n = 60; time = np.arange(n)/200
        ball = np.column_stack([time, np.zeros((n, 2))])
        predicted = ball.copy(); predicted[:, 0] += .001
        source = dict(time=time, rate=200., events=[dict(label='Event', time=.15)],
                      markers={'Ball': ball}, valid={'Ball': np.ones(n, bool)})
        module = SimpleNamespace(dynamic_hand=Mock(return_value=(np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1)), np.ones(n, bool), None)),
                                 fit_offset=Mock(return_value=np.zeros((2, 3))),
                                 predict_marker=Mock(return_value=predicted))
        rows = guarded_case(module, source, None, 'R', 0)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['marker_position_rmse_mm'], 1.)
        self.assertAlmostEqual(rows[0]['marker_speed_mae_kmh'], 0.)
        source['valid']['Ball'][:] = False
        with self.assertRaisesRegex(ValueError, 'missing_guard'):
            guarded_case(module, source, None, 'R', 0)

    def test_guard_verdict_requires_both_position_and_speed_per_case(self):
        rows = [dict(trial='A', event_shift_frames=0, model='rigid_offset', marker_position_rmse_mm=2., marker_speed_mae_kmh=1.),
                dict(trial='A', event_shift_frames=0, model='linear_relative_offset', marker_position_rmse_mm=1., marker_speed_mae_kmh=.5)]
        self.assertTrue(guard_verdict(rows)['pilot_joint_improvement'])
        rows[1]['marker_speed_mae_kmh'] = 2.
        self.assertFalse(guard_verdict(rows)['pilot_joint_improvement'])
        self.assertFalse(guard_verdict(rows)['center_speed_validated'])

    def test_guard_run_main_and_hash_checks(self):
        rows = [dict(model='rigid_offset', event_shift_frames=0, marker_position_rmse_mm=2., marker_speed_mae_kmh=1.),
                dict(model='linear_relative_offset', event_shift_frames=0, marker_position_rmse_mm=1., marker_speed_mae_kmh=.5)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'a.c3d').write_bytes(b'synthetic')
            summary = root/'summary.json'; summary.write_text(json.dumps({'inputs_sha256': {'a.c3d': file_digest(root/'a.c3d')}}))
            module = SimpleNamespace(CASES=[('a.c3d', 'a.c3d', 'R')], load_c3d=Mock(return_value={}), calibrate_wrist=Mock(return_value={}))
            with patch.object(guard_module, 'load_pilot', return_value=module), \
                    patch.object(guard_module, 'guarded_case', return_value=rows), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = run_guard(root, root/'module', summary, root/'output')
                self.assertFalse(result['verdict']['center_speed_validated'])
                with patch('sys.argv', ['guard', '--data', str(root), '--pilot-module', str(root/'module'), '--pilot-summary', str(summary), '--output', str(root/'output')]):
                    guard_module.main()
                (root/'a.c3d').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'changed_guard_source'):
                    run_guard(root, root/'module', summary, root/'output')


if __name__ == '__main__':
    unittest.main()
