import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import analysis as a


def paired_fixture():
    pairs = pd.DataFrame([dict(user='1', pitch_i='a', pitch_j='b')])
    nominal = pd.DataFrame([dict(user='1', pitch_i='a', pitch_j='b', cutoff_hz=c,
                                 faster_pitch='a', slower_pitch='b', recorded_speed_gap_kmh=2.) for c in a.CUTOFFS])
    states = {(p, c): np.tile(row, (9, 1)) for p, row in [('a', [8., 90., -10., 20.]),
                                                         ('b', [10., 100., -5., 22.])] for c in a.CUTOFFS}
    return pairs, nominal, states


def event_fixture():
    time = np.arange(361) / 360.
    signals = dict(elbow_moment_y=np.full(361, 10.), shoulder_upper_arm_force_x=np.full(361, 3.),
                   shoulder_upper_arm_force_y=np.full(361, 4.), shoulder_upper_arm_force_z=np.full(361, -5.),
                   lead_knee_angle_x=-100 * time, lead_knee_velo_x=np.full(361, 100.))
    module = SimpleNamespace(filtered_signals=lambda *args: signals,
                             window_peak=lambda t, v, start, end: (0., float(max(v[(t >= start) & (t <= end)])), False),
                             at_event=lambda t, v, event: float(np.interp(event, t, v)))
    prepared = (time, None, None, dict(pkh_time=.2, fp_10_time=.4, BR_time=.8), None)
    expected = dict(elbow_varus_peak_nm=10., shoulder_resultant_peak_n=np.sqrt(50.),
                    shoulder_distraction_peak_n=5., lead_knee_extension_deg=40.)
    return module, prepared, expected


class SensitivityTests(unittest.TestCase):
    def test_error_margins_known_bounds_and_strict_speed_order(self):
        high, low = np.array([8., 90., -10.]), np.array([10., 100., -5.])
        margins = a.error_margins(high, low, 2.)
        self.assertEqual(margins['speed_order_error_limit_kmh'], 1.)
        self.assertEqual(margins['gap_at_least_1_error_limit_kmh'], .5)
        self.assertEqual(margins['elbow_peak_error_limit_nm'], 1.)
        self.assertEqual(margins['shoulder_resultant_error_limit_n'], 5.)
        self.assertEqual(margins['signed_tension_error_limit_n'], 10.)
        self.assertTrue(a.survives(high, low, 2., [.99, 1., 5.]))
        self.assertFalse(a.survives(high, low, 2., [1., 0., 0.]))
        self.assertTrue(a.survives(high, low, 2., [.5, 0., 0.], 1.))
        self.assertFalse(a.survives(high, low, 2., [.501, 0., 0.], 1.))

    def test_tension_piecewise_limit_for_compression_tension_and_mixed(self):
        for fast, slow in [(-10., -5.), (-5., -10.), (-10., 5.), (-5., 10.), (2., 8.), (0., 0.)]:
            high, low = [0., 0., fast], [1000., 1000., slow]
            limit = a.error_margins(high, low, 100.)['signed_tension_error_limit_n']
            self.assertTrue(a.survives(high, low, 100., [0., 0., limit]))
            self.assertFalse(a.survives(high, low, 100., [0., 0., limit + .001]))
        with self.assertRaises(ValueError):
            a.error_margins([3., 1., 1.], [2., 2., 2.], 1.)

    def test_stress_input_checks_and_broadcast(self):
        high = np.tile([1., 2., -5.], (3, 9, 1, 1))
        low = np.tile([2., 3., -4.], (3, 1, 9, 1))
        self.assertEqual(a.survives(high, low, 1., [0., 0., 0.]).shape, (3, 9, 9))
        for profile in ([-1, 0, 0], [0, 0], [0, np.nan, 0]):
            with self.assertRaises(ValueError):
                a.survives([1, 2, 3], [2, 3, 4], 1., profile)

    def test_event_states_nominal_and_known_knee_shifts(self):
        module, prepared, expected = event_fixture()
        states, audit, difference = a.event_states(module, prepared, 20, expected)
        self.assertEqual(states.shape, (9, 4))
        self.assertLess(difference, 1e-12)
        self.assertAlmostEqual(states[-1, 3] - states[0, 3], 200 / 360.)
        self.assertGreater(audit['samples'], 3)
        with self.assertRaisesRegex(ValueError, 'nominal_reproduction_failed'):
            a.event_states(module, prepared, 20, dict(expected, elbow_varus_peak_nm=11.))
        changed = list(prepared)
        changed[3] = dict(prepared[3], pkh_time=0.)
        with self.assertRaisesRegex(ValueError, 'shifted_event'):
            a.event_states(module, changed, 20, expected)

    def test_knee_direction_and_invalid_support(self):
        time = np.arange(361) / 360.
        audit = a.knee_direction_audit(time, -50 * time ** 2, 100 * time, .2, .8)
        self.assertGreater(audit['extension_derivative_correlation'], .999999)
        self.assertLess(audit['same_direction_mae_dps'], 1e-10)
        self.assertGreater(audit['opposite_direction_mae_dps'], 50.)
        for grid, start, stop in [(np.ones(361), .2, .8), (time, .99, .99)]:
            with self.assertRaises(ValueError):
                a.knee_direction_audit(grid, -50 * time ** 2, 100 * time, start, stop)

    def test_all_81_combinations_and_error_profiles(self):
        pairs, nominal, states = paired_fixture()
        records, events, margins = a.inspect_pairs(pairs, nominal, states)
        self.assertEqual(len(events), 243)
        self.assertEqual(len(margins), 3)
        self.assertTrue(records[0]['independent_event_stable'])
        self.assertTrue(records[0]['profiles'][1]['independent_event'])
        self.assertFalse(records[0]['profiles'][2]['independent_event'])
        self.assertAlmostEqual(records[0]['minimum_margins_across_filters']['joint_shoulder_error_limit_n'], 5.)

    def test_common_offset_can_pass_while_independent_offset_fails(self):
        pairs, nominal, states = paired_fixture()
        for c in a.CUTOFFS:
            states[('a', c)][0, 0] = 11.
            states[('b', c)][0, 0] = 13.
        records, _, _ = a.inspect_pairs(pairs, nominal, states)
        self.assertTrue(records[0]['common_event_stable'])
        self.assertFalse(records[0]['independent_event_stable'])
        self.assertEqual(records[0]['worst_elbow_delta_nm'], 1.)
        states[('a', 20)][a.NOMINAL, 0] = 20.
        with self.assertRaisesRegex(ValueError, 'prior_nominal'):
            a.inspect_pairs(pairs, nominal, states)

    def test_summary_flags_profiles_and_no_records(self):
        pairs, nominal, states = paired_fixture()
        records, _, _ = a.inspect_pairs(pairs, nominal, states)
        audits = [dict(extension_derivative_correlation=1., same_direction_mae_dps=1., opposite_direction_mae_dps=100.)]
        result = a.summarize(records, audits, 0.)
        self.assertEqual(result['independent_event_stable_pairs'], 1)
        self.assertEqual(result['engineering_profiles'][1]['independent_event_gap_at_least_1_pairs'], 1)
        for flag in ('measurement_robustness_validated', 'same_capacity_validated', 'intervention_effect_validated',
                     'coaching_recommendation_ready', 'speed_prediction_gate_changed'):
            self.assertIs(result[flag], False)
        with self.assertRaises(ValueError):
            a.summarize([], audits, 0.)

    def test_prior_hash_and_cohort_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = pd.DataFrame([dict(user=str(i), pitch_i='a', pitch_j='b', recorded_speed_gap_kmh=2.,
                                        filter_stable_observed_nonincrease=True) for i in range(59)])
            nominal = pd.DataFrame([dict(user=str(i), pitch_i='a', pitch_j='b', cutoff_hz=c,
                                          status='observed_nonincrease') for i in range(59) for c in a.CUTOFFS])
            pairs.to_csv(root / 'all_pair_stability.csv', index=False)
            nominal.to_csv(root / 'all_pair_filter_comparisons.csv', index=False)
            (root / 'summary.json').write_text('{}')
            hashes = {name: a.file_digest(root / name) for name in a.PRIOR_HASHES}
            with patch.object(a, 'PRIOR_HASHES', hashes):
                selected, rows = a.load_prior(root)
                self.assertEqual((len(selected), len(rows)), (59, 177))
                (root / 'summary.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'changed_prior'):
                    a.load_prior(root)

    def test_frozen_source_loader_and_source_hash_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'full_sig').mkdir()
            (root / 'extractor/results').mkdir(parents=True)
            source = root / 'full_sig/forces_moments.csv'
            pd.DataFrame(dict(session_pitch=['a', 'a'], time=[0., 1.], pkh_time=[0., 0.],
                              fp_10_time=[.5, .5], BR_time=[1., 1.], x=[1., 2.])).to_csv(source, index=False)
            extractor = root / 'extractor/metrics.py'
            extractor.write_text("COLUMNS={'forces_moments':['x']}\nEVENTS=['pkh_time','fp_10_time','BR_time']\ndef prepare_pitch(tables):\n return tables['forces_moments']\n")
            manifest = {'sources': {'full_sig/forces_moments.csv': {'sha256': a.file_digest(source)}}}
            (root / 'extractor/results/manifest.json').write_text(json.dumps(manifest))
            with patch.object(a, 'EXTRACTOR_SHA256', a.file_digest(extractor)):
                _, prepared, _ = a.load_sources(root, extractor, {'a'})
                self.assertEqual(len(prepared['a']), 2)
                with self.assertRaisesRegex(ValueError, 'missing_full_signal_pitch'):
                    a.load_sources(root, extractor, {'a', 'missing'})
                source.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed_full_signal'):
                    a.load_sources(root, extractor, {'a'})
            with self.assertRaisesRegex(ValueError, 'changed_upstream_extractor'):
                a.load_sources(root, extractor, {'a'})

    def test_run_output_and_main_dispatch(self):
        pairs, nominal, states = paired_fixture()
        records, events, margins = a.inspect_pairs(pairs, nominal, states)
        audit = dict(samples=5, extension_derivative_correlation=1., same_direction_mae_dps=1., opposite_direction_mae_dps=100.)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'results').mkdir()
            pd.DataFrame([dict(session_pitch=p, cutoff_hz=c) for p in ('a', 'b') for c in a.CUTOFFS]).to_csv(root / 'results/pitch_metrics.csv', index=False)
            with patch.object(a, 'load_prior', return_value=(pairs, nominal)), \
                    patch.object(a, 'load_sources', return_value=(None, {'a': None, 'b': None}, {})), \
                    patch.object(a, 'file_digest', return_value='c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4'), \
                    patch.object(a, 'event_states', return_value=(np.zeros((9, 4)), audit, 0.)), \
                    patch.object(a, 'inspect_pairs', return_value=(records, pd.DataFrame({'case': range(14337)}), margins)), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = a.run(root, root, root, root / 'output')
                self.assertEqual(result['selected_pairs'], 1)
                self.assertTrue((root / 'output/manifest.json').exists())
        with patch('sys.argv', ['analysis', '--prior-results', 'prior', '--metrics-dir', 'metrics', '--data-dir', 'data']), \
                patch.object(a, 'run') as mocked:
            a.main()
            self.assertEqual(mocked.call_args.args[3], Path(a.__file__).with_name('results'))
        with tempfile.TemporaryDirectory() as temporary, \
                patch('sys.argv', ['analysis', '--prior-results', 'prior', '--metrics-dir', 'metrics', '--data-dir', 'data', '--output', temporary]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            a.main()


if __name__ == '__main__':
    unittest.main()
