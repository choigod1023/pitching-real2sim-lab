import contextlib
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import audit as a


def peak_fixture():
    time = np.arange(361) / 360.
    pelvis = 20 + 100 * np.exp(-((time - .6) / .04) ** 2)
    torso = 20 + 100 * np.exp(-((time - .65) / .04) ** 2)
    events = dict(pkh_time=.2, fp_10_time=.4, BR_time=.9)
    signals = dict(pelvis_velo_z=pelvis, torso_velo_z=torso)
    module = SimpleNamespace(filtered_signals=lambda *args: signals)
    mask = (time >= events['pkh_time']) & (time <= events['BR_time'])
    pi, ti = (np.flatnonzero(mask)[np.argmax(v[mask])] for v in (pelvis, torso))
    expected = []
    for c in a.CUTOFFS:
        expected.append(dict(cutoff_hz=c, sequence_lag_ms=(time[ti] - time[pi]) * 1000,
                             pelvis_peak_dps=pelvis[pi], torso_peak_dps=torso[ti],
                             pelvis_peak_time_fp10_ms=(time[pi] - events['fp_10_time']) * 1000,
                             torso_peak_time_fp10_ms=(time[ti] - events['fp_10_time']) * 1000))
    return module, (time, None, None, events, None), pd.DataFrame(expected).set_index('cutoff_hz')


def pair_fixture():
    rows, pitches = [], {}
    for i in range(35):
        user = str(i % 23)
        fast, slow = 'f' + str(i), 's' + str(i)
        rows.append(dict(user=user, faster_pitch=fast, slower_pitch=slow, independent_event_stable=True))
        for pitch, base, lo, hi in [(fast, 5., 7, 10), (slow, 1., 2, 3)]:
            pitches[pitch] = dict(user=user, nominal_lag_ms={str(c): base + c / 10 for c in a.CUTOFFS},
                                  ranges={str(r): dict(minimum_lag_steps=lo, maximum_lag_steps=hi) for r in a.TOLERANCES})
    return {'pair_records': rows}, pitches


class PeakTests(unittest.TestCase):
    def test_order_resolution_is_sample_based(self):
        self.assertEqual(a.order_state(2, 3), 'torso_after')
        self.assertEqual(a.order_state(-3, -2), 'torso_before')
        self.assertEqual(a.order_state(-2, 2), 'both_orders_possible')
        self.assertEqual(a.order_state(1, 4), 'one_sample_or_overlap')
        self.assertEqual(a.order_state(-1, 1), 'one_sample_or_overlap')
        for bounds in [(2, 1), (np.nan, 1)]:
            with self.assertRaises(ValueError): a.order_state(*bounds)

    def test_flat_top_and_disconnected_candidates(self):
        time = np.arange(9) / 360.
        values = [1., 8., 9.96, 10., 9.97, 5., 9.94, 9.98, 1.]
        peak = a.describe_peak(time, values, 0., time[-1])
        self.assertEqual(peak['index'], 3)
        band = peak['bands']['0.01']
        self.assertEqual((band['first_index'], band['last_index'], band['components']), (2, 7, 2))
        self.assertAlmostEqual(band['selected_component_width_ms'], 2000 / 360)
        self.assertAlmostEqual(band['width_ms'], 5000 / 360)
        self.assertFalse(peak['on_window_boundary'])

    def test_exact_tie_uses_earliest_and_boundary(self):
        time = np.arange(4) / 360.
        peak = a.describe_peak(time, [1, 10, 10, 1], 0, time[-1])
        self.assertEqual(peak['index'], 1)
        self.assertEqual(peak['bands']['0.01']['samples'], 2)
        boundary = a.describe_peak(time, [1, 2, 3, 4], 0, time[-1])
        self.assertTrue(boundary['on_window_boundary'])

    def test_peak_support_and_positive_relative_peak_required(self):
        for time, values, start, end in [([0, 1], [-1, -2], 0, 1),
                                         ([0, 0], [1, 2], 0, 1),
                                         ([0, 1], [1, np.nan], 0, 1),
                                         ([0, 1], [1, 2], -.1, 1),
                                         ([0, 1], [1, 2], .2, .3)]:
            with self.assertRaises(ValueError): a.describe_peak(time, values, start, end)

    def test_nominal_reproduction_and_all_event_cases(self):
        module, prepared, reference = peak_fixture()
        result, rows = a.audit_pitch(module, prepared, reference)
        self.assertEqual(len(rows), 54)
        self.assertEqual(result['nominal_reproduction_error'], 0.)
        self.assertEqual(result['nominal_order_across_filters'], 'torso_after')
        self.assertEqual(result['ranges']['0.01']['classification'], 'torso_after')
        self.assertTrue(all(r['compatible_with_fixed20_one_percent_band'] for r in result['filter_peak_movements']['torso']))
        changed = reference.copy(); changed.loc[20, 'sequence_lag_ms'] += 1.
        with self.assertRaisesRegex(ValueError, 'reproduction_failed'): a.audit_pitch(module, prepared, changed)

    def test_event_order_or_support_failure(self):
        module, prepared, reference = peak_fixture()
        changed = list(prepared); changed[3] = dict(prepared[3], pkh_time=0.)
        with self.assertRaisesRegex(ValueError, 'support'): a.audit_pitch(module, changed, reference)
        changed[3] = dict(prepared[3], BR_time=prepared[3]['fp_10_time'])
        with self.assertRaisesRegex(ValueError, 'event_order'): a.audit_pitch(module, changed, reference)

    def test_pair_ranges_are_differences_not_segment_order(self):
        source, pitches = pair_fixture()
        result = a.audit_pairs(source, pitches)
        self.assertEqual((len(result['pairs']), len(result['players'])), (35, 23))
        self.assertEqual(result['classifications']['0.01']['delta_positive'], 35)
        self.assertEqual(result['pairs'][0]['ranges']['0.01']['minimum_delta_steps'], 4)
        self.assertEqual(result['pairs'][0]['ranges']['0.01']['maximum_delta_steps'], 8)
        self.assertEqual(result['player_balanced_delta_ms']['20'], 4.)
        source['pair_records'][0]['user'] = '1'
        with self.assertRaisesRegex(ValueError, 'cross_player'): a.audit_pairs(source, pitches)
        source['pair_records'].pop()
        with self.assertRaisesRegex(ValueError, 'changed_fixed'): a.audit_pairs(source, pitches)

    def test_summary_retains_uncertainty_and_gate_flags(self):
        module, prepared, reference = peak_fixture()
        result, rows = a.audit_pitch(module, prepared, reference)
        result['user'] = '1'
        source, pitches = pair_fixture()
        summary = a.summarize({'p': result}, rows, a.audit_pairs(source, pitches))
        self.assertEqual(summary['segments']['torso']['nominal_pitch_filter_cases'], 3)
        self.assertEqual(summary['interval_order_classifications']['0.01']['torso_after'], 1)
        for flag in ('calibrated_timing_accuracy_validated', 'timing_feature_ready_for_coaching',
                     'speed_prediction_gate_changed', 'new_speed_model_fitted',
                     'athlete_effect_validated', 'spin_or_movement_improvement_validated'):
            self.assertIs(summary[flag], False)

    def test_changed_inputs_fail_and_cli_boundary(self):
        with patch.object(a, 'file_digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'changed_frozen'): a.run('data', 'metrics', 'summary', 'out')
        with patch('sys.argv', ['audit', '--data-dir', 'd', '--metrics-dir', 'm', '--source-summary', 's']), patch.object(a, 'run') as mocked:
            a.main()
            self.assertEqual(mocked.call_args.args[0:3], (a.Path('d'), a.Path('m'), a.Path('s')))
        with tempfile.TemporaryDirectory() as directory, patch('sys.argv', ['audit', '--data-dir', 'd', '--metrics-dir', 'm', '--source-summary', 's', '--output', directory]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            a.main()


if __name__ == '__main__':
    unittest.main()
