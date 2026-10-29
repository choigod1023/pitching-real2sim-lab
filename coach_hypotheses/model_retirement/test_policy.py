"""Regression checks for retirement, including cached and direct entry points."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from coach_hypotheses.model_retirement.policy import (
    RetiredPredictionModelError, diagnostic_result, legacy_diagnostic_only,
    model_usage_metadata, require_legacy_diagnostic,
)

ROOT = Path(__file__).resolve().parents[2]
SEARCHES = ('joint_intervention', 'multi_joint_timing', 'constrained_motion_search',
            'observed_candidate_simulation', 'coupled_body_search', 'empirical_budget_search')


def example_state(name='baseline'):
    return dict(name=name, position_m=[0, 0, 1.8], velocity_m_s=[0, 35, 0],
                joint_parameters={}, max_joint_rotation_deg={}, max_ball_shift_mm=0.,
                work_change_j=0., sampled_budget_pass=False)


class RetirementTests(unittest.TestCase):
    def test_only_literal_opt_in_is_accepted(self):
        for value in (False, None, 1, 'true', [], {}):
            with self.subTest(value=value), self.assertRaises(RetiredPredictionModelError):
                require_legacy_diagnostic(value)
        self.assertFalse(require_legacy_diagnostic(True)['prediction_available'])

    def test_metadata_is_fresh_and_does_not_reenable_predictions(self):
        changed = model_usage_metadata()
        changed['prediction_available'] = True
        actual = model_usage_metadata()
        self.assertEqual(actual['model_status'], 'retired')
        self.assertFalse(actual['prediction_available'])
        self.assertFalse(actual['coaching_recommendation_ready'])
        self.assertIsNone(actual['predicted_ball_speed_kmh'])
        self.assertIsNone(actual['predicted_speed_gain_kmh'])

    def test_guard_runs_before_function_and_tags_opted_in_result(self):
        calls = []

        @legacy_diagnostic_only
        def calculation():
            calls.append(True)
            return {'prediction_available': True}

        with self.assertRaises(RetiredPredictionModelError):
            calculation()
        self.assertEqual(calls, [])
        result = calculation(legacy_diagnostic=True)
        self.assertEqual(calls, [True])
        self.assertFalse(result['prediction_available'])

    def test_guard_preserves_non_dictionary_return(self):
        @legacy_diagnostic_only
        def calculation():
            return None

        self.assertIsNone(calculation(legacy_diagnostic=True))

    def test_diagnostic_fields_preserve_values_without_mutating_input(self):
        row = dict(release_speed_kmh=110., plate_speed_kmh=101., speed_change_kmh=.6,
                   target_error_cm=50., target_error_reduction_cm=2., human_trial_ready=True)
        original = dict(records=[row], human_trial_ready=True, prediction_available=True)
        result = diagnostic_result(original)
        self.assertEqual(row['release_speed_kmh'], 110.)
        self.assertTrue(original['human_trial_ready'])
        for key in row:
            if key != 'human_trial_ready':
                self.assertNotIn(key, result['records'][0])
        self.assertEqual(result['records'][0]['diagnostic_assumed_point_speed_kmh'], 110.)
        self.assertEqual(result['records'][0]['diagnostic_plate_speed_kmh'], 101.)
        self.assertEqual(result['records'][0]['diagnostic_speed_difference_kmh'], .6)
        self.assertEqual(result['records'][0]['diagnostic_target_error_cm'], 50.)
        self.assertEqual(result['records'][0]['diagnostic_target_difference_cm'], 2.)
        self.assertFalse(result['human_trial_ready'])
        self.assertFalse(result['records'][0]['human_trial_ready'])
        self.assertFalse(result['prediction_available'])

    def test_missing_and_excluded_results_are_not_zero_filled(self):
        result = diagnostic_result({'records': [{'status': 'excluded', 'reason': 'ground'}]})
        row = result['records'][0]
        self.assertEqual(row['reason'], 'ground')
        self.assertNotIn('diagnostic_speed_difference_kmh', row)
        self.assertIsNone(row['predicted_ball_speed_kmh'])

    def test_build_is_blocked_before_read_or_output_creation(self):
        from pitch_outcomes import build_states
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'not_created'
            with self.assertRaises(RetiredPredictionModelError):
                build_states(Path(temp)/'absent.npz', output)
            self.assertFalse(output.exists())

    def test_gui_is_blocked_before_creating_a_window(self):
        from pitch_outcomes_gui import launch
        with patch('pitch_outcomes_gui.tk.Tk', side_effect=AssertionError('window created')):
            with self.assertRaises(RetiredPredictionModelError):
                launch({}, Path('unused'))

    def test_search_apis_reject_implicit_use(self):
        for name in SEARCHES:
            with self.subTest(module=name), self.assertRaises(RetiredPredictionModelError):
                importlib.import_module(name).run()

    def test_all_default_cli_entry_points_are_blocked(self):
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        for name in ('pitch_outcomes',) + SEARCHES:
            with self.subTest(module=name):
                run = subprocess.run([sys.executable, str(ROOT/f'{name}.py')],
                                     cwd=ROOT, env=env, capture_output=True,
                                     encoding='utf-8', errors='replace', timeout=45)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn('--legacy-diagnostic', run.stderr)
                self.assertIn('error:', run.stderr)

    def test_cached_states_cannot_bypass_default_cli_guard(self):
        from pitch_outcomes import main
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)/'states.json'
            cache.write_text(json.dumps({'prediction_available': True,
                                         'states': [example_state()]}), encoding='utf-8')
            output = Path(temp)/'not_created'
            argv = ['pitch_outcomes.py', '--states', str(cache), '--output', str(output), '--gui']
            with patch.object(sys, 'argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_explicit_cache_replay_exports_only_diagnostic_names(self):
        from pitch_outcomes import main
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)/'states.json'
            cache.write_text(json.dumps({'prediction_available': True,
                                         'states': [example_state()]}), encoding='utf-8')
            output = Path(temp)/'diagnostics'
            argv = ['pitch_outcomes.py', '--states', str(cache), '--output', str(output),
                    '--legacy-diagnostic']
            with patch.object(sys, 'argv', argv), patch('builtins.print'):
                main()
            result = json.loads((output/'legacy_diagnostic_results.json').read_text(encoding='utf-8'))
            self.assertFalse((output/'results.json').exists())
            self.assertFalse(result['prediction_available'])
            self.assertFalse(result['provenance']['prediction_available'])
            row = result['records'][0]
            self.assertEqual(row['status'], 'ok')
            self.assertAlmostEqual(row['diagnostic_assumed_point_speed_kmh'], 126.)
            self.assertNotIn('release_speed_kmh', row)
            self.assertIsNone(row['predicted_speed_gain_kmh'])

    @unittest.skipUnless(os.name == 'nt', 'Windows launcher')
    def test_default_windows_launcher_does_not_start_the_model(self):
        run = subprocess.run(['cmd', '/c', str(ROOT/'start_outcome_simulator.cmd')],
                             cwd=ROOT, capture_output=True, timeout=10)
        self.assertEqual(run.returncode, 2)
        self.assertIn(b'coach_hypotheses/model_retirement/README.md', run.stdout)

    def test_explicit_gui_replay_labels_and_exports_diagnostics(self):
        import tkinter as tk
        from tkinter import ttk
        from pitch_outcomes_gui import launch
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as temp:
                data = {'states': [example_state(), example_state('same')]}
                with patch('pitch_outcomes_gui.tk.Tk', return_value=root), patch.object(root, 'mainloop'):
                    launch(data, Path(temp), legacy_diagnostic=True)
                root.update_idletasks()
                self.assertIn('과거 진단', root.title())
                tree = next(w for w in root.winfo_children() if isinstance(w, ttk.Treeview))
                self.assertEqual(tree.heading('speed')['text'], '가정점 속력 km/h')
                self.assertEqual(tree.heading('gain')['text'], '진단 속력차 km/h')
                self.assertEqual(len(tree.get_children()), 2)
                result = json.loads((Path(temp)/'legacy_diagnostic_results.json').read_text(encoding='utf-8'))
                self.assertFalse(result['prediction_available'])
                self.assertIsNone(result['records'][1]['predicted_speed_gain_kmh'])
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
