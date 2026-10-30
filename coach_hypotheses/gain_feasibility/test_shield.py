import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from capacity_budget_gate import rollout
from coupled_payload import model_for
from .shield import actual_accounting, replay_controls, run_experiment, shield_rollout


class ShieldTests(unittest.TestCase):
    def setUp(self):
        self.lengths = np.array([.3, .25])
        self.dt = .001
        self.model = model_for(self.lengths, self.dt)
        self.q = np.tile(self.model.qpos0, (13, 1))
        self.controls = np.full((len(self.q), self.model.nv), 20.)

    def test_inactive_matches_existing_rk4_and_saved_replay_exactly(self):
        for substeps in (1, 2):
            expected, pos, vel = rollout(self.lengths, self.q, self.controls, self.dt, substeps)
            result, actual_pos, actual_vel, trace = shield_rollout(self.lengths, self.q, self.controls, self.dt, substeps)
            np.testing.assert_array_equal(actual_pos[::substeps], pos)
            np.testing.assert_array_equal(actual_vel[::substeps], vel)
            for key in ('speed_kmh', 'positive_work_per_dof_j', 'negative_work_magnitude_per_dof_j', 'net_work_per_dof_j'):
                np.testing.assert_array_equal(result[key], expected[key])
            self.assertEqual(result['virtual_retry_count'], 0)
            rp, rv = replay_controls(self.lengths, actual_pos[0], actual_vel[0], trace['applied_loads'], trace['dt'])
            np.testing.assert_array_equal(rp, actual_pos)
            np.testing.assert_array_equal(rv, actual_vel)

    def test_baseline_own_total_budgets_do_not_activate_shield(self):
        for substeps in (1, 2):
            baseline, pos, vel = rollout(self.lengths, self.q, self.controls, self.dt, substeps)
            result, actual_pos, actual_vel, trace = shield_rollout(self.lengths, self.q, self.controls, self.dt, substeps, baseline)
            self.assertEqual(result['suppressed_substeps'], 0)
            np.testing.assert_array_equal(actual_pos[::substeps], pos)
            np.testing.assert_array_equal(actual_vel[::substeps], vel)
            np.testing.assert_array_equal(trace['scale_per_substep'], 1.)

    def test_rejected_trials_restore_exact_zero_force_trajectory_and_gravity(self):
        budgets = {'positive_work_per_dof_j': np.zeros(self.model.nv),
                   'negative_work_magnitude_per_dof_j': np.zeros(self.model.nv)}
        for substeps in (1, 2):
            _, zero_pos, zero_vel = rollout(self.lengths, self.q, np.zeros_like(self.controls), self.dt, substeps)
            result, pos, vel, trace = shield_rollout(self.lengths, self.q, self.controls, self.dt, substeps, budgets)
            self.assertEqual(result['zero_force_substeps'], (len(self.q)-1)*substeps)
            np.testing.assert_array_equal(trace['virtual_retries_per_substep'], 41)
            np.testing.assert_array_equal(pos[::substeps], zero_pos)
            np.testing.assert_array_equal(vel[::substeps], zero_vel)
            self.assertLess(vel[-1, 2], 0.)
            self.assertEqual(result['total_positive_component_work_j'], 0.)
            self.assertEqual(result['total_negative_component_work_magnitude_j'], 0.)

    def test_cumulative_budgets_and_input_direction_preserved_every_substep(self):
        caps = np.full(self.model.nv, 1e-5)
        budgets = {'positive_work_per_dof_j': caps, 'negative_work_magnitude_per_dof_j': caps}
        result, pos, vel, trace = shield_rollout(self.lengths, self.q, self.controls, self.dt, 2, budgets)
        self.assertGreater(result['virtual_retry_count'], 0)
        self.assertTrue(np.all(np.asarray(result['positive_work_per_dof_j']) <= caps))
        self.assertTrue(np.all(np.asarray(result['negative_work_magnitude_per_dof_j']) <= caps))
        np.testing.assert_array_equal(trace['applied_loads'], np.repeat(self.controls[:-1], 2, axis=0)*trace['scale_per_substep'][:, None])
        self.assertTrue(np.all((trace['scale_per_substep'] >= 0) & (trace['scale_per_substep'] <= 1)))
        np.testing.assert_array_equal(trace['scale_by_source_interval'].reshape(-1), trace['scale_per_substep'])
        rp, rv = replay_controls(self.lengths, pos[0], vel[0], trace['applied_loads'], trace['dt'])
        np.testing.assert_array_equal(rp, pos)
        np.testing.assert_array_equal(rv, vel)
        # Positive/negative ledgers are monotone; each saved prefix stays in budget.
        for stop in range(2, len(pos)+1):
            prefix = actual_accounting(self.lengths, pos[:stop], vel[:stop], trace['applied_loads'][:stop-1], trace['dt'])
            self.assertTrue(np.all(np.asarray(prefix['positive_work_per_dof_j']) <= caps))
            self.assertTrue(np.all(np.asarray(prefix['negative_work_magnitude_per_dof_j']) <= caps))

    def test_actual_accounting_uses_accepted_forces_and_terminal_is_unused(self):
        controls = self.controls.copy()
        controls[-1] = 1e10
        result, pos, vel, trace = shield_rollout(self.lengths, self.q, controls, self.dt, 2)
        forces = trace['applied_loads']
        expected_net = np.sum(forces*(vel[:-1]+vel[1:])*trace['dt']/2, axis=0)
        np.testing.assert_allclose(result['net_work_per_dof_j'], expected_net, atol=1e-15, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(result['positive_work_per_dof_j'])-result['negative_work_magnitude_per_dof_j'], expected_net, atol=1e-15, rtol=1e-14)
        np.testing.assert_array_equal(result['peak_abs_generalized_force_per_dof'], 20.)
        independent = actual_accounting(self.lengths, pos, vel, forces, trace['dt'])
        self.assertEqual(result['speed_kmh'], independent['speed_kmh'])
        with self.assertRaises(ValueError):
            actual_accounting(self.lengths, pos, vel, forces[:-1], trace['dt'])

    def test_invalid_budget_and_source_hash_fail_closed(self):
        with self.assertRaises(ValueError):
            shield_rollout(self.lengths, self.q, self.controls, self.dt, budgets={
                'positive_work_per_dof_j': [-1]*self.model.nv,
                'negative_work_magnitude_per_dof_j': [0]*self.model.nv})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'source').write_bytes(b'wrong-source')
            (root/'report').write_text(json.dumps({'source_sha256': 'wrong'}))
            with patch('coach_hypotheses.gain_feasibility.shield.rollout') as run:
                with self.assertRaises(ValueError):
                    run_experiment(root/'source', root/'report', root/'out')
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
