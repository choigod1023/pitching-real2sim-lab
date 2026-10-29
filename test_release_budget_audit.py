import unittest

import mujoco
import numpy as np

from release_budget_audit import energy_work_audit, terminal_discrepancy, trapezoid_window


class ReleaseBudgetTests(unittest.TestCase):
    def test_energy_is_not_mass_times_relative_speed_squared(self):
        r = terminal_discrepancy([30., 0, 0], 36.*3.6)
        self.assertAlmostEqual(r['minimum_terminal_momentum_difference_ns'], .145*6.)
        self.assertAlmostEqual(r['terminal_ball_translational_energy_difference_j'], .5*.145*(36.**2-30.**2))
        self.assertNotAlmostEqual(r['terminal_ball_translational_energy_difference_j'], .5*.145*6.**2)
        self.assertIsNone(r['corrected_speed_kmh'])
        self.assertIsNone(r['actual_contact_force_n'])

    def test_unknown_direction_is_lower_bound_not_vector_error(self):
        r = terminal_discrepancy([30., 0, 0], 36.*3.6)
        self.assertEqual(r['minimum_velocity_vector_difference_mps'], 6.)
        self.assertGreater(np.linalg.norm(np.array([0., 36., 0])-np.array([30., 0, 0])), 6.)
        self.assertIsNone(r['actual_release_velocity_error_kmh'])

    def test_record_slower_has_negative_energy_difference(self):
        r = terminal_discrepancy([36., 0, 0], 30.*3.6)
        self.assertLess(r['terminal_ball_translational_energy_difference_j'], 0)
        self.assertGreater(r['minimum_terminal_momentum_difference_ns'], 0)

    def test_reject_missing_and_invalid_state(self):
        for v, speed, mass in [([np.nan, 0, 0], 130, .145), ([1, 2], 130, .145), ([1, 2, 3], np.inf, .145), ([1, 2, 3], 130, 0)]:
            with self.assertRaises(ValueError):
                terminal_discrepancy(v, speed, mass)

    def test_off_grid_window_integrates_constant_and_linear_power(self):
        t = np.linspace(0., .020, 7)
        power = np.column_stack([np.ones(len(t))*12., 100*t])
        integral = trapezoid_window(t, power, .012, .020)
        np.testing.assert_allclose(integral, [12*.008, 50*(.020**2-.012**2)], atol=1e-14)
        with self.assertRaises(ValueError):
            trapezoid_window(t, power, -.001, .020)

    def slider(self, gravity):
        return mujoco.MjModel.from_xml_string(f'''<mujoco>
            <option gravity="0 0 {gravity}"/><worldbody><body name="ball_payload">
            <joint type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".0364" mass=".145"/><site name="ball_center"/>
            </body></worldbody></mujoco>''')

    def test_actual_mujoco_work_energy_closes_for_accelerated_slider(self):
        t = np.linspace(0., .02, 41); acc = 120.
        q = (.5*acc*t**2)[:, None]; v = (acc*t)[:, None]; f = np.full_like(q, .145*acc)
        original_q, original_v, original_f = q.copy(), v.copy(), f.copy()
        result = energy_work_audit(self.slider(0), q, v, f, t)
        for window in result['windows'].values():
            self.assertAlmostEqual(window['net_work_minus_energy_change_j'], 0., places=12)
        np.testing.assert_array_equal(q, original_q)
        np.testing.assert_array_equal(v, original_v)
        np.testing.assert_array_equal(f, original_f)

    def test_gravitational_fall_changes_ke_without_external_work(self):
        t = np.linspace(0., .02, 41)
        q = (1.-.5*9.81*t**2)[:, None]; v = (-9.81*t)[:, None]
        result = energy_work_audit(self.slider(-9.81), q, v, np.zeros_like(v), t)
        self.assertGreater(result['ball_translational_ke_at_release_j'], 0.)
        self.assertAlmostEqual(result['windows']['whole_saved_throw']['mechanical_energy_change_j'], 0., places=12)
        self.assertEqual(result['windows']['whole_saved_throw']['generalized_net_work_j'], 0.)


if __name__ == '__main__':
    unittest.main()
