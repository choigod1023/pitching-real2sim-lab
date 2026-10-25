import unittest
import numpy as np
from ball_transition import attachment_load, free_flight, MASS_RANGE_KG, RADIUS_RANGE_M


class BallTransitionTests(unittest.TestCase):
    def test_release_preserves_velocity_and_matches_ballistics(self):
        result = free_flight([0., 0., 1.8], [40., 2., 3.], .145, .037)
        self.assertEqual(result['release_velocity_jump_mps'], 0.)
        self.assertLess(result['analytical_position_error_m'], 1e-10)
        self.assertAlmostEqual(result['initial_speed_kmh'], np.linalg.norm([40, 2, 3])*3.6)

    def test_mass_changes_energy_but_not_prescribed_release_speed(self):
        a, b = [free_flight([0., 0., 1.8], [40., 0., 0.], m, RADIUS_RANGE_M.mean()) for m in MASS_RANGE_KG]
        np.testing.assert_allclose(a['final_position_m'], b['final_position_m'], atol=1e-12)
        self.assertAlmostEqual(b['initial_kinetic_energy_j']/a['initial_kinetic_energy_j'], 1.05)

    def test_attachment_force_balances_gravity_and_acceleration(self):
        t = np.arange(101)/1000
        x = np.column_stack([t*t, np.zeros(101), np.ones(101)])
        force = attachment_load(x, 1000, .145)
        np.testing.assert_allclose(force[2:-2], np.tile([.29, 0., .145*9.81], (97, 1)), atol=1e-10)

    def test_invalid_release_rejected(self):
        with self.assertRaises(ValueError):
            free_flight([0, 0, 1], [np.nan, 0, 0], .145, .037)
