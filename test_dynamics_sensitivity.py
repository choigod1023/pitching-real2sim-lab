import unittest
import numpy as np
from dynamics_sensitivity import demand
from torque_arm import make_model


class WorkTests(unittest.TestCase):
    def test_stationary_arm_requires_force_but_zero_work(self):
        q = np.tile([0., 0., 1.5, 1., 0., 0., 0., 1.], (101, 1))
        result, forces, _, _ = demand(make_model([.3, .25], .001), q, .001)
        self.assertGreater(forces[0, 2], 30)
        self.assertEqual(result['work']['total']['positive_j'], 0.)
        self.assertAlmostEqual(result['energy_change_j'], 0.)

    def test_constant_upward_speed_work_matches_potential_energy(self):
        q = np.tile([0., 0., 1.5, 1., 0., 0., 0., 1.], (101, 1))
        q[:, 2] += np.arange(101)*.001*2.
        result, _, _, _ = demand(make_model([.3, .25], .001), q, .001)
        expected = (2.14154+1.280182)*9.81*.2
        self.assertAlmostEqual(result['work']['total']['net_j'], expected, places=8)
        self.assertAlmostEqual(result['energy_change_j'], expected, places=8)

    def test_uniform_mass_scaling_scales_required_loads(self):
        t = np.arange(101)*.001
        q = np.tile([0., 0., 1.5, 1., 0., 0., 0., 1.], (101, 1))
        q[:, 0] = t**2
        q[:, 7] += .1*np.sin(10*t)
        _, base, _, _ = demand(make_model([.3, .25], .001), q, .001)
        _, scaled, _, _ = demand(make_model([.3, .25], .001,
                                  masses=np.array([2.14154, 1.280182])*1.2), q, .001)
        np.testing.assert_allclose(scaled, 1.2*base, atol=1e-10)
