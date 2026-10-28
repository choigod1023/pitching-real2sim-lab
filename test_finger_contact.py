import unittest
import numpy as np
from finger_contact import ContactParams, contact, derivative, rk4_step, totals


class ContactTests(unittest.TestCase):
    def state(self, gap=-.001):
        s = np.zeros((2, 9))
        s[0, 0] = .0444+gap
        return s

    def test_unilateral_and_friction_bound(self):
        p = ContactParams()
        for gap in (-.001, 0., .001):
            for vx in (-10., 0., 10.):
                s = self.state(gap); s[0, 3:6] = [vx, 100, 0]
                c = contact(s, p)
                self.assertGreaterEqual(c['normal_force_n'], 0.)
                self.assertLessEqual(np.linalg.norm(c['tangent_force_n']), p.friction*c['normal_force_n']+1e-12)
                self.assertGreaterEqual(c['dissipation_w'], -1e-12)
                if gap >= 0:
                    np.testing.assert_array_equal(c['force_on_ball'], np.zeros(3))

    def test_ballistic_and_gravity_energy(self):
        p = ContactParams(gravity=(0, 0, -9.81))
        s = self.state(.1); s[0, 3:6] = [1, 2, 3]
        first = s.copy(); e0 = totals(s, p)['energy_j']; dt = .001
        for _ in range(100):
            s = rk4_step(s, dt, p)
        t = .1
        np.testing.assert_allclose(s[:, :3], first[:, :3]+first[:, 3:6]*t+.5*np.array(p.gravity)*t*t, atol=1e-13)
        self.assertAlmostEqual(totals(s, p)['energy_j'], e0, places=12)

    def test_elastic_collision_and_step_convergence(self):
        p = ContactParams(damping=0, friction=0)
        initial = self.state(.0005); initial[0, 3] = -1.
        e0 = totals(initial, p)['energy_j']; results = []
        for dt in (1e-4, 5e-5, 2.5e-5):
            s = initial.copy()
            for _ in range(round(.008/dt)):
                s = rk4_step(s, dt, p)
            results.append(s)
            np.testing.assert_allclose(totals(s, p)['linear_momentum'], totals(initial, p)['linear_momentum'], atol=1e-13)
        exact = (p.masses[0]-p.masses[1])/sum(p.masses)*-1
        self.assertLess(abs(results[-1][0, 3]-exact), 2e-4)
        self.assertLess(abs(totals(results[-1], p)['energy_j']-e0), 2e-5)
        self.assertLess(np.linalg.norm(results[-1]-results[-2]), np.linalg.norm(results[-2]-results[-3]))

    def test_oblique_reaction_momenta_and_dissipation(self):
        p = ContactParams()
        s = self.state(-.0001); s[0, 3:6] = [-1, .8, .2]; s[1, 6:9] = [0, 2, -3]
        first = totals(s, p); previous = first['energy_j']
        for _ in range(400):
            s = rk4_step(s, 2e-5, p)
            current = totals(s, p)
            self.assertLessEqual(current['energy_j'], previous+1e-7)
            previous = current['energy_j']
        np.testing.assert_allclose(current['linear_momentum'], first['linear_momentum'], atol=1e-12)
        np.testing.assert_allclose(current['angular_momentum'], first['angular_momentum'], atol=1e-10)
        self.assertLess(current['energy_j'], first['energy_j'])

    def test_spin_participates_and_invalid_inputs(self):
        s = self.state(); p = ContactParams(); s[0, 8] = 10
        self.assertGreater(np.linalg.norm(contact(s, p)['tangent_force_n']), 0.)
        self.assertGreater(np.linalg.norm(derivative(s, p)[:, 6:9]), 0.)
        for kwargs in ({'masses': (0, .1)}, {'friction': -1}, {'gravity': (0, np.nan, 0)}):
            with self.assertRaises(ValueError): ContactParams(**kwargs)
        with self.assertRaises(ValueError): rk4_step(s, 0, p)
        with self.assertRaises(ValueError): contact(np.zeros((2, 9)), p)


if __name__ == '__main__':
    unittest.main()
