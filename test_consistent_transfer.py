import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from consistent_transfer import fit_consistent_chain
from frozen_transfer import fit_chain, chain_state


class ConsistentTransferTests(unittest.TestCase):
    def test_rotation_observation_noise_against_known_future(self):
        t = np.arange(4) * .01
        clean = Rotation.from_rotvec(t[:, None] * [0, 0, 5]).as_matrix()
        # Zero intercept/slope perturbation in rotation observations.
        angles = 5*t + [0, .04, -.02, 0]
        observed = Rotation.from_rotvec(np.c_[0*t, 0*t, angles]).as_matrix()
        origin = np.zeros((4, 3))
        ball = np.einsum('nij,nj->ni', clean, [.1, .2, 0] + t[:, None]*[1, 0, 0])
        before = chain_state(fit_chain(t, ball, origin, observed), .05)[0]
        after = chain_state(fit_consistent_chain(t, ball, origin, observed), .05)[0]
        expected = Rotation.from_rotvec([0, 0, .25]).apply([.15, .2, 0])
        self.assertGreater(np.linalg.norm(before-expected), 1e-4)
        np.testing.assert_allclose(after, expected, atol=1e-12)

    def test_translation_noise_does_not_leak_into_relative_velocity(self):
        t = np.arange(4) * .01
        rotation = np.tile(np.eye(3), (4, 1, 1))
        origin = np.c_[t * 2, [.0, .003, -.004, .002], t * 0]
        ball = np.c_[t * 10 + .1, t * 0 + .2, t * 0 + .3]
        model = fit_consistent_chain(t, ball, origin, rotation)
        np.testing.assert_allclose(chain_state(model, .05)[1], [10, 0, 0], atol=1e-12)

    def test_known_rotating_motion_and_analytic_derivative(self):
        t = np.arange(4) * .01
        rotation = Rotation.from_rotvec(t[:, None] * [0, 0, 5]).as_matrix()
        origin = np.c_[2*t, 0*t, 0*t]
        relative = [.1, .2, 0] + t[:, None] * [1, 0, 0]
        ball = origin + np.einsum('nij,nj->ni', rotation, relative)
        model = fit_consistent_chain(t, ball, origin, rotation)
        p, v = chain_state(model, .05)
        expected_r = Rotation.from_rotvec([0, 0, .25]).as_matrix()
        expected_v = [2, 0, 0] + expected_r @ (np.cross([0, 0, 5], [.15, .2, 0]) + [1, 0, 0])
        np.testing.assert_allclose(v, expected_v, atol=1e-10)
        eps = 1e-6
        numeric = (chain_state(model, .05+eps)[0]-chain_state(model, .05-eps)[0])/(2*eps)
        np.testing.assert_allclose(v, numeric, atol=1e-9)
        ball[:] = 999; origin[:] = 999; rotation[:] = 0
        np.testing.assert_allclose(chain_state(model, .05)[0], p)


if __name__ == '__main__':
    unittest.main()
