import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from covariant_release import fit_covariant, fit_forward_regularized


class CovariantReleaseTests(unittest.TestCase):
    def sample(self):
        t = np.arange(4)/360.
        g = np.array([0., 0., -9.81])
        p = np.array([.5, .1, 1.3]); v = np.array([25., -2., 3.])
        ball = p+t[:, None]*v+.5*t[:, None]**2*g
        origin = np.tile([.2, -.1, 1.], (4, 1))
        rotation = np.tile(np.eye(3), (4, 1, 1))
        return t, ball, origin, rotation, g, p, v

    def test_known_ballistic_state_recovered_without_recorded_speed(self):
        t, b, o, r, g, p, v = self.sample()
        pos, vel, detail = fit_covariant(t, b, o, r, g, forced_lambda=1.)
        np.testing.assert_allclose(pos, b[-1], atol=1e-8)
        np.testing.assert_allclose(vel, v+g*t[-1], atol=1e-6)
        self.assertFalse(detail['future_hand_or_ball_input'])

    def test_coordinate_change_preserves_fitted_state(self):
        t, b, o, r, g, _, _ = self.sample()
        p, v, _ = fit_covariant(t, b, o, r, g, forced_lambda=.1)
        q = Rotation.from_rotvec([.2, -.3, .5]).as_matrix(); shift = np.array([.1, .3, -.1])
        pp, vv, _ = fit_covariant(t, b@q.T+shift, o@q.T+shift, q@r, q@g, forced_lambda=.1)
        np.testing.assert_allclose(pp, q@p+shift, atol=1e-7)
        np.testing.assert_allclose(vv, q@v, atol=1e-5)

    def test_forward_validation_recovers_ballistic_prefix(self):
        t, b, o, r, g, _, v = self.sample()
        p, vv, detail = fit_forward_regularized(t, b, o, r, g)
        np.testing.assert_allclose(p, b[-1], atol=1e-8)
        np.testing.assert_allclose(vv, v+g*t[-1], atol=1e-6)
        self.assertFalse(detail['forward_validation_future_hand_used'])
        self.assertLess(max(detail['forward_validation_position_mse_m2']), 1e-12)

    def test_rejects_added_future_frame_and_invalid_rotation(self):
        t, b, o, r, g, _, _ = self.sample()
        with self.assertRaises(ValueError):
            fit_forward_regularized(np.r_[t, .1], np.vstack([b, b[-1]]), np.vstack([o, o[-1]]), np.concatenate([r, r[-1:]]), g)
        r[0, 0, 0] = -1.
        with self.assertRaises(ValueError):
            fit_covariant(t, b, o, r, g)

    def test_inputs_are_not_modified(self):
        t, b, o, r, g, _, _ = self.sample()
        inputs = (t, b, o, r, g); copies = tuple(a.copy() for a in inputs)
        fit_forward_regularized(*inputs)
        for a, original in zip(inputs, copies):
            np.testing.assert_array_equal(a, original)


if __name__ == '__main__':
    unittest.main()
