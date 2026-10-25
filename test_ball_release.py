import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from ball_release import hand_frame, rigid_offset_velocity


class BallReleaseTests(unittest.TestCase):
    def test_frame_handedness_and_rotation_equivariance(self):
        rotation = Rotation.from_euler('xyz', [.2, -.7, 1.1]).as_matrix()
        fin = np.array([[1., 0, 0]]) @ rotation.T
        wra = np.array([[0., 1, 0]]) @ rotation.T
        wrb = -wra
        origin, frames = hand_frame(fin, wra, wrb)
        np.testing.assert_allclose(origin, fin)
        np.testing.assert_allclose(frames[0], rotation, atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(frames[0]), 1.)

    def test_translation_and_rotational_offset_velocity(self):
        rate = 10000.
        time = np.arange(1001)/rate
        omega = 3.
        frames = Rotation.from_euler('z', (omega*time)[:, None]).as_matrix()
        origin = time[:, None]*np.array([2., 3., -1.])
        positions, velocities = rigid_offset_velocity(origin, frames, [.1, 0, 0], rate)
        expected_offset = .1*np.column_stack([np.cos(omega*time), np.sin(omega*time), np.zeros(len(time))])
        expected_velocity = np.array([2., 3., -1.])+np.cross([0, 0, omega], expected_offset)
        np.testing.assert_allclose(positions, origin+expected_offset, atol=1e-12)
        np.testing.assert_allclose(velocities, expected_velocity, atol=1e-8)

    def test_degenerate_frame_rejected(self):
        with self.assertRaises(ValueError):
            hand_frame([[0, 0, 0]], [[1, 0, 0]], [[-1, 0, 0]])

    def test_zero_offset_is_origin_velocity(self):
        origin = np.arange(5)[:, None]*np.array([[1., -2., 3.]])
        _, velocity = rigid_offset_velocity(origin, np.tile(np.eye(3), (5, 1, 1)), [0., 0, 0], 10.)
        np.testing.assert_allclose(velocity, np.tile([10., -20., 30.], (5, 1)))


if __name__ == '__main__':
    unittest.main()
