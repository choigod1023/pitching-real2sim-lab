import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.seam_axis_flight.rotation import TextureAgreement, fit_axis
from experiments.seam_axis_flight.study import optical_inputs


class SeamAxisFlightTests(unittest.TestCase):
    def test_known_rotation_recovers_orientation_and_beats_static_texture(self):
        grid = np.linspace(-1., 1., 55)
        x, y = np.meshgrid(grid, grid)
        z = np.sqrt(np.maximum(0., 1. - x*x - y*y))
        xyz = np.stack((x, -y, z), axis=-1)
        axis = np.array([.7, -.2, -.6])
        axis /= np.linalg.norm(axis)
        rate = .39
        frames = []
        for i in range(15):
            coordinates = Rotation.from_rotvec(-i * rate * axis).apply(xyz.reshape(-1, 3))
            u, v, w = coordinates.T
            texture = np.sin(17*u + 3*v + w) + .7*np.cos(12*v - 3*w) + .4*np.sin(11*w - u)
            frames.append(texture.reshape(x.shape))
        agreement = TextureAgreement(np.stack(frames), 0, 11, .60)
        recovered, score = fit_axis(agreement, rate)
        error = np.rad2deg(np.arccos(np.clip(np.dot(recovered, axis), -1, 1)))
        self.assertLess(error, 2.)
        self.assertGreater(score, agreement.score(np.zeros(3)) + .4)

    def test_rear_camera_transforms_to_right_handed_flight_coordinates(self):
        basis = ([0., 1., 0.], [1., 0., 0.], [0., 0., 1.])
        field, transverse, active, tilt = optical_inputs([.8, 0., -.6], basis)
        np.testing.assert_allclose(field, [.8, .6, 0.])
        np.testing.assert_allclose(transverse, [1., 0., 0.])
        self.assertAlmostEqual(active, .8)
        self.assertAlmostEqual(tilt, 0.)
        with self.assertRaises(ValueError):
            optical_inputs([0., 0., 1.], basis)
