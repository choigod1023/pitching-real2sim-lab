import unittest

import numpy as np

from experiments.optical_axis_flight import study


def sample(**changes):
    return dict(release_speed=95., release_spin_rate=2300., spin_axis=215., release_pos_x=-2.,
                release_pos_y=54., release_pos_z=6., vx0=7., vy0=-137., vz0=-6.,
                ax=-8., ay=32., az=-19.) | changes


class OpticalFlightTests(unittest.TestCase):
    def test_axis_is_transverse_and_preserves_azimuth(self):
        velocity = np.array([[2., 40., -1.]])
        tilt = np.deg2rad([35.])
        axis = study.transverse_axis(velocity, tilt)
        np.testing.assert_allclose(np.sum(velocity * axis, axis=1), 0., atol=1e-12)
        np.testing.assert_allclose(np.arctan2(-axis[:, 2], axis[:, 0]), tilt)

    def test_180_degree_optical_axis_matches_backspin(self):
        row = sample(spin_axis=180.)
        baseline = study.simulate([row], False)
        optical = study.simulate([row], True)
        np.testing.assert_allclose(baseline[0], optical[0])
        np.testing.assert_allclose(baseline[1], optical[1])

    def test_targets_and_exported_acceleration_do_not_enter_model(self):
        original = study.simulate([sample()], True)
        changed = study.simulate([sample(ax=0., ay=0., az=0., plate_z=0., pfx_z=0.)], True)
        np.testing.assert_allclose(original[0], changed[0])
        np.testing.assert_allclose(original[1], changed[1])

    def test_reference_begins_at_release_and_finishes_at_front(self):
        path, _ = study.reference_trajectory([sample()])
        np.testing.assert_allclose(path[0, 0, [0, 2]], [2 * study.FT, 6 * study.FT])
        self.assertAlmostEqual(path[0, -1, 1], 0.)

    def test_compare_removes_observed_ties_from_direction_but_not_mae(self):
        metrics = study.compare(np.array([1., 2., 4.]), np.array([0., 0., -1.]),
                                np.array([[0, 1], [1, 2]]))
        self.assertEqual((metrics["correct"], metrics["non_tie_pairs"], metrics["observed_ties"]), (0, 1, 1))
        self.assertAlmostEqual(metrics["pair_delta_mae_cm"], 2.)
