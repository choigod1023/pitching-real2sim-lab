import unittest
import numpy as np
from centered_release import evaluate_fold, fit_centered, center_by_athlete
from release_calibration import predict_speed


class CenteredReleaseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261025)
        self.users = np.repeat(np.arange(4), 5)
        self.v = rng.normal(size=(20, 3)) + np.array([25., 0., 0.])
        self.d = rng.normal(size=(20, 3, 3)) * 30
        self.y = predict_speed(self.v, self.d, np.array([.02, -.01, .03]))

    def test_heldout_targets_cannot_change_fit_or_predictions(self):
        first = evaluate_fold(self.v, self.d, self.y, self.users, 3)
        changed = self.y.copy()
        changed[self.users == 3] += np.arange(5) * 1000
        second = evaluate_fold(self.v, self.d, changed, self.users, 3)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_per_athlete_constant_target_shift_invariance(self):
        first_offset, first_slope = fit_centered(self.v, self.d, self.y, self.users)
        shifted = self.y + np.array([-10., 4., 15., -3.])[self.users]
        second_offset, second_slope = fit_centered(self.v, self.d, shifted, self.users)
        np.testing.assert_allclose(first_offset, second_offset, atol=1e-7)
        self.assertAlmostEqual(first_slope, second_slope, places=12)

    def test_centering_and_known_offset_recovery(self):
        centered = center_by_athlete(self.y, self.users)
        for user in np.unique(self.users):
            self.assertAlmostEqual(centered[self.users == user].mean(), 0., places=12)
        offset, _ = fit_centered(self.v, self.d, self.y, self.users)
        np.testing.assert_allclose(offset, [.02, -.01, .03], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
