import unittest

import numpy as np

from coordinate_error_budget import derivative_weights, example_budgets, velocity_noise_sigma


class CoordinateErrorBudgetTests(unittest.TestCase):
    def test_constant_velocity_irregular_timestamps(self):
        t = np.array([0.001, 0.0021, 0.0039, 0.0062, 0.007])
        w = derivative_weights(t, 1, 0.004)
        self.assertAlmostEqual(float(w @ (1.5 + 37.0 * t)), 37.0, places=10)
        self.assertAlmostEqual(float(w.sum()), 0.0, places=10)

    def test_acceleration_at_arbitrary_time(self):
        t = np.array([0, 0.001, 0.0022, 0.003, 0.0047, 0.006])
        for e in [0.0, 0.0035, 0.007]:
            w = derivative_weights(t, 2, e)
            x = 0.12 + 30 * t + 0.5 * 1200 * t**2
            self.assertAlmostEqual(float(w @ x), 30 + 1200 * e, places=9)

    def test_monte_carlo_independent_heterogeneous_noise(self):
        t = np.array([0, 0.001, 0.0022, 0.0039, 0.0051, 0.007])
        sigmas = np.array([0.2, 0.4, 0.3, 0.7, 0.5, 0.6]) / 1000
        w = derivative_weights(t, 2, 0.003)
        rng = np.random.default_rng(20261025)
        draws = rng.normal(size=(100000, len(t))) * sigmas
        empirical = np.std(draws @ w, ddof=1)
        predicted = velocity_noise_sigma(t, sigmas, 2, 0.003)
        self.assertLess(abs(empirical / predicted - 1), 0.015)

    def test_two_point_and_eleven_point_closed_forms(self):
        for hz in (360, 1000):
            self.assertAlmostEqual(velocity_noise_sigma([0, 1/hz], 0.0005),
                                   np.sqrt(2) * 0.0005 * hz)
            t = np.arange(-5, 6) / hz
            self.assertAlmostEqual(velocity_noise_sigma(t, 0.0005),
                                   0.0005 * hz / np.sqrt(110))
        self.assertEqual(len(example_budgets()), 4)

    def test_invalid_inputs_and_rank_deficiency(self):
        for t in ([0], [0, 0], [1, 0], [0, np.nan], [[0, 1]]):
            with self.assertRaises(ValueError):
                derivative_weights(t)
        for order in (0, 2, 1.5, True):
            with self.assertRaises(ValueError):
                derivative_weights([0, 1], order)
        with self.assertRaises(ValueError):
            derivative_weights([0, 1], evaluation_time=np.inf)
        with self.assertRaises(ValueError):
            derivative_weights([0, 1e-20, 2e-20, 1], 3)
        for sigma in (-1, np.inf, [1], [1, -1]):
            with self.assertRaises(ValueError):
                velocity_noise_sigma([0, 1], sigma)

    def test_release_crossing_bias_is_not_noise(self):
        # Continuous position/velocity at release, but acceleration switches.
        t = np.arange(-5, 6) / 1000
        position = 35 * t + 0.5 * 2000 * np.minimum(t, 0)**2
        estimate = derivative_weights(t) @ position
        self.assertGreater(abs(estimate - 35), 1)
        self.assertEqual(velocity_noise_sigma(t, 0), 0)


if __name__ == "__main__":
    unittest.main()
