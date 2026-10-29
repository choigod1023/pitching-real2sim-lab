import unittest
import numpy as np
from coach_hypotheses.measurement_evidence.audit_public_sample import event_index, fixed_window_speed, valid_points


class AuditTests(unittest.TestCase):
    def test_vicon_one_based_event_mapping(self):
        self.assertEqual(event_index(1.29499996, 200, 1), 258)
        self.assertEqual(event_index(1.29499996, 200, 251), 8)

    def test_event_off_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            event_index(1.291, 200, 1)

    def test_finite_coordinates_with_invalid_residual_are_missing(self):
        xyz = np.array([[0, 1, np.nan], [0, 1, 1], [0, 1, 1]])
        np.testing.assert_array_equal(valid_points(xyz, np.array([0, -1, 0])), [True, False, False])

    def test_fixed_window_speed_and_translation_invariance(self):
        xyz = np.array([2, -3, 6])[:, None] * np.arange(10)[None, :] / 200
        v = np.ones(10, dtype=bool)
        self.assertAlmostEqual(fixed_window_speed(xyz, v, 2, 200), 7.0)
        self.assertAlmostEqual(fixed_window_speed(xyz + 100, v, 2, 200), 7.0)

    def test_missing_sample_is_not_bridged(self):
        xyz = np.zeros((3, 10)); v = np.ones(10, dtype=bool); v[5] = False
        self.assertIsNone(fixed_window_speed(xyz, v, 2, 200))
        self.assertIsNone(fixed_window_speed(xyz, np.ones(10, dtype=bool), 5, 200))


if __name__ == "__main__":
    unittest.main()
