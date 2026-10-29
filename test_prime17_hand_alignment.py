import unittest
import numpy as np
from prime17_hand_alignment import box_residual, fit_grid


class SyntheticProjection:
    """Known2D clock mapping for a code check, not experimental evidence."""
    def residuals(self, rows, offset, delta, margin=0.):
        return np.array([box_residual([offset+delta[0]*100, 0.], r['box'], margin) for r in rows])

    def summary(self, rows, offset, delta, margin=0.):
        d = np.linalg.norm(self.residuals(rows, offset, delta, margin), axis=1)
        return {'inside': int((d <= 1e-9).sum()), 'mean_outside_px': float(d.mean())}


class RegionCalibrationTests(unittest.TestCase):
    def test_inside_is_zero_for_different_points_not_point_accuracy(self):
        np.testing.assert_equal(box_residual([1, 2], [0, 0, 10, 10]), [0, 0])
        np.testing.assert_equal(box_residual([9, 8], [0, 0, 10, 10]), [0, 0])

    def test_signed_outside_components(self):
        np.testing.assert_equal(box_residual([-2, 13], [0, 0, 10, 10]), [-2, 3])

    def test_boundary_expansion_and_invalid_inputs(self):
        np.testing.assert_equal(box_residual([-2, 13], [0, 0, 10, 10], 3), [0, 0])
        for point, box in [([np.nan, 1], [0, 0, 2, 2]), ([1, 2], [2, 2, 0, 0])]:
            with self.assertRaises(ValueError):
                box_residual(point, box)

    def test_known_clock_recovered_when_synthetic_point_observed(self):
        rows = [{'box': [1.75, 0, 1.75, 0]}]
        fitted, _ = fit_grid(SyntheticProjection(), rows, False)
        self.assertEqual(fitted['offset_frames'], 1.75)

    def test_ambiguous_regions_keep_nominal_clock_instead_of_arbitrary_gain(self):
        rows = [{'box': [.5, -1, 2.25, 1]}]
        fitted, states = fit_grid(SyntheticProjection(), rows, False)
        self.assertEqual(fitted['offset_frames'], 1.)
        self.assertEqual(sum(c['training']['inside'] == 1 for c in states), 8)


if __name__ == '__main__':
    unittest.main()
