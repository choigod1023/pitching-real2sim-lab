import unittest

import numpy as np

from experiments.rapsodo_flight_attribution import study


PITCH = {"speed_mph": 78., "rpm": 1893., "release_height_ft": 5.3,
         "horizontal_angle_deg": 0., "vertical_angle_deg": 0.}


class FlightAttributionTests(unittest.TestCase):
    def test_published_lift_table_and_range(self):
        self.assertAlmostEqual(study.lift_coefficient(.175, "lyu_four_seam"), .19)
        self.assertAlmostEqual(study.lift_coefficient(.175, "sawicki"), .195)
        with self.assertRaises(ValueError):
            study.lift_coefficient(.21, "lyu_four_seam")

    def test_crossing_height_and_reference_identity(self):
        mode = study.run_flight(PITCH, .80, 18.)
        speed = PITCH["speed_mph"] * .44704
        vacuum = PITCH["release_height_ft"] * .3048 - .5 * 9.80665 * (16.8 / speed)**2
        self.assertAlmostEqual(mode["gravity_only_height_cm"], 100 * vacuum)
        self.assertAlmostEqual(mode["path_m"][-1, 1], 16.8)
        self.assertAlmostEqual(mode["plate_height_cm"] - mode["no_lift_height_cm"], mode["ivb_cm"])
        self.assertAlmostEqual(mode["plate_height_cm"] - mode["gravity_only_height_cm"], mode["ivb_gravity_only_cm"])

    def test_plane_changes_only_where_the_path_stops(self):
        near = study.run_flight(PITCH, .80, 18., study.DISTANCE_M - study.PLATE_DEPTH_M)
        far = study.run_flight(PITCH, .80, 18., study.DISTANCE_M + study.PLATE_DEPTH_M)
        np.testing.assert_array_equal(near["path_m"][:-1], far["path_m"][:len(near["path_m"]) - 1])
        self.assertLess(near["ivb_cm"], far["ivb_cm"])
        self.assertGreater(near["plate_height_cm"], far["plate_height_cm"])

    def test_direction_changes_magnus_at_fixed_efficiency_and_cl(self):
        backspin = study.run_flight(PITCH, .80, 0.)
        sidespin = study.run_flight(PITCH, .80, 90.)
        self.assertEqual(backspin["cl"], sidespin["cl"])
        self.assertGreater(backspin["ivb_cm"], sidespin["ivb_cm"])
        self.assertLess(backspin["horizontal_break_cm"], sidespin["horizontal_break_cm"])

    def test_marginal_contributions_sum_in_presence_of_interaction(self):
        factorial = {(i, j, k): {"ivb_cm": i + 2*j + 4*k + 2*i*j, "plate_height_cm": j - 3*k}
                     for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        attributed = study.averaged_contributions(factorial)
        self.assertEqual([attributed[name]["ivb_cm"] for name in study.FACTOR_NAMES], [2., 3., 4.])
        for key in study.FIELDS:
            self.assertAlmostEqual(sum(values[key] for values in attributed.values()),
                                   factorial[1, 1, 1][key] - factorial[0, 0, 0][key])


if __name__ == "__main__":
    unittest.main()
