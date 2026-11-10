import unittest

import numpy as np

from experiments.trackman_v3_flight import study
from pitch_outcomes import flight, flight_acceleration


def release(**changes):
    return dict(RelSpeed=90, SpinRate=2300, SpinAxis=180, RelHeight=6, RelSide=2,
                Extension=6, VertRelAngle=-2, HorzRelAngle=-3) | changes


class TrackmanFlightTests(unittest.TestCase):
    def test_batched_acceleration_matches_single_pitch_acceleration(self):
        v = np.array([[0., 39., -1.], [1., 40., 0.]])
        axes = np.array([[1., 0., 0.], [0., 0., -1.]])
        cls = np.array([.2, .18])
        batched = flight_acceleration(v, axes, cls[:, None], .35)
        separately = [flight_acceleration(u, a, cl, .35) for u, a, cl in zip(v, axes, cls)]
        np.testing.assert_allclose(batched, separately, atol=1e-12)

    def test_distance_rk4_agrees_with_existing_time_rk4(self):
        pitches = [release(), release(SpinAxis=223)]
        config = study.VARIANTS["both"]
        xz, velocity, distance, axes, cls = study.release_inputs(pitches, *config)
        ivb, _, final, trace, ground = study.simulate(pitches, config, keep=(0, 1))
        self.assertFalse(ground.any())
        self.assertEqual(trace.shape, (257, 2, 6))
        for i in range(len(pitches)):
            path, time = flight([xz[i, 0], 0, xz[i, 1]], velocity[i], axes[i], cls[i],
                                .35, distance[i], .0005)
            np.testing.assert_allclose(final[i, :2], path[-1, [0, 2]], atol=2e-6)
            self.assertAlmostEqual(final[i, 5], time, places=6)
        self.assertGreater(ivb[0], ivb[1])

    def test_distinguishes_unimpeded_ground_crossing(self):
        _, _, _, _, ground = study.simulate([release(VertRelAngle=-18)], study.VARIANTS["both"])
        self.assertTrue(ground[0])

    def test_reference_polynomial_ends_at_front_and_converts_lateral_sign(self):
        row = {f"PitchTrajectory{axis}c{i}": coefficient
               for axis, vector in zip("XYZ", ((55, -130, 14), (5, -4, -9), (-3, 5, -3)))
               for i, coefficient in enumerate(vector)}
        reference, time = study.reference_trajectory(row)
        self.assertAlmostEqual(reference[0, 0], 3 * study.FT)
        self.assertAlmostEqual(reference[0, 1], (study.PLATE_FT - 55) * study.FT)
        self.assertAlmostEqual(reference[-1, 1], 0)
        self.assertAlmostEqual(reference[-1, 2], (5 - 4 * time - 9 * time * time) * study.FT)

    def test_tie_does_not_match_nonzero_change_and_pair_mae_is_in_cm(self):
        values = np.array([0., 0., -1.])
        observed = np.array([2., 3., 1.])
        result = study.metrics(values, observed, np.array([[0, 1], [1, 2]]))
        self.assertEqual((result["correct"], result["ties"]), (1, 1))
        self.assertAlmostEqual(result["pair_delta_mae_cm"], 1.)
