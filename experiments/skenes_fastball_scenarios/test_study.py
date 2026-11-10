import unittest
import numpy as np

from experiments.skenes_fastball_scenarios.study import lift_model, release_state, scenario, slot_offset


def pitch(angle=24., spin=2200., vx=10.):
    return {"arm_angle_deg": angle, "row": {
        "release_pos_x": -2.5, "release_pos_y": 54., "release_pos_z": 6.,
        "vx0": vx, "vy0": -145., "vz0": -7., "ax": -20., "ay": 35., "az": -17.,
        "release_speed": 100., "release_spin_rate": spin, "spin_axis": 180.,
    }}


class FlightScenarioTests(unittest.TestCase):
    def test_release_plane_uses_forward_world_axis_and_preserves_recorded_speed(self):
        p, v, distance = release_state(pitch())
        np.testing.assert_allclose(p, [-.762, 0, 1.8288])
        self.assertAlmostEqual(np.linalg.norm(v) * 3.6, 160.9344)
        self.assertGreater(distance, 15.)
        self.assertGreater(v[1], 0.)
        self.assertGreater(v[0] / v[1], 10. / 145.)

    def test_frontal_rotation_moves_release_but_does_not_change_spin_or_launch_velocity(self):
        low, high = pitch(), pitch(22., 2400., 12.)
        np.testing.assert_allclose(slot_offset(24, 24, .65), np.zeros(3))
        base = scenario(low, high, "baseline", .65, .85, .35, .001)
        slot = scenario(low, high, "slot_only", .65, .85, .35, .001)
        self.assertEqual(slot["initial_velocity_m_s"], base["initial_velocity_m_s"])
        self.assertEqual(slot["rpm_input"], base["rpm_input"])
        np.testing.assert_allclose(slot["lift_displacement_xz_cm"], base["lift_displacement_xz_cm"], atol=1e-8)
        self.assertLess(slot["plate_xz_cm"][1], base["plate_xz_cm"][1])

    def test_spin_and_direction_are_separate_controlled_inputs(self):
        low, high = pitch(), pitch(22., 2400., 12.)
        rpm = scenario(low, high, "rpm_only", .65, .85, .35, .001)
        direction = scenario(low, high, "direction_only", .65, .85, .35, .001)
        base = scenario(low, high, "baseline", .65, .85, .35, .001)
        self.assertEqual(rpm["initial_velocity_m_s"], base["initial_velocity_m_s"])
        self.assertEqual(rpm["initial_xyz_m"], direction["initial_xyz_m"])
        self.assertEqual(direction["rpm_input"], base["rpm_input"])
        self.assertAlmostEqual(direction["initial_speed_kmh"], base["initial_speed_kmh"])
        self.assertGreater(rpm["lift_displacement_xz_cm"][1], base["lift_displacement_xz_cm"][1])
        cl, axis = lift_model(0, .85, 180, 44.)
        self.assertEqual(cl, 0)
        self.assertEqual(axis, (1, 0, 0))
