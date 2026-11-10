import unittest

import numpy as np

from experiments.skenes_fastball_scenarios.test_study import pitch
from experiments.skenes_finger_release.study import (
    INERTIA, MASS, NOMINAL, RPM, contact_pulse, impulse_fraction, simulate,
)


class ContactPulseTests(unittest.TestCase):
    def test_pulse_integral(self):
        phase = np.linspace(0, 1, 2001)
        cdf, area = impulse_fraction(phase)
        np.testing.assert_allclose(cdf[[0, 1000, -1]], [0, .5, 1])
        np.testing.assert_allclose(area[[0, -1]], [0, .5])
        self.assertTrue(np.all(np.diff(cdf) >= 0))

    def test_control_preserves_translation_and_integrates_torque(self):
        p, v = np.array([0., 0., 1.7]), np.array([0., 45., 0.])
        reference = contact_pulse(p, v, 2300., 180., 0., .1, .35, .85)
        candidate = contact_pulse(p, v, 2300., 180., 4., .1, .35, .85)
        self.assertEqual(candidate["status"], "computed")
        self.assertAlmostEqual(reference["rpm"], 2300.)
        np.testing.assert_allclose(reference["center_xyz_m"], candidate["center_xyz_m"])
        np.testing.assert_allclose(np.asarray(candidate["center_xyz_m"])[-1], p, atol=1e-12)
        np.testing.assert_allclose(np.asarray(candidate["velocity_m_s"])[-1], v)
        np.testing.assert_allclose(np.subtract(*np.asarray(candidate["velocity_m_s"])[[-1,0]]), [0., .1/MASS, 0.])
        final, initial = np.asarray(candidate["omega_rad_s"])[[-1,0]]
        np.testing.assert_allclose(INERTIA*(final-initial), candidate["torque_impulse_nms"])
        self.assertAlmostEqual(candidate["rotational_work_j"],
                               np.dot((final+initial)/2, candidate["torque_impulse_nms"]))
        self.assertGreater(candidate["rpm"], reference["rpm"])
        self.assertGreater(candidate["rotational_work_j"], reference["rotational_work_j"])

    def test_friction_violation_has_no_speed_result(self):
        row = contact_pulse(np.array([0.,0.,1.7]), np.array([0.,45.,0.]), 2300., 180., 4., .1, .25, .85)
        self.assertEqual(row["status"], "friction_rejected")
        self.assertNotIn("rpm",row)
        self.assertGreater(row["mu_required"], row["mu_limit"])

    def test_positive_spin_changes_upward_lift_at_identical_release_speed(self):
        baseline = simulate(pitch(), 0., *NOMINAL)
        candidate = simulate(pitch(), 4., *NOMINAL)
        self.assertEqual(candidate["release_speed_kmh"], baseline["release_speed_kmh"])
        self.assertGreater(candidate["lift_xz_cm"][1], baseline["lift_xz_cm"][1])
        self.assertAlmostEqual(candidate["active_rpm"] - baseline["active_rpm"],
                               (candidate["torque_impulse_nms"][0] - baseline["torque_impulse_nms"][0]) / INERTIA * RPM,
                               delta=.3)
