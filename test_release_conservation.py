import unittest
import numpy as np
from coupled_payload import fit_pose
from release_conservation import detach, passive_check


class DetachmentTests(unittest.TestCase):
    def test_rotating_translating_state_conserves_full_system(self):
        from scipy.spatial.transform import Rotation
        points = np.array([[[.2,-.1,1.5], [.2,-.1,1.2], [.4,-.1,1.05]]])
        frame = Rotation.from_euler('xyz', [30,40,70], degrees=True).as_matrix()[None]
        q, _ = fit_pose(points, np.array([[.3,-.05,1.125]]), frame)
        velocity = np.array([3.,2.,1., 2.,-3.,4., 5.,-2., 4.,3.,-6.])
        m, d, report = detach([.3,.25], q[0], velocity, .0001)
        for key in ['mass_jump_kg','energy_jump_j','linear_momentum_jump_kg_mps',
                    'angular_momentum_jump_kg_m2ps','position_jump_m','velocity_jump_mps']:
            self.assertLess(abs(report[key]), 1e-10, key)
        after = passive_check(m,d,.01)
        self.assertLess(after['ball_ballistic_error_m'],1e-10)
        self.assertLess(after['gravity_corrected_linear_residual_kg_mps'],1e-7)
        self.assertLess(abs(after['mechanical_energy_drift_j']),1e-6)
