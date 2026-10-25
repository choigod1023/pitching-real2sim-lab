import unittest
import mujoco
import numpy as np
from coupled_payload import fit_pose, model_for, inverse_and_rollout


class PayloadTests(unittest.TestCase):
    def pose(self):
        points = np.array([[[0.,0.,1.5], [0.,0.,1.2], [.2,0.,1.05]]])
        marker = np.array([[.1,.05,1.125]])
        return fit_pose(points, marker, np.eye(3)[None])[0]

    def test_hand_orientation_reconstructed(self):
        m = model_for([.3,.25],.001); d = mujoco.MjData(m); d.qpos[:] = self.pose()[0]
        mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.xmat[m.body('hand').id].reshape(3,3), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(d.site_xpos[m.site('ball_center').id], [.35,0.,1.05], atol=1e-12)

    def test_payload_gravity_reaches_shoulder_and_wrist(self):
        q = np.repeat(self.pose(), 11, axis=0)
        a, fa = inverse_and_rollout(model_for([.3,.25],.001,ball_mass=0.), q,.001)
        b, fb = inverse_and_rollout(model_for([.3,.25],.001), q,.001)
        self.assertAlmostEqual(fb[0,2]-fa[0,2], .145*9.81, places=10)
        self.assertAlmostEqual(np.linalg.norm(fb[0,8:11]-fa[0,8:11]), .15*.145*9.81, places=10)
        self.assertLess(b['ball_point_rmse_mm'],1e-8)

    def test_unobservable_forearm_marker_rejected(self):
        points = np.array([[[0.,0.,1.5], [0.,0.,1.2], [.2,0.,1.05]]])
        with self.assertRaises(ValueError):
            fit_pose(points, np.array([[.1,0.,1.125]]), np.eye(3)[None])
