import unittest
import mujoco
import numpy as np
from torque_arm import arm_poses,make_model,run_dynamics


class TorqueArmTests(unittest.TestCase):
    def test_marker_directions_reconstructed(self):
        shoulder=np.array([[.1,.2,1.5]])
        elbow=shoulder+np.array([[0.,0.,-.3]])
        wrist=elbow+np.array([[.2,0.,-.15]])
        q=arm_poses(shoulder,elbow,wrist)
        m=make_model([.3,.25],.001);d=mujoco.MjData(m);d.qpos[:]=q[0]
        mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.xpos[m.body('forearm').id],elbow[0],atol=1e-12)
        np.testing.assert_allclose(d.site_xpos[m.site('wrist').id],wrist[0],atol=1e-12)

    def test_static_inverse_force_balances_gravity(self):
        m=make_model([.3,.25],.001)
        q=np.tile(np.array([0.,0.,1.5,1.,0.,0.,0.,1.]),(101,1))
        result,arrays=run_dynamics(m,q,.001)
        self.assertLess(result['wrist_max_error_mm'],1e-7)
        self.assertAlmostEqual(arrays['required_generalized_forces'][0,2],(2.14154+1.280182)*9.81,places=9)
        self.assertEqual(result['state_resets_after_initialization'],0)

    def test_straight_elbow_twist_not_identifiable(self):
        s=np.array([[0.,0.,0.]])
        with self.assertRaises(ValueError):arm_poses(s,s+[0,0,-1],s+[0,0,-2])

    def test_no_force_control_falls_instead_of_replaying(self):
        m=make_model([.3,.25],.001)
        q=np.tile(np.array([0.,0.,1.5,1.,0.,0.,0.,1.]),(101,1))
        result,arrays=run_dynamics(m,q,.001,force_scale=0.)
        # Semi-implicit Euler free fall after 100 steps, with no state reset.
        expected_drop=9.81*.001**2*100*101/2
        self.assertAlmostEqual(q[0,2]-arrays['integrated_qpos'][-1,2],expected_drop,places=10)
        self.assertGreater(result['wrist_max_error_mm'],49)
