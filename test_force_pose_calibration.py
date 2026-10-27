import unittest
import numpy as np
import mujoco
from scipy.interpolate import BSpline
from force_pose_calibration import corrected

class ForcePoseTests(unittest.TestCase):
    def test_zero_correction_and_rotation_normalization(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        q=np.tile(m.qpos0,(20,1));b=np.ones((20,1))
        actual,_=corrected(m,q,b,np.zeros(6));np.testing.assert_array_equal(q,actual)
        actual,delta=corrected(m,q,b,np.array([.001,.002,.003,.01,0,0]))
        np.testing.assert_allclose(actual[:,:3],delta[:,:3])
        np.testing.assert_allclose(np.linalg.norm(actual[:,3:7],axis=1),1.)
    def test_cubic_basis_preserves_component_bounds(self):
        t=np.linspace(0,1,1000);knots=np.r_[np.zeros(4),np.linspace(0,1,6)[1:-1],np.ones(4)]
        b=BSpline.design_matrix(t,knots,3).toarray()
        np.testing.assert_allclose(b.sum(1),1.)
        self.assertGreaterEqual(b.min(),0.)
        x=np.random.default_rng(42).uniform(-1,1,(8,6))
        self.assertLessEqual(np.abs(b@x).max(),1.)

if __name__=='__main__':unittest.main()
