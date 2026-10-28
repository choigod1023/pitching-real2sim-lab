import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from render_joint_detail import relative_angle

class DetailTests(unittest.TestCase):
    def test_relative_rotation_and_quaternion_sign(self):
        q=Rotation.from_euler('z',np.array([[30.],[60.],[90.]]),degrees=True).as_quat()[:,[3,0,1,2]]
        np.testing.assert_allclose(relative_angle(q),[0,30,60],atol=1e-12)
        q[1]*=-1;np.testing.assert_allclose(relative_angle(q),[0,30,60],atol=1e-12)
    def test_invalid(self):
        with self.assertRaises(ValueError):relative_angle(np.zeros((2,3)))

if __name__=='__main__':unittest.main()
