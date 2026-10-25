import unittest
import numpy as np
from joint_video import aligned_index, elbow_angle

class JointVideoTests(unittest.TestCase):
    def test_alignment_bounds(self):
        self.assertEqual(aligned_index(780,1,1406),781)
        with self.assertRaises(ValueError): aligned_index(1405,1,1406)
        with self.assertRaises(ValueError): aligned_index(0,-1,1406)

    def test_elbow_geometry(self):
        p={'r_uarm':np.array([0.,1,0]),'r_larm':np.zeros(3),'r_hand':np.array([1.,0,0])}
        self.assertAlmostEqual(elbow_angle(p),90.)
        p['r_hand']=np.array([0.,-1,0])
        self.assertAlmostEqual(elbow_angle(p),180.)
        p['r_hand']=np.zeros(3)
        with self.assertRaises(ValueError): elbow_angle(p)
