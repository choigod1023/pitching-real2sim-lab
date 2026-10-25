import unittest
import numpy as np
from ball_holdout_audit import project

class HoldoutProjectionTests(unittest.TestCase):
    def test_known_pinhole_translation(self):
        k=np.array([[500,0,320],[0,600,240],[0,0,1.]])
        p=np.column_stack([np.eye(3),[10,20,0]])
        np.testing.assert_allclose(project([90,80,1000],k,p),[370,300])
    def test_invalid_depth_rejected(self):
        for point in [[0,0,0],[1,2,-3],[float('nan'),0,1]]:
            with self.assertRaises(ValueError):
                project(point,np.eye(3),np.column_stack([np.eye(3),np.zeros(3)]))

if __name__=='__main__':unittest.main()
