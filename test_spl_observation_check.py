import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from spl_observation_check import frame,rigid_point,ballistic


class ObservationTests(unittest.TestCase):
    def test_rigid_mapping_is_rotation_equivariant(self):
        w=np.array([[1.,2.,3.]]);thumb=w+[[1.,-1.,0.]];pinky=w+[[1.,1.,0.]]
        r=frame(w,thumb,pinky)[0];local=np.array([.1,.2,.3])
        q=Rotation.from_rotvec([.2,.3,.1]).as_matrix()
        rq=frame(w@q.T,thumb@q.T,pinky@q.T)[0]
        np.testing.assert_allclose(rigid_point(w[0]@q.T,rq,local),q@rigid_point(w[0],r,local),atol=1e-12)

    def test_two_ball_gravity_prediction_exact_for_projectile(self):
        p=np.array([1.,2.,3.]);v=np.array([2.,3.,5.]);g=np.array([0.,0.,-9.81]);dt=1/60;h=3/60
        previous=p-v*dt+.5*g*dt**2
        np.testing.assert_allclose(ballistic(previous,p,dt,h),p+v*h+.5*g*h*h,atol=1e-12)

    def test_degenerate_hand_rejected(self):
        with self.assertRaises(ValueError):frame(np.zeros((1,3)),np.ones((1,3)),np.ones((1,3)))
