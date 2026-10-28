import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from relative_ball_state import ball_velocity,local_offset


class RelativeBallTests(unittest.TestCase):
    def test_rotating_moving_frame_matches_position_derivative(self):
        # Independent central difference of a world trajectory with local sliding.
        t=.02;eps=1e-6;omega=np.array([0.,0.,13.]);v=np.array([20.,3.,1.])
        r0=np.array([.15,.01,0.]);rdot=np.array([1.2,-.3,.1])
        rot=lambda s:Rotation.from_rotvec(omega*s).as_matrix()
        position=lambda s:v*s+rot(s)@(r0+rdot*s)
        expected=(position(t+eps)-position(t-eps))/(2*eps)
        actual=ball_velocity(v,rot(t),omega,r0+rdot*t,rdot)
        np.testing.assert_allclose(actual['ball_velocity_mps'],expected,atol=1e-8)
        self.assertGreater(np.linalg.norm(actual['rigid_velocity_mps']-expected)*3.6,1.)
        np.testing.assert_allclose(local_offset(v*t,rot(t),position(t)),r0+rdot*t)

    def test_rigid_limit_and_common_translation(self):
        r=Rotation.from_euler('xyz',[.3,.1,-.5]).as_matrix()
        a=ball_velocity([3,4,5],r,[10,0,3],[.15,0,0],[0,0,0])
        np.testing.assert_array_equal(a['ball_velocity_mps'],a['rigid_velocity_mps'])
        b=ball_velocity([5,7,9],r,[10,0,3],[.15,0,0],[0,0,0])
        np.testing.assert_allclose(b['ball_velocity_mps']-a['ball_velocity_mps'],[2,3,4])

    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):local_offset([0,0,0],np.diag([1,1,-1]),[1,0,0])


if __name__=='__main__':unittest.main()
