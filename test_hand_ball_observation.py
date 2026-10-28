import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from hand_ball_observation import relative_position, decompose

class HandBallTests(unittest.TestCase):
    def setUp(self):
        self.q=Rotation.from_euler('z',np.linspace(0,1,8)[:,None]).as_matrix()
        self.o=np.column_stack([np.arange(8)*.01,np.zeros((8,2))])
        self.r=np.tile([.1,.03,-.02],(8,1))

    def test_rigid_point_has_zero_relative_motion(self):
        b=self.o+np.einsum('nij,nj->ni',self.q,self.r)
        r,v,vo,vr,vrel=decompose(b,self.o,self.q,.01)
        np.testing.assert_allclose(r,self.r,atol=1e-15)
        np.testing.assert_allclose(vrel,0,atol=1e-13)
        np.testing.assert_allclose(v,vo+vr+vrel,atol=1e-13)

    def test_global_transform_invariance(self):
        b=self.o+np.einsum('nij,nj->ni',self.q,self.r)
        g=Rotation.from_euler('xyz',[.3,.2,.7]).as_matrix(); t=np.array([3.,-1.,5.])
        changed=relative_position(b@g.T+t,self.o@g.T+t,np.einsum('ij,njk->nik',g,self.q))
        np.testing.assert_allclose(changed,self.r,atol=2e-15)

    def test_moving_relative_point_decomposition(self):
        self.r[:,0]+=np.arange(8)*.005
        b=self.o+np.einsum('nij,nj->ni',self.q,self.r)
        r,v,vo,vr,vrel=decompose(b,self.o,self.q,.01)
        np.testing.assert_allclose(np.diff(r,axis=0)/.01,np.tile([.5,0,0],(7,1)),atol=1e-13)
        np.testing.assert_allclose(v,vo+vr+vrel,atol=1e-13)
        self.assertGreater(np.linalg.norm(vrel),0)

if __name__=='__main__': unittest.main()
