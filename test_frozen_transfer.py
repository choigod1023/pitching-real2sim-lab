import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from frozen_transfer import fit_chain,chain_state


class ChainTests(unittest.TestCase):
    def test_training_linear_motion_exact(self):
        t=np.arange(4)*.01;r=np.tile(np.eye(3),(4,1,1));o=np.c_[t*2,t*0,t*0]
        ball=o+[.1,.2,.3]+t[:,None]*[3,0,0]
        m=fit_chain(t,ball,o,r);p,v=chain_state(m,.04)
        np.testing.assert_allclose(p,[.3,.2,.3],atol=1e-12);np.testing.assert_allclose(v,[5,0,0],atol=1e-12)

    def test_rotating_offset_derivative(self):
        t=np.arange(4)*.01;r=Rotation.from_rotvec(t[:,None]*[0,0,5]).as_matrix();o=np.zeros((4,3));ball=np.einsum('nij,j->ni',r,[.1,0,0])
        m=fit_chain(t,ball,o,r);p,v=chain_state(m,.03);eps=1e-6
        numeric=(chain_state(m,.03+eps)[0]-chain_state(m,.03-eps)[0])/(2*eps)
        np.testing.assert_allclose(v,numeric,atol=1e-9)
        np.testing.assert_allclose(v,np.cross([0,0,5],p),atol=1e-10)

    def test_future_arrays_not_required(self):
        t=np.arange(3)*.01;r=np.tile(np.eye(3),(3,1,1));o=np.zeros((3,3));ball=np.c_[t*10,t*0,t*0]
        m=fit_chain(t,ball,o,r);ball[:]=999;o[:]=999;r[:]=0
        np.testing.assert_allclose(chain_state(m,.05)[1],[10,0,0],atol=1e-10)

    def test_invalid_time_and_rotation(self):
        with self.assertRaises(ValueError):fit_chain([0,0,1],np.zeros((3,3)),np.zeros((3,3)),np.tile(np.eye(3),(3,1,1)))
        with self.assertRaises(ValueError):fit_chain([0,1,2],np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3,3)))


if __name__=='__main__':unittest.main()
