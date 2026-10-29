import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from joint_origin_audit import relative_origin,fixed_local_prediction
from arm_root_replay import replay


def matrices(p,r=None):
    p=np.asarray(p,float);t=np.tile(np.eye(4),(len(p),1,1));t[:,:3,3]=p
    if r is not None:t[:,:3,:3]=r
    return t


class SourceGraphTests(unittest.TestCase):
    def test_rotated_parent_local_origin_is_correct(self):
        r=Rotation.from_euler('z',[[90.],[-60.]],degrees=True).as_matrix();p=np.array([[1,2,3],[4,5,6.]])
        local=np.array([.1,.2,-.3]);parent=matrices(p,r);child=matrices(p+np.einsum('nij,j->ni',r,local))
        np.testing.assert_allclose(relative_origin(parent,child),np.tile(local,(2,1)),atol=1e-14)

    def test_fixed_attachment_uses_only_training_mean(self):
        parent=matrices([[0,0,0],[0,0,0]])
        pred,center=fixed_local_prediction(parent,[[1,2,3],[1,2,3]])
        np.testing.assert_array_equal(pred,[[1,2,3],[1,2,3]])
        np.testing.assert_array_equal(center,[1,2,3])

    def test_free_root_replay_and_fixed_root_are_different_assumptions(self):
        shoulder=np.array([[0,0,1],[.1,0,1],[.2,0,1]])
        t={'torso':matrices([[0,0,1]]*3),'l_uarm':matrices(shoulder),
           'l_larm':matrices(shoulder+[0,0,-.3]),'l_hand':matrices(shoulder+[0,0,-.55])}
        free=replay(t,[0],False);fixed=replay(t,[0],True)
        np.testing.assert_allclose(free[:,2],shoulder+[0,0,-.55])
        self.assertAlmostEqual(np.linalg.norm(free[-1,2]-fixed[-1,2]),.2)
        np.testing.assert_allclose(np.linalg.norm(free[:,2]-free[:,1],axis=1),.25)
        np.testing.assert_allclose(np.linalg.norm(fixed[:,2]-fixed[:,1],axis=1),.25)

    def test_bad_transform_rejected(self):
        a=matrices([[0,0,0]]);a[0,0,0]=2
        with self.assertRaises(ValueError):relative_origin(a,a)


if __name__=='__main__':unittest.main()
