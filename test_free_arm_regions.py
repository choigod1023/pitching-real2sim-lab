import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from free_arm_regions import minimal_rotation, retarget, unit, observations, KNOTS, Reconstruction


class ReconstructionChecks(unittest.TestCase):
    def test_rotation_parallel_antiparallel_and_general(self):
        for b in [[1,0,0],[-1,0,0],[0,1,1]]:
            r=minimal_rotation([1,0,0],b)
            np.testing.assert_allclose(r@np.array([1,0,0]),unit(b),atol=1e-12)
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r),1.)

    def test_degenerate_vector_rejected(self):
        for x in [[0,0,0],[np.nan,1,0]]:
            with self.assertRaises(ValueError):minimal_rotation(x,[1,0,0])

    def test_length_projection_and_root_motion(self):
        points=np.array([[[0,0,0],[1,0,0],[1,1,0]],[[.2,.3,.4],[1.2,.3,.4],[1.2,1.3,.4]]])
        result=retarget(points,np.ones((2,2,3))*.1,np.array([.02,-.03,.01]),[1.,1.])
        np.testing.assert_allclose(np.linalg.norm(np.diff(result,axis=1),axis=2),1.,atol=1e-12)
        np.testing.assert_allclose(result[1,0]-result[0,0],points[1,0]-points[0,0],atol=1e-12)

    def test_zero_correction_is_identity(self):
        points=np.array([[0,0,0],[1,0,0],[1,1,0]])
        np.testing.assert_allclose(retarget(points,np.zeros((2,3)),np.zeros(3),[1,1]),points,atol=1e-12)

    def test_split_excludes_all_camera17_and_future_frames(self):
        train,test=observations()
        self.assertEqual(len(train),12)
        self.assertTrue(all(r['camera'] in (20,21) and r['frame'] in KNOTS for r in train))
        self.assertEqual(sum(r['camera']==17 for r in test),22)
        keys=lambda rows:{(r['camera'],r['frame'],r['segment']) for r in rows}
        self.assertFalse(keys(train)&keys(test))

    def test_endpoint_hold_and_relative_orientation_transport(self):
        class Fake:
            origins={k:(lambda t,i=i:np.broadcast_to(np.array([[0,0,0],[1,0,0],[1,1,0]])[i],np.asarray(t).shape+(3,))) for i,k in enumerate(['l_uarm','l_larm','l_hand'])}
            hand=staticmethod(lambda t:Rotation.from_rotvec(np.zeros((len(np.atleast_1d(t)),3))))
        displacement=np.tile([[.1,.1,0],[.2,0,.1]],(3,1,1))
        m=Reconstruction(Fake(),[.02,0,0],displacement,[1.,1.])
        np.testing.assert_allclose(m.points(960),m.points(979),atol=1e-12)
        p=m.points([960,979]);r=m.hand([960,979])
        for q,a in zip(r,p):
            np.testing.assert_allclose(q@np.array([0,1,0]),unit(a[2]-a[1]),atol=1e-12)


if __name__=='__main__':unittest.main()
