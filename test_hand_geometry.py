import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from hand_geometry import fit_cluster,smooth_cluster,distances


class GeometryTests(unittest.TestCase):
    def test_rigid_motion_recovered_and_smoothed_shape_preserved(self):
        template=np.array([[.08,0,0],[0,.03,0],[0,-.03,0]])
        t=np.arange(100)/360
        r=Rotation.from_rotvec(np.column_stack([0*t,0*t,15*t])).as_matrix()
        origin=np.column_stack([2*t,t,1+t])
        points=origin[:,None,:]+np.einsum('nij,kj->nki',r,template)
        o,rot,fit=fit_cluster(template,points)
        np.testing.assert_allclose(fit,points,atol=1e-12)
        np.testing.assert_allclose(rot,r,atol=1e-12)
        _,sr,sm=smooth_cluster(template,o,rot,360)
        np.testing.assert_allclose(np.linalg.det(sr),1.,atol=1e-12)
        self.assertLess(np.ptp(distances(sm),axis=0).max(),1e-12)

    def test_collinear_cluster_rejected(self):
        p=np.array([[0.,0,0],[1.,0,0],[2.,0,0]])
        with self.assertRaises(ValueError):fit_cluster(p,p[None])

    def test_observed_degenerate_frame_rejected(self):
        template=np.array([[.08,0,0],[0,.03,0],[0,-.03,0]])
        observed=np.array([[[0.,0,0],[1.,0,0],[2.,0,0]]])
        with self.assertRaises(ValueError):fit_cluster(template,observed)
