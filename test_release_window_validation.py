import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from release_window_validation import interpolate,fit_similarity,fit_models
from fixed_alignment_audit import similarity
from third_view_validation import cameras
from ball_trajectory_fit import triangulate


class AlignmentTests(unittest.TestCase):
    def test_fractional_samples_and_bounds(self):
        x=np.arange(15.).reshape(5,3)
        np.testing.assert_allclose(interpolate(x,[0.,1.5,4.]),[x[0],(x[1]+x[2])/2,x[4]])
        for bad in [[-0.1],[4.1],[np.nan]]:
            with self.assertRaises(ValueError):interpolate(x,bad)

    def test_similarity_proper_rotation_scale_and_translation(self):
        x=np.random.default_rng(3).normal(size=(20,3))*1000
        truth=dict(scale=.0011,rotation=Rotation.from_rotvec([.2,-.3,.1]).as_matrix().tolist(),translation=[1.,2.,3.])
        a=fit_similarity(x,similarity(x,truth))
        np.testing.assert_allclose(similarity(x,a),similarity(x,truth),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(a['rotation']),1.)
        with self.assertRaises(ValueError):fit_similarity(np.zeros((4,3)),np.ones((4,3)))

    def test_training_only_recovers_fractional_delay(self):
        rng=np.random.default_rng(31)
        t=np.arange(60.)[:,None,None]
        anchors=rng.normal(size=(1,5,3))*400
        # Different accelerations across landmarks make time/translation separable.
        x=anchors+np.sin(t*.13+rng.normal(size=(1,5,3)))*80+t*t*rng.normal(size=(1,5,3))*.07
        baseline=dict(scale=.001,rotation=np.eye(3).tolist(),translation=[0.,0.,4.])
        frames=np.arange(10.,30.)
        target=similarity(interpolate(x,frames+.625),baseline)
        models,_=fit_models(x,frames,target,baseline)
        self.assertAlmostEqual(models['M1']['offset'],.625,places=5)
        self.assertAlmostEqual(models['M3']['offset'],.625,places=4)
        # Altering observations outside the supplied fitting rows cannot enter the fit.
        np.testing.assert_allclose(similarity(interpolate(x,frames+models['M3']['offset']),models['M3']['alignment']),target,atol=1e-7)

    def test_time_spatial_degeneracy_is_visible_in_profile(self):
        t=np.arange(30.)[:,None,None]
        anchors=np.random.default_rng(4).normal(size=(1,5,3))
        x=anchors+t*np.array([[[.1,.2,.3]]])
        a=dict(scale=1.,rotation=np.eye(3).tolist(),translation=[0.,0.,0.])
        f=np.arange(5.,20.)
        _,profile=fit_models(x,f,interpolate(x,f+.5),a)
        self.assertLess(max(p['refit_mse_m2'] for p in profile),1e-20)

    def test_third_camera_projection_preserves_mm_and_positive_depth(self):
        intr={f'optitrack_{i}':dict(fx_px=800.,fy_px=820.,cx_px=640.,cy_px=360.,distortion=[0,0,0,0,0]) for i in [17,20,21]}
        rig={str(i):dict(reference_to_camera_rotation_matrix=np.eye(3).tolist(),reference_to_camera_translation_mm=t) for i,t in [(17,[0,0,0]),(20,[-300,0,0]),(21,[0,-200,0])]}
        cs=cameras(intr,rig,[20,21,17]);point=np.array([120.,40.,3000.,1.]);uv=[]
        for k,p in cs:
            q=k@p@point;uv.append(q[:2]/q[2])
        x,res,depth=triangulate(uv,cs)
        np.testing.assert_allclose(x,point[:3],atol=1e-7)
        self.assertLess(max(res),1e-7);self.assertGreater(min(depth),0)

    def test_distorted_camera_is_not_silently_treated_as_pinhole(self):
        with self.assertRaises(ValueError):
            cameras({'optitrack_17':dict(distortion=[.1,0,0,0,0])},{'17':{}},[17])


if __name__=='__main__':unittest.main()
