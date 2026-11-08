import unittest
import numpy as np
from edge_calibration import fit_camera,project,triangulate,register_cube,metric_screen

class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.c={'K':np.array([[1200.,0,640],[0,1200,512],[0,0,1]]),'dist':np.array([-.1,.01,0.,0,0]),'R':np.eye(3),'t':np.zeros(3)}
        self.d=dict(self.c,t=np.array([-500.,0,0]))
    def test_distorted_multiview_roundtrip(self):
        x=np.array([125.,230.,3500.]);uv=[project([x],c)[0] for c in [self.c,self.d]]
        y,e,z=triangulate(uv,[self.c,self.d]);np.testing.assert_allclose(y,x,atol=1e-5);self.assertLess(np.linalg.norm(e),1e-7);self.assertTrue(min(z)>0)
    def test_cube_pnp_roundtrip(self):
        x=np.array([[a,b,c+3000.] for a in [0,698.5] for b in [0,698.5] for c in [0,698.5]])
        uv=project(x,self.c);fit=fit_camera(x,uv,self.c['K'],self.c['dist']);np.testing.assert_allclose(project(x,fit),uv,atol=1e-5)
    def test_reject_planar_cube_and_unmatched_views(self):
        with self.assertRaises(ValueError):fit_camera(np.zeros((8,3)),np.zeros((8,2)),self.c['K'],self.c['dist'])
        with self.assertRaises(ValueError):triangulate([[1,2]],[self.c,self.d])
    def test_cube_registration_preserves_projection(self):
        c=dict(self.c,t=np.array([0.,0,3000.]));x=np.array([0.,698.5,698.5]);s=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);centre=np.array([349.25,349.25,0]);world=s@(x-centre)+centre
        np.testing.assert_allclose(project([x],c),project([world],register_cube(c,1,698.5)),atol=1e-8)
    def test_metric_check_fails_closed(self):
        ref={'a':200.,'b':150.};self.assertTrue(metric_screen(ref,ref,.5,1000));self.assertFalse(metric_screen({'a':204.,'b':150},ref,.1,1000));self.assertFalse(metric_screen({'a':np.nan,'b':150},ref,.1,1000));self.assertFalse(metric_screen(ref,ref,.1,-1))
    def test_zero_reprojection_does_not_establish_metric_scale(self):
        x=np.array([0.,0,3000.]);y=x+np.array([200.,0,0]);uvx=[project([x],c)[0] for c in [self.c,self.d]];uvy=[project([y],c)[0] for c in [self.c,self.d]]
        wrong=dict(self.d,t=self.d['t']*.95);xx,e,_=triangulate(uvx,[self.c,wrong]);yy,f,_=triangulate(uvy,[self.c,wrong]);self.assertLess(np.linalg.norm(e)+np.linalg.norm(f),1e-7);self.assertAlmostEqual(np.linalg.norm(xx-yy),190.,places=5);self.assertFalse(metric_screen({'length':190.},{'length':200.},0,1000))
if __name__=='__main__':unittest.main()
