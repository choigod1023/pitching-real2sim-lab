import unittest
import numpy as np
from pixel_launch_validation import fit,trajectory,project

class PixelLaunchTests(unittest.TestCase):
    def setUp(self):
        k=np.diag([800.,800.,1.]);self.cameras=[(k,np.eye(3),np.zeros(3)),(k,np.eye(3),np.array([1.,0,0]))]
        self.g=np.array([0.,0.,-9.81]);self.t=np.linspace(-.03,0,8)
        self.x=np.array([.2,.1,5.,12.,3.,-2.]);self.p=trajectory(self.x,self.t,self.g)
        self.uv=np.array([project(self.p,c) for c in self.cameras])
    def test_known_state(self):
        for mode in ('G','P','R'):
            np.testing.assert_allclose(fit(self.t,self.p,self.uv,self.cameras,self.g,mode),self.x,atol=1e-7)
    def test_pixel_fit_corrects_biased_initializer(self):
        biased=self.p+np.array([.01,.005,.02])
        np.testing.assert_allclose(fit(self.t,biased,self.uv,self.cameras,self.g,'P'),self.x,atol=1e-7)
    def test_training_only(self):
        allp=np.vstack([self.p,[[1,2,3]]]);a=fit(self.t,allp[:8],self.uv,self.cameras,self.g,'P')
        allp[8]=999999
        np.testing.assert_array_equal(a,fit(self.t,allp[:8],self.uv,self.cameras,self.g,'P'))
    def test_invalid(self):
        with self.assertRaises(ValueError):fit(self.t[::-1],self.p,self.uv,self.cameras,self.g,'P')
        with self.assertRaises(ValueError):project([[0,0,-1]],self.cameras[0])

if __name__=='__main__':unittest.main()
