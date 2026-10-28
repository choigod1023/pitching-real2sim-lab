import unittest
import numpy as np
from curl_geometry import path
from grip_bench import DIR,RADIUS,PAD_RADIUS,DEPTH,SMOOTH
from friction_contact import contact_force


class CurlTests(unittest.TestCase):
    def test_balanced_preload_and_radius(self):
        for angle in [-10,0,10]:
            dirs,rate=path(.012,DIR,np.array([0,1.,0]),angle)
            np.testing.assert_allclose(np.linalg.norm(dirs,axis=1),1,atol=1e-12)
            np.testing.assert_allclose(dirs.sum(axis=0),0,atol=1e-12)
            np.testing.assert_allclose(np.sum(dirs*rate,axis=1),0,atol=1e-12)
            self.assertAlmostEqual(3*.5*(10000/3)*DEPTH**2,.005)

    def test_velocity_matches_curved_position(self):
        t=.007;h=1e-7;axis=np.array([0,1.,0]);dirs,v=path(t,DIR,axis,-10)
        numeric=(path(t+h,DIR,axis,-10)[0]-path(t-h,DIR,axis,-10)[0])/(2*h)
        np.testing.assert_allclose(v,numeric,atol=1e-7)
        for t in [0,.020,.03]:np.testing.assert_allclose(path(t,DIR,axis,10)[1],0)

    def test_curl_actuator_power_is_in_energy_balance(self):
        t=.007;x=np.array([.0001,-.0002,.0001]);v=np.array([.01,.02,-.01]);w=np.array([1.,2.,-1.]);axis=np.array([0,1.,0]);radius=RADIUS+PAD_RADIUS-DEPTH
        def evaluate(t,x):
            dirs,rate=path(t,DIR,axis,10);energy=power=loss=ball_power=0.
            for a,ad in zip(dirs,rate):
                delta=x-radius*a;n=delta/np.linalg.norm(delta);penetration=RADIUS+PAD_RADIUS-np.linalg.norm(delta)
                offset=-RADIUS*n;vp=radius*ad;vb=v+np.cross(w,offset)
                f,e,l=contact_force(n,penetration,vb-vp,10000/3,2/3,.6,SMOOTH)
                energy+=e;loss+=l;power+=f@vp;ball_power+=f@vb
            return energy,loss,power,ball_power
        e,l,p,b=evaluate(t,x);h=1e-7;edot=(evaluate(t+h,x+h*v)[0]-evaluate(t-h,x-h*v)[0])/(2*h)
        self.assertAlmostEqual(b+edot,p-l,places=7)


if __name__=='__main__':unittest.main()
