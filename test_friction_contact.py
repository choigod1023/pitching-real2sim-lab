import unittest
import numpy as np
from friction_contact import contact_force,RADIUS,rollout
from coupled_payload import model_for


class FrictionContactTests(unittest.TestCase):
    def test_no_contact_no_adhesion(self):
        for depth in [-.01,0]:
            f,e,l=contact_force([0,0,1],depth,[0,0,-100],10000,2,.6)
            np.testing.assert_equal(f,0);self.assertEqual(e,0);self.assertEqual(l,0)
        f,_,loss=contact_force([0,0,1],.001,[0,0,100],10000,2,.6)
        np.testing.assert_equal(f,0);self.assertGreater(loss,0)

    def test_friction_cone_energy_and_torque(self):
        n=np.array([0.,0,1]);v=np.array([2.,1.,-.1]);depth=.002;k=10000
        f,e,loss=contact_force(n,depth,v,k,2,.6)
        normal=float(n@f);ft=f-normal*n
        self.assertLessEqual(np.linalg.norm(ft),.6*normal)
        self.assertLess(float(ft@v),0);self.assertGreater(np.linalg.norm(np.cross(-RADIUS*n,f)),0)
        self.assertAlmostEqual(float(f@v)-k*depth*float(n@v),-loss)
        self.assertGreaterEqual(loss,0)

    def test_reaction_momentum_and_no_friction_spin(self):
        p=np.array([.1,.2,.3]);f,_,_=contact_force([0,0,1],.002,[1,0,-.1],10000,2,0)
        np.testing.assert_allclose(np.cross(-RADIUS*np.array([0,0,1]),f),0)
        np.testing.assert_allclose(np.cross(p,f)+np.cross(p,-f),0)

    def test_detached_ball_matches_gravity_without_hidden_contact(self):
        m=model_for([.3,.25],.001,free_ball=True)
        source=dict(dt=.001,lengths=np.array([.3,.25]),qpos=np.tile(m.qpos0[:13],(41,1)),qvel=np.zeros((41,11)),controls=np.zeros((41,11)))
        r,_=rollout(source,no_contact=True,substeps=2)
        self.assertAlmostEqual(r['speed_kmh'],9.81*.03*3.6,places=9)
        self.assertEqual(r['peak_net_force_n'],0)
        self.assertEqual(r['contact_fraction'],0)
        self.assertAlmostEqual(r['energy_balance_residual_j'],0,places=8)


if __name__=='__main__':unittest.main()
