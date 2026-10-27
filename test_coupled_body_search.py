import unittest
import numpy as np
import mujoco
from coupled_body_search import build,modify,inverse,simulate,compare,OUT

class TestCoupledBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m,cls.q,cls.dt,_,_=build()
    def test_zero_change_and_same_initial_state(self):
        np.testing.assert_allclose(modify(self.m,self.q,self.dt,0,0),self.q,atol=1e-14)
        changed=modify(self.m,self.q,self.dt,8,2.37)
        np.testing.assert_allclose(changed[:4],self.q[:4],atol=1e-14)
    def test_true_dynamic_coupling_through_pelvis(self):
        d=mujoco.MjData(self.m);d.qpos[:]=self.q[100];mujoco.mj_forward(self.m,d)
        mass=np.zeros((self.m.nv,self.m.nv));mujoco.mj_fullM(self.m,d,mass)
        inv=np.linalg.inv(mass);k=int(self.m.joint('rear_knee').dofadr[0]);s=int(self.m.joint('pitch_shoulder').dofadr[0])
        self.assertGreater(np.linalg.norm(inv[s:s+3,k]),1e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh(mass)>0))
    def test_retiming_stays_in_observed_domain(self):
        with self.assertRaises(ValueError):modify(self.m,self.q,self.dt,8,0,{'t':[0,.01],'angle':[0,1]})
    def test_shifted_linear_reference_obeys_delay(self):
        t=np.arange(len(self.q))*self.dt;adr=int(self.m.joint('rear_knee').qposadr[0]);q=self.q.copy();q[:,adr]=t
        out=modify(self.m,q,self.dt,8,0,{'t':[-1,1],'angle':[-1,1]})
        np.testing.assert_allclose(out[-20:,adr],q[-20:,adr]-.008,atol=1e-12)
    def test_budget_gate_rejects_added_support(self):
        base={'speed_kmh':100.,'costs':{'external_force':{'peak':10.,'positive_j':1.,'negative_j':2.}}}
        r=compare(self.m,base,base,self.q,self.q);self.assertTrue(r['budget_pass']);self.assertEqual(r['gain_kmh'],0)
        other={'speed_kmh':101.,'costs':{'external_force':{'peak':10.,'positive_j':1.1,'negative_j':2.}}}
        r=compare(self.m,base,other,self.q,self.q);self.assertFalse(r['budget_pass']);self.assertAlmostEqual(r['violations'][0]['excess'],.1)

if __name__=='__main__':unittest.main()
