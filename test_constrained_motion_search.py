import unittest
import numpy as np
from coupled_payload import model_for
from constrained_motion_search import modify, budget_vector, select_resolved_candidate

class MotionSearchTests(unittest.TestCase):
    def setUp(self):
        self.dt=.001;self.m=model_for(np.array([.3,.25]),self.dt)
        self.q=np.tile(self.m.qpos0,(101,1))
        t=np.arange(101)*self.dt
        self.q[:,7]=.2+np.tanh((t-.05)/.015)*.3
    def test_initial_state_and_translation_preserved(self):
        changed=modify(self.m,self.q,self.dt,np.array([.5,-.5,.5,-.5,.5,.5]))
        np.testing.assert_array_equal(changed[:4],self.q[:4])
        np.testing.assert_array_equal(changed[:,:3],self.q[:,:3])
        self.assertFalse(np.array_equal(changed,self.q))
    def test_zero_is_exact_identity(self):
        np.testing.assert_array_equal(modify(self.m,self.q,self.dt,np.zeros(6)),self.q)
    def test_numerically_unresolved_gain_falls_back(self):
        def row(index,sub,speed,gain):
            return dict(evaluation_id=index,substeps=sub,speed_kmh=speed,speed_gain_kmh=gain,all_pass=True)
        rows=[row(0,1,100.,0.),row(0,2,100.01,0.),row(1,1,100.005,.005),row(1,2,100.015,.005)]
        self.assertEqual(select_resolved_candidate(rows)['evaluation_id'],0)
        rows[2].update(speed_kmh=100.20,speed_gain_kmh=.20)
        rows[3].update(speed_kmh=100.21,speed_gain_kmh=.20)
        self.assertEqual(select_resolved_candidate(rows)['evaluation_id'],1)
    def test_bounds_rejected(self):
        with self.assertRaises(ValueError):modify(self.m,self.q,self.dt,np.ones(6)*2)
        with self.assertRaises(ValueError):modify(self.m,self.q,self.dt,np.zeros(5))

if __name__=='__main__':unittest.main()
