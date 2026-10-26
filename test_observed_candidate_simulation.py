import unittest
import numpy as np
from observed_candidate_simulation import warp_elbow
from coupled_payload import model_for
from torque_arm import derivatives

class ObservedCandidateTests(unittest.TestCase):
    def trajectory(self):
        t=np.linspace(0,.16,401);q=np.zeros((len(t),13));q[:,3]=1;q[:,9]=1
        q[:,7]=.5+np.arctan((t-.13)/.02)
        return q,t[1]-t[0]

    def test_zero_delay_identity(self):
        q,dt=self.trajectory(); changed,mapped=warp_elbow(q,dt,0.)
        np.testing.assert_allclose(changed,q,atol=1e-14)
        np.testing.assert_allclose(mapped,np.arange(len(q))*dt)

    def test_initial_state_and_only_elbow_change(self):
        q,dt=self.trajectory(); changed,mapped=warp_elbow(q,dt,1/1440)
        m=model_for([.31,.28],dt)
        v,_=derivatives(m,q,dt); cv,_=derivatives(m,changed,dt)
        np.testing.assert_array_equal(q[:4],changed[:4]);np.testing.assert_array_equal(q[-1],changed[-1])
        np.testing.assert_array_equal(v[0],cv[0])
        np.testing.assert_array_equal(np.delete(q,7,axis=1),np.delete(changed,7,axis=1))
        self.assertTrue(np.all(np.diff(mapped)>0)); self.assertGreater(np.max(np.abs(changed[:,7]-q[:,7])),0)

    def test_invalid_time_mapping_is_rejected_not_clipped(self):
        q,dt=self.trajectory()
        with self.assertRaises(ValueError):warp_elbow(q,dt,1.)
        with self.assertRaises(ValueError):warp_elbow(q,dt,float('nan'))

if __name__=='__main__':unittest.main()
