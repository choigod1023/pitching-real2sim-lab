import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from hand_speed_derivative import analytic_state,numerical_rotations,improves

class HandDerivativeTests(unittest.TestCase):
    def sample(self,relative_degree=1):
        t=np.linspace(-.009,0,4);o=np.array([.4,.2,1.5])+t[:,None]*[2.,1.,-.5]+t[:,None]**2*[3.,-.2,1.]
        theta=20*t+80*t*t;r=Rotation.from_rotvec(theta[:,None]*[0,0,1]).as_matrix();local=np.array([.15,.03,.01])+t[:,None]*[1.,-.4,.2]
        if relative_degree==2:local+=t[:,None]**2*[4.,2.,-1.]
        ball=o+np.einsum('nij,nj->ni',r,local);expected=np.array([2.,1.,-.5])+np.cross([0,0,20.],local[-1])+[1.,-.4,.2]
        return t,ball,o,r,expected
    def test_velocity_identity_from_training_quadratic_hand_motion(self):
        t,b,o,r,v=self.sample();p,pv,_=analytic_state(t,b,o,r,2);np.testing.assert_allclose(p,b[-1],atol=1e-12);np.testing.assert_allclose(pv,v,atol=1e-10)
    def test_quadratic_relative_motion_endpoint_derivative(self):
        t,b,o,r,v=self.sample(2);p,pv,_=analytic_state(t,b,o,r,2,2);np.testing.assert_allclose(p,b[-1],atol=1e-12);np.testing.assert_allclose(pv,v,atol=1e-10)
    def test_world_frame_change_preserves_velocity_and_position(self):
        t,b,o,r,_=self.sample();p,v,_=analytic_state(t,b,o,r,2);q=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();shift=np.array([1.,-2.,.5]);pp,vv,_=analytic_state(t,b@q.T+shift,o@q.T+shift,q@r,2);np.testing.assert_allclose(pp,q@p+shift,atol=1e-10);np.testing.assert_allclose(vv,q@v,atol=1e-10)
    def test_float32_roundoff_repair_rejects_large_matrix_defect(self):
        _,_,_,r,_=self.sample();raw=r.astype(np.float32).astype(float);proper,info=numerical_rotations(raw);np.testing.assert_allclose(np.swapaxes(proper,1,2)@proper,np.eye(3)[None]+np.zeros_like(proper),atol=1e-14)
        with self.assertRaises(ValueError):numerical_rotations(raw*1.001)
    def test_scalar_label_agreement_cannot_hide_worse_future_motion(self):
        base={'mean_absolute_speed_discrepancy_kmh':1.,'mean_velocity_vector_discrepancy_kmh':5.,'position_rmse_mm':10.};self.assertFalse(improves(dict(base,position_rmse_mm=11.),base));self.assertTrue(improves({k:v*.9 for k,v in base.items()},base))
if __name__=='__main__':unittest.main()
