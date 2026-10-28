import unittest
import mujoco
import numpy as np
from trace_error_audit import integrate_inputs
from torque_arm import derivatives

class TraceInputTests(unittest.TestCase):
    def model(self,dt=.1):
        return mujoco.MjModel.from_xml_string(f'<mujoco><option timestep="{dt}" gravity="0 0 0"/><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    def test_linear_force_at_rk4_stages_matches_analytic_solution(self):
        m=self.model();t=np.arange(11)*.1;loads=np.zeros((11,6));loads[:,0]=t
        p,v,d=integrate_inputs(m,m.qpos0.copy(),np.zeros(6),loads,.1,1,'foh')
        np.testing.assert_allclose(p[:,0],t**3/6,atol=1e-12)
        np.testing.assert_allclose(v[:,0],t**2/2,atol=1e-12)
        self.assertGreater(d['callback_evaluations'],10)
        self.assertIsNone(mujoco.get_mjcb_control())
    def test_fixed_input_substeps_do_not_repair_zoh_input_error(self):
        m=self.model();loads=np.zeros((11,6));loads[:,0]=np.arange(11)*.1
        a,av,_=integrate_inputs(m,m.qpos0.copy(),np.zeros(6),loads,.1,1,'zoh')
        b,bv,_=integrate_inputs(m,m.qpos0.copy(),np.zeros(6),loads,.1,2,'zoh')
        np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_allclose(av,bv,atol=1e-12)
        self.assertGreater(abs(a[-1,0]-1/6),.01)
    def test_constant_force_policies_agree(self):
        m=self.model();loads=np.zeros((11,6));loads[:,0]=2.
        a,av,_=integrate_inputs(m,m.qpos0.copy(),np.zeros(6),loads,.1,1,'zoh')
        b,bv,_=integrate_inputs(m,m.qpos0.copy(),np.zeros(6),loads,.1,1,'foh')
        np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_allclose(av,bv,atol=1e-12)
    def test_padded_derivative_removes_quadratic_boundary_velocity_bias(self):
        m=self.model(.01);t=np.arange(-2,103)*.01;q=np.tile(m.qpos0,(len(t),1));q[:,0]=t*t
        v,a=derivatives(m,q,.01);vb,_=derivatives(m,q[2:-2],.01)
        np.testing.assert_allclose(v[2:-2,0],2*t[2:-2],atol=1e-12)
        np.testing.assert_allclose(a[2:-2,0],2,atol=1e-10)
        self.assertAlmostEqual(vb[0,0],.01)
    def test_foreign_callback_is_preserved_and_not_used(self):
        m=self.model();old=lambda m,d:None
        mujoco.set_mjcb_control(old)
        try:
            with self.assertRaises(RuntimeError):integrate_inputs(m,m.qpos0.copy(),np.zeros(6),np.zeros((3,6)),.1,1,'foh')
            self.assertIs(mujoco.get_mjcb_control(),old)
        finally:mujoco.set_mjcb_control(None)

if __name__=='__main__':unittest.main()
