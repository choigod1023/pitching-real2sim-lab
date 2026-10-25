import unittest
import numpy as np
from capacity_budget_gate import shift_controls, shift_applied_intervals, split_linear_power, rollout, gate
import test_coupled_payload
from coupled_payload import model_for, inverse_and_rollout


class CapacityBudgetTests(unittest.TestCase):
    def test_signed_work_zero_crossing(self):
        a,b,c=split_linear_power(np.array([-2.,2.,-3.]),np.array([2.,2.,-3.]),1.)
        np.testing.assert_allclose(a,[.5,2,0])
        np.testing.assert_allclose(b,[.5,0,3])
        np.testing.assert_allclose(a-b,c)

    def test_shift_zero_pad_preserves_unselected(self):
        x=np.arange(33).reshape(3,11)
        for sign in [-1,1]:
            y=shift_controls(x,slice(8,11),sign)
            np.testing.assert_array_equal(y[:,:8],x[:,:8])
            np.testing.assert_array_equal(y[0 if sign>0 else -1,8:],0)
        np.testing.assert_array_equal(shift_controls(x,slice(8,11),0),x)

    def test_actual_velocity_work_differs_with_forcing(self):
        q=np.repeat(test_coupled_payload.PayloadTests().pose(),21,axis=0)
        m=model_for([.3,.25],.001)
        _,loads=inverse_and_rollout(m,q,.001)
        # Reference is stationary; a horizontal force causes actual motion/work.
        loads[:,0]+=5.
        coarse,pos,vel=rollout([.3,.25],q,loads,.001,1)
        fine,_,_=rollout([.3,.25],q,loads,.001,2)
        self.assertGreater(coarse['positive_work_per_dof_j'][0],0)
        self.assertGreater(vel[-1,0],0)
        self.assertLess(abs(fine['energy_change_minus_applied_work_j']),1e-6)
        self.assertLess(abs(fine['energy_change_minus_applied_work_j']),abs(coarse['energy_change_minus_applied_work_j'])/3)
        self.assertLess(abs(fine['total_net_work_j']-coarse['total_net_work_j']),1e-6)

    def test_unused_terminal_load_never_enters_rollout(self):
        x=np.ones((5,11)); x[-1,8:]=1000
        y=shift_applied_intervals(x,slice(8,11),-1)
        self.assertLessEqual(np.max(y[:-1,8:]),1)
        np.testing.assert_array_equal(y[-1],x[-1])
        np.testing.assert_array_equal(y[-2,8:],0)

    def test_gate_detects_more_work_at_same_force(self):
        a={'positive_work_per_dof_j':[1.,2.], 'negative_work_magnitude_per_dof_j':[0.,1.],
           'peak_abs_generalized_force_per_dof':[5.,6.], 'shoulder_translation_net_work_j':1.,
           'shoulder_rotation_net_work_j':0., 'peak_joint_torque_norm_nm':{'wrist':6.}}
        self.assertTrue(gate(a,a)['strict_model_budget_pass'])
        b=dict(a,positive_work_per_dof_j=[1.1,2.])
        self.assertFalse(gate(a,b)['strict_model_budget_pass'])


if __name__=='__main__': unittest.main()
