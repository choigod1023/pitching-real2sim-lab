import unittest
import mujoco
import numpy as np
from coupled_payload import model_for
from torque_arm import derivatives
from capacity_budget_gate import rollout
from joint_intervention import modify_trajectory, inverse_controls, initial_energy


class JointInterventionTests(unittest.TestCase):
    def setUp(self):
        self.dt=.001
        self.lengths=np.array([.3,.25])
        self.model=model_for(self.lengths,self.dt)
        self.q=np.tile(self.model.qpos0,(101,1))

    def test_zero_modification_exact(self):
        changed,_=modify_trajectory(self.model,self.q,self.dt,6,0)
        np.testing.assert_array_equal(changed,self.q)
        np.testing.assert_array_equal(inverse_controls(self.model,changed,self.dt),inverse_controls(self.model,self.q,self.dt))

    def test_only_window_and_declared_quaternion_axis_changes(self):
        for dof in (6,8,9,10):
            for angle in (-2,2):
                changed,_=modify_trajectory(self.model,self.q,self.dt,dof,angle)
                np.testing.assert_array_equal(changed[:71],self.q[:71])
                difference=np.zeros(self.model.nv)
                mujoco.mj_differentiatePos(self.model,difference,1.,self.q[-1],changed[-1])
                expected=np.zeros(self.model.nv); expected[dof]=np.deg2rad(angle)
                np.testing.assert_allclose(difference,expected,atol=1e-14)

    def test_initial_state_velocity_energy_equal(self):
        changed,_=modify_trajectory(self.model,self.q,self.dt,9,2)
        v0,_=derivatives(self.model,self.q,self.dt)
        v1,_=derivatives(self.model,changed,self.dt)
        np.testing.assert_array_equal(v0[0],v1[0])
        self.assertEqual(initial_energy(self.model,self.q[0],v0[0]),initial_energy(self.model,changed[0],v1[0]))

    def test_terminal_inverse_force_is_not_applied(self):
        controls=inverse_controls(self.model,self.q,self.dt)
        a,qa,va=rollout(self.lengths,self.q,controls,self.dt,2)
        controls[-1]=1e10
        b,qb,vb=rollout(self.lengths,self.q,controls,self.dt,2)
        np.testing.assert_array_equal(qa,qb)
        np.testing.assert_array_equal(va,vb)
        self.assertEqual(a['peak_abs_generalized_force_per_dof'],b['peak_abs_generalized_force_per_dof'])

    def test_ramp_changes_velocity_not_only_terminal_pose(self):
        changed,_=modify_trajectory(self.model,self.q,self.dt,6,2)
        v,_=derivatives(self.model,changed,self.dt)
        self.assertAlmostEqual(v[-1,6],np.deg2rad(2)*(1-(29/30)**3)/self.dt,places=10)
        self.assertGreater(v[-1,6],3.)


if __name__=='__main__':
    unittest.main()
