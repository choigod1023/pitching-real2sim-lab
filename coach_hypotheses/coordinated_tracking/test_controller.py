import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from capacity_budget_gate import rollout
from coupled_payload import model_for
from coach_hypotheses.gain_feasibility.shield import replay_controls
from .controller import (target_states, force_limits, bound_force, controller_force,
                         coordinated_rollout, run_experiment, FREQUENCIES)


class CoordinatedTests(unittest.TestCase):
    def setUp(self):
        self.lengths=np.array([.3,.25]);self.dt=.001
        self.model=model_for(self.lengths,self.dt)
        self.q=np.tile(self.model.qpos0,(101,1));self.v=np.zeros((101,self.model.nv))

    def test_target_zero_exact_and_endpoint_quaternion_angle(self):
        q,v,df=target_states(self.model,self.q,self.v,self.dt,0.,'cubic_in')
        np.testing.assert_array_equal(q,self.q);np.testing.assert_array_equal(v,self.v);np.testing.assert_array_equal(df,0.)
        q,v,df=target_states(self.model,self.q,self.v,self.dt,2.,'smootherstep')
        diff=np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model,diff,1.,self.q[-1],q[-1])
        self.assertAlmostEqual(np.rad2deg(diff[10]),2.)
        np.testing.assert_array_equal(q[:2],self.q[:2]);np.testing.assert_array_equal(v[0],self.v[0])
        self.assertTrue(np.isfinite(df).all())
        with self.assertRaises(ValueError): target_states(self.model,self.q[:3],self.v[:3],self.dt,1.,'cubic_in')

    def test_limits_ignore_terminal_and_clip_all_groups(self):
        controls=np.ones((4,11));controls[-1]=1e20
        limits=force_limits(controls)
        np.testing.assert_array_equal(limits['components'],1.)
        result=bound_force(np.arange(11.)*100,limits)
        self.assertTrue(np.all(np.abs(result)<=1.))
        for name,part in [('shoulder_rotation',slice(3,6)),('wrist_marker_rotation',slice(8,11))]:
            self.assertLessEqual(np.linalg.norm(result[part]),limits['groups'][name]+1e-15)
        with self.assertRaises(ValueError): force_limits(np.zeros((2,10)))
        with self.assertRaises(ValueError): bound_force(np.full(11,np.nan),limits)

    def test_feedback_mass_matrix_formula_and_zero_error(self):
        data=mujoco.MjData(self.model);mujoco.mj_forward(self.model,data)
        limits=force_limits(np.full((3,11),1e12))
        q=data.qpos.copy();v=data.qvel.copy()
        force,error,feedback=controller_force(self.model,data,np.zeros(11),np.zeros(11),q,v,10.,limits)
        np.testing.assert_array_equal(force,0.);np.testing.assert_array_equal(error,0.)
        target=q.copy();delta=np.zeros(11);delta[10]=.02
        mujoco.mj_integratePos(self.model,target,delta,1.)
        f,e,actual=controller_force(self.model,data,np.zeros(11),np.zeros(11),target,v,5.,limits)
        matrix=np.empty((11,11));mujoco.mj_fullM(self.model,data,matrix)
        np.testing.assert_allclose(actual,matrix@((2*np.pi*5)**2*e),atol=1e-12)
        np.testing.assert_array_equal(f,actual)
        with self.assertRaises(ValueError): controller_force(self.model,data,np.zeros(11),np.zeros(11),q,v,-1.,limits)

    def test_zero_target_identity_and_accepted_input_replay(self):
        controls=np.ones((101,11))*.1
        for sub in (1,2):
            b,q,v=rollout(self.lengths,self.q,controls,self.dt,sub)
            for frequency in FREQUENCIES:
                result,p,w,trace=coordinated_rollout(self.lengths,q,v,controls,q,v,np.zeros_like(v),self.dt,sub,frequency)
                np.testing.assert_array_equal(p[::sub],q);np.testing.assert_array_equal(w[::sub],v)
                np.testing.assert_array_equal(trace['source_controls'],controls[:-1])
                self.assertEqual(result['speed_kmh'],b['speed_kmh'])
                rp,rv=replay_controls(self.lengths,p[0],w[0],trace['applied_loads'],trace['dt'])
                np.testing.assert_array_equal(rp,p);np.testing.assert_array_equal(rv,w)

    def test_mismatched_hash_rejected_before_rollout(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'source').write_bytes(b'not approved')
            (p/'report').write_text(json.dumps({'source_sha256':'wrong'}))
            with patch('coach_hypotheses.coordinated_tracking.controller.rollout') as mocked:
                with self.assertRaises(ValueError):run_experiment(p/'source',p/'report',p/'out')
                mocked.assert_not_called()


if __name__=='__main__':unittest.main()
