import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json

import mujoco
import numpy as np

from coupled_payload import model_for
from joint_intervention import modify_trajectory
from coach_hypotheses.gain_feasibility.profiles import (
    profile_value, modify_wrist, bounded_wrist_controls, ball_states,
    movement_metrics, run_experiment,
)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.model = model_for([.3, .25], .001)
        self.q = np.tile(self.model.qpos0, (101,1))

    def test_profiles_have_same_endpoint_but_different_velocity(self):
        h = 1e-6
        for name in ('cubic_in','smoothstep','smootherstep'):
            np.testing.assert_allclose(profile_value([0,1], name), [0,1])
            slope = (profile_value(1.,name)-profile_value(1-h,name))/h
            self.assertAlmostEqual(float(slope), 3. if name=='cubic_in' else 0., delta=1e-4)
        with self.assertRaises(ValueError): profile_value([np.nan], 'cubic_in')
        with self.assertRaises(ValueError): profile_value([-1], 'cubic_in')
        with self.assertRaises(ValueError): profile_value([.5], 'unknown')

    def test_cubic_matches_original_and_preserves_initial_state(self):
        q, a = modify_wrist(self.model, self.q, .001, .030, 'cubic_in')
        old, oa = modify_trajectory(self.model, self.q, .001, 10, 2)
        np.testing.assert_array_equal(q,old)
        np.testing.assert_array_equal(a,oa)
        np.testing.assert_array_equal(q[:71], self.q[:71])
        for name in ('smoothstep','smootherstep'):
            altered, angles = modify_wrist(self.model, self.q, .001, .030, name)
            dq = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model,dq,1.,self.q[-1],altered[-1])
            self.assertAlmostEqual(np.rad2deg(dq[10]),2.)
        with self.assertRaises(ValueError): modify_wrist(self.model,self.q,.001,.1,'cubic_in')

    def test_clip_uses_applied_intervals_and_preserves_other_inputs(self):
        b = np.ones((5,11)); b[:-1,8:11] = [[1,2,3],[-1,-2,-3],[1,2,3],[-1,-2,-3]]
        b[-1] = 1e8
        requested = np.full_like(b,100.)
        original = b.copy()
        out = bounded_wrist_controls(b,requested,1.)
        np.testing.assert_array_equal(b,original)
        np.testing.assert_array_equal(out[:,:8],b[:,:8])
        np.testing.assert_array_equal(out[-1],b[-1])
        self.assertTrue(np.all(np.abs(out[:-1,8:11]) <= [1,2,3]))
        self.assertTrue(np.all(np.linalg.norm(out[:-1,8:11],axis=1) <= np.sqrt(14)+1e-14))
        np.testing.assert_array_equal(bounded_wrist_controls(b,requested,0),b)
        with self.assertRaises(ValueError): bounded_wrist_controls(b,requested,-.1)
        with self.assertRaises(ValueError): bounded_wrist_controls(b[:,:10],requested[:,:10],1)

    def test_ball_jacobian_matches_position_difference(self):
        q = self.model.qpos0.copy(); v = np.arange(self.model.nv)*.1
        p, vel = ball_states(self.model,q[None],v[None])
        h=1e-6; positions=[]
        for step in (-h,h):
            state=q.copy();mujoco.mj_integratePos(self.model,state,v,step)
            positions.append(ball_states(self.model,state[None],np.zeros((1,self.model.nv)))[0][0])
        np.testing.assert_allclose((positions[1]-positions[0])/(2*h),vel[0],atol=1e-9)
        with self.assertRaises(ValueError): ball_states(self.model,[],[])

    def test_movement_metrics_zero_and_direction_changes(self):
        vel=np.zeros((101,self.model.nv));vel[:,0]=1.
        result=movement_metrics(self.model,self.q,vel,self.q,vel)
        self.assertTrue(result['pose_screen_pass'])
        self.assertEqual(result['max_ball_position_change_mm'],0)
        q2=self.q.copy();q2[:,1]+=.03
        shifted=movement_metrics(self.model,self.q,vel,q2,vel)
        self.assertFalse(shifted['pose_screen_pass'])
        self.assertAlmostEqual(shifted['max_ball_position_change_mm'],30.)
        v2=vel.copy();v2[:,1]=1.
        direction=movement_metrics(self.model,self.q,vel,self.q,v2)
        self.assertAlmostEqual(direction['release_direction_change_deg'],45.)

    def test_mismatched_source_hash_fails_before_simulation(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'source').write_bytes(b'untrusted')
            (p/'report').write_text(json.dumps({'source_sha256':'wrong'}))
            with patch('coach_hypotheses.gain_feasibility.profiles.rollout') as roll:
                with self.assertRaises(ValueError): run_experiment(p/'source',p/'out',p/'report')
                roll.assert_not_called()


if __name__ == '__main__':
    unittest.main()
