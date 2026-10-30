import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from coupled_payload import model_for
from coach_hypotheses.gain_feasibility.shield import shield_rollout
from .search import (temporal_basis,applied_controls,group_work,cost_vector,policy_gate,
                     passes_policy,normalized_margins,evaluate,select_resolved,run_search,POLICIES)


class BudgetInputTests(unittest.TestCase):
    def setUp(self):
        self.dt=.001;self.lengths=np.array([.3,.25]);self.model=model_for(self.lengths,self.dt)
        self.q=np.tile(self.model.qpos0,(101,1));self.controls=np.ones((101,11))*.1
        b,p,v,t=shield_rollout(self.lengths,self.q,self.controls,self.dt)
        b.update(group_work(t['applied_loads'],v,self.dt));self.base=(b,p,v)

    def test_basis_support_and_zero_parameter_identity(self):
        basis=temporal_basis(101,self.dt)
        self.assertEqual(basis.shape,(101,3));np.testing.assert_array_equal(basis[:40],0.)
        np.testing.assert_array_equal(basis[-1],0.)
        self.assertLessEqual(basis.max(),1.)
        self.assertTrue(np.all(np.count_nonzero(basis,axis=1)<=1))
        raw=self.controls.copy();raw[-1]=1e10
        np.testing.assert_array_equal(applied_controls(raw,basis,np.zeros(33)),raw)
        with self.assertRaises(ValueError):temporal_basis(3,self.dt)

    def test_parameter_bound_input_peaks_terminal_and_no_mutation(self):
        basis=temporal_basis(101,self.dt);raw=self.controls.copy();raw[-1]=1e10
        result=applied_controls(raw,basis,np.ones(33))
        np.testing.assert_array_equal(result[:40],raw[:40]);np.testing.assert_array_equal(result[-1],raw[-1])
        self.assertTrue(np.all(np.abs(result[:-1])<=.1+1e-15))
        np.testing.assert_array_equal(self.controls,.1)
        with self.assertRaises(ValueError):applied_controls(raw,basis,np.ones(32))
        with self.assertRaises(ValueError):applied_controls(raw,basis,np.full(33,2.))

    def test_group_power_splits_after_dot_and_rotation_invariance(self):
        rng=np.random.default_rng(2);f=rng.normal(size=(6,11));v=rng.normal(size=(7,11))
        before=group_work(f,v,.01)
        r=np.linalg.qr(rng.normal(size=(3,3)))[0]
        fr,vr=f.copy(),v.copy();fr[:,:3]=fr[:,:3]@r;vr[:,:3]=vr[:,:3]@r
        after=group_work(fr,vr,.01)
        np.testing.assert_allclose(before['group_positive_work_j'],after['group_positive_work_j'],atol=1e-15)
        f=np.zeros((1,11));f[0,:2]=1
        v=np.zeros((2,11));v[:,:2]=[1,-1]
        self.assertEqual(group_work(f,v,.1)['group_total_positive_work_j'],0.)
        with self.assertRaises(ValueError):group_work(f,v,-1.)

    def test_legacy_and_exploratory_costs_and_work_constraints_separate(self):
        b=self.base[0];c=copy.deepcopy(b)
        c['positive_work_per_dof_j'][6]+=.1
        self.assertFalse(policy_gate(b,c,POLICIES[0])['strict_model_budget_pass'])
        diagnostic=policy_gate(b,c,POLICIES[1])
        self.assertTrue(diagnostic['policy_budget_pass'])
        self.assertFalse(diagnostic['original_strict_gate']['strict_model_budget_pass'])
        c['group_total_positive_work_j']+=.1
        self.assertFalse(policy_gate(b,c,POLICIES[1])['policy_budget_pass'])
        with self.assertRaises(ValueError):cost_vector(b,'unknown')

    def test_evaluate_baseline_is_feasible_and_work_margins_detect_excess(self):
        for policy in POLICIES:
            r,_=evaluate(self.lengths,self.q,self.controls,self.dt,1,self.base,policy,np.zeros(33))
            self.assertTrue(passes_policy(r,policy));self.assertTrue(r['policy_pose_pass'])
            self.assertEqual(r['paired_gain_kmh'],0.)
            self.assertTrue(np.all(normalized_margins(self.base[0],r,policy)>=-1e-12))
            r['group_total_positive_work_j']+=1
            if policy==POLICIES[1]:self.assertLess(normalized_margins(self.base[0],r,policy).min(),0)

    def test_selection_never_discards_feasible_baseline_for_failed_gain(self):
        b={'speed_kmh':10.,'paired_gain_kmh':0.,'policy_pose_pass':True}
        fallback={'parameters':[0.]*33,'coarse':b,'fine':b}
        bad={'parameters':[1.]*33,'coarse':dict(b,speed_kmh=12.,paired_gain_kmh=2.,policy_pose_pass=False),
             'fine':dict(b,speed_kmh=12.,paired_gain_kmh=2.,policy_pose_pass=False)}
        self.assertIs(select_resolved([fallback,bad],POLICIES[0]),fallback)
        good=copy.deepcopy(bad);good['coarse']['policy_pose_pass']=True;good['fine']['policy_pose_pass']=True
        self.assertIs(select_resolved([fallback,good],POLICIES[0]),good)
        good['fine']['speed_kmh']+=.01
        good['fine']['paired_gain_kmh']+=.01
        self.assertIs(select_resolved([fallback,good],POLICIES[0]),fallback)

    def test_source_hash_rejected_before_any_optimization(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'source').write_bytes(b'unknown')
            (p/'prior').write_text(json.dumps({'source_sha256':'wrong'}))
            with patch('coach_hypotheses.budgeted_input_search.search.minimize') as optimizer:
                with self.assertRaises(ValueError):run_search(p/'source',p/'prior',p/'out')
                optimizer.assert_not_called()


if __name__=='__main__':unittest.main()
