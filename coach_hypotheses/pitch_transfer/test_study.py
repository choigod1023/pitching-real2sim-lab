import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from coupled_payload import model_for
from coach_hypotheses.gain_feasibility.shield import shield_rollout
from coach_hypotheses.budgeted_input_search.search import group_work,POLICIES
from .study import (load_frozen_candidate,absolute_transfer,raw_margins,transfer_pass,
                    extra_diagnostics,prepare,pair_evaluation,independent_check,refit,
                    run_pitch,main,COHORT)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.dt=.001;self.lengths=np.array([.3,.25])
        self.q=np.tile(model_for(self.lengths,self.dt).qpos0,(101,1))
        self.controls=np.ones((101,11))*.1
        self.bases={}
        for sub in (1,2):
            b,p,v,t=shield_rollout(self.lengths,self.q,self.controls,self.dt,sub)
            b.update(group_work(t['applied_loads'],v,self.dt/sub))
            self.bases[sub]=b,p,v

    def test_frozen_archive_and_manifest(self):
        x,pair,digest=load_frozen_candidate()
        self.assertEqual(x.shape,(33,));self.assertEqual(len(digest),64)
        self.assertAlmostEqual(pair['fine']['paired_gain_kmh'],.6473596802644153)
        self.assertEqual(len(COHORT),9);self.assertEqual(len(set(COHORT)),9)

    def test_absolute_identity_interpolation_support_and_caps(self):
        out,diagnostic=absolute_transfer(self.controls,self.controls,self.dt,self.controls,.001)
        np.testing.assert_array_equal(out,self.controls)
        source=self.controls.copy();source[50:90,6]=-.5
        out,_=absolute_transfer(self.controls,source,.001,self.controls,.001)
        np.testing.assert_array_equal(out[:40],self.controls[:40])
        np.testing.assert_array_equal(out[-1],self.controls[-1])
        self.assertLessEqual(np.abs(out[:-1]).max(),.1+1e-15)
        self.assertLess(out[60,6],0)
        with self.assertRaises(ValueError):absolute_transfer(self.controls,source,-1,self.controls,.001)
        with self.assertRaises(ValueError):absolute_transfer(self.controls,source[:4],.001,self.controls,.001)

    def test_absolute_transfer_uses_relative_release_not_start(self):
        source=self.controls.copy();source[80,6]=0.
        recipient=np.ones((121,11))*.1
        out,_=absolute_transfer(self.controls,source,.001,recipient,.001)
        self.assertAlmostEqual(out[100,6],0.)
        self.assertAlmostEqual(out[80,6],.1)

    def test_pair_evaluation_and_raw_constraint_no_gain_floor(self):
        pair,tr=pair_evaluation(self.q,self.dt,self.lengths,self.controls,self.bases,np.zeros(33))
        self.assertEqual(set(tr),{1,2})
        self.assertEqual(raw_margins(self.bases,pair).shape,(54,))
        self.assertGreaterEqual(raw_margins(self.bases,pair).min(),-1e-14)
        pair['fine']['positive_work_per_dof_j'][0]+=5e-10
        self.assertLess(raw_margins(self.bases,pair).min(),0)

    def test_transfer_needs_budget_pose_positive_and_resolution(self):
        row={'policy_pose_pass':True,'paired_gain_kmh':.6,'speed_kmh':100.6}
        pair={'coarse':copy.deepcopy(row),'fine':copy.deepcopy(row)}
        baseline={'coarse':{'speed_kmh':100.},'fine':{'speed_kmh':100.}}
        self.assertTrue(transfer_pass(pair,baseline)['pass'])
        pair['fine']['policy_pose_pass']=False
        self.assertFalse(transfer_pass(pair,baseline)['pass'])
        pair['fine']['policy_pose_pass']=True;pair['fine']['paired_gain_kmh']=-.1
        self.assertFalse(transfer_pass(pair,baseline)['pass'])
        pair['fine']['paired_gain_kmh']=.61
        self.assertFalse(transfer_pass(pair,baseline)['pass'])

    def test_independent_replay_accounting_detects_tampering(self):
        pair,tr=pair_evaluation(self.q,self.dt,self.lengths,self.controls,self.bases,np.zeros(33))
        check=independent_check(tr[2],pair['fine']);self.assertTrue(check['accounting_pass'])
        bad=copy.deepcopy(tr[2]);bad['qpos'][-1,0]+=.001
        with self.assertRaises(AssertionError):independent_check(bad,pair['fine'])

    def test_extra_costs_are_not_hidden_by_A(self):
        pair,tr=pair_evaluation(self.q,self.dt,self.lengths,self.controls,self.bases,np.zeros(33))
        base={'summary':self.bases[2][0],'root_peak':np.linalg.norm(tr[2]['applied_loads'][:,:3],axis=1).max()}
        r=extra_diagnostics(base,tr[2],pair['fine'])
        self.assertAlmostEqual(r['net_work_change_j'],0)
        self.assertAlmostEqual(r['root_force_norm_peak_change_n'],0)
        self.assertTrue(r['B_gate']['policy_budget_pass'])

    def test_prepare_freezes_lengths_without_mutating_cached_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);np.savez(root/'ref.npz',qpos=self.q,dt=self.dt,lengths=self.lengths)
            fixed=self.lengths+.01
            with patch('coach_hypotheses.pitch_transfer.study.CACHED',{'sample':'ref.npz'}):
                q,dt,lengths,u,ledger=prepare('sample',root,root/'output',fixed)
            np.testing.assert_array_equal(lengths,fixed)
            np.testing.assert_array_equal(np.load(root/'ref.npz')['lengths'],self.lengths)
            np.testing.assert_allclose(ledger['fixed_minus_native_mm'],10.)
            self.assertEqual(u.shape,(101,11));self.assertEqual(len(ledger['input_sha256']),64)

    def test_refit_zero_fallback_cannot_become_transfer_success(self):
        pair,_=pair_evaluation(self.q,self.dt,self.lengths,self.controls,self.bases,np.zeros(33))
        class Result:
            success=True;message='test zero optimum';nit=1;x=np.zeros(33)
        with tempfile.TemporaryDirectory() as temp:
            with patch('coach_hypotheses.pitch_transfer.study.minimize',return_value=Result()):
                selected,record=refit(self.q,self.dt,self.lengths,self.controls,self.bases,pair,np.zeros(33),Path(temp))
        self.assertTrue(record['baseline_fallback']);self.assertEqual(len(record['messages']),2)
        self.assertFalse(transfer_pass(selected,pair)['pass'])

    def test_run_pitch_saved_zero_transfer_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger={'fixed_lengths_m':self.lengths.tolist()}
            with patch('coach_hypotheses.pitch_transfer.study.prepare',return_value=(self.q,self.dt,self.lengths,self.controls,ledger)):
                result=run_pitch('fixture',temp,temp,np.zeros(33),self.controls,self.controls,self.dt,self.lengths,False)
            self.assertFalse(result['transfers']['normalized']['judgement']['pass'])
            self.assertTrue((Path(temp)/'normalized-sub2.npz').exists())
            self.assertFalse(result['real_athlete_effect_verified'])

    def test_main_argument_guard(self):
        with patch('sys.argv',['study','--lab','missing','--out','missing','--pitch','invalid']):
            with self.assertRaises(SystemExit):main()


if __name__=='__main__':unittest.main()
