import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from coupled_payload import model_for
from coach_hypotheses.gain_feasibility.shield import shield_rollout
from coach_hypotheses.budgeted_input_search.search import group_work
from .policy import (TRAIN,EVALUATION,state_features,fit_scaler,transform,policy_controls,
                     work_costs,C_gate,assess,training_pass,choose)
from .study import load_context,evaluate,margins,optimize,run


class SharedPolicyTests(unittest.TestCase):
    def setUp(self):
        lengths=np.array([.3,.25]);dt=.001
        q=np.tile(model_for(lengths,dt).qpos0,(101,1));controls=np.ones((101,11))*.1
        baselines={};traces={};pair={}
        for sub,key in ((1,'coarse'),(2,'fine')):
            b,p,v,tr=shield_rollout(lengths,q,controls,dt,sub)
            b.update(group_work(tr['applied_loads'],v,dt/sub));baselines[sub]=b,p,v;traces[sub]=tr
            pair[key]=dict(b,paired_gain_kmh=0.,policy_pose_pass=True)
        features,index=state_features(traces[2])
        self.ctx={'input':dict(qpos=q,controls=controls,dt=dt,lengths=lengths),
                  'baselines':baselines,'baseline_traces':traces,'baseline_pair':pair,
                  'features':features,'feature_index':index,'receipt':{}}
        self.scaler=fit_scaler([features,features,features]);self.direction=np.ones(33)*.5

    def test_features_do_not_read_release_or_postintervention_states(self):
        trace=copy.deepcopy(self.ctx['baseline_traces'][2]);x,i=state_features(trace)
        self.assertLessEqual((i-len(trace['qpos'])+1)*trace['dt'],-.060)
        trace['qpos'][i+1:,7]+=20;trace['qvel'][i+1:,6:]+=20
        np.testing.assert_array_equal(state_features(trace)[0],x)
        trace['dt']=.00001
        with self.assertRaises(ValueError):state_features(trace)

    def test_scaler_uses_training_only_and_finite_floor(self):
        x=np.array([[1.,2,3],[1,4,5],[1,6,7]])
        s=fit_scaler(x);self.assertEqual(s['scale'][0],1e-6)
        np.testing.assert_array_equal(s['mean'],[1,4,5])
        z,valid=transform([1,4,5],s);self.assertTrue(valid)
        z,valid=transform([20,4,5],s);self.assertFalse(valid);self.assertLessEqual(np.abs(z).max(),2)
        self.assertEqual(s['maximum'][0],1.)
        with self.assertRaises(ValueError):fit_scaler([[1,2,3]])

    def test_policy_same_rule_bounded_and_root_unchanged(self):
        b=self.ctx['input']['controls'];w=np.ones(12)*-.2
        out,c,m=policy_controls(b,.001,self.ctx['features'],self.scaler,w,self.direction)
        np.testing.assert_array_equal(out[:,:6],b[:,:6]);np.testing.assert_array_equal(out[:40],b[:40])
        self.assertTrue(m['supported']);self.assertEqual(c.shape,(33,))
        self.assertLessEqual(np.abs(c).max(),1.)
        self.assertTrue(np.any(out[:,6:]!=b[:,6:]))
        with self.assertRaises(ValueError):policy_controls(b,.001,self.ctx['features'],self.scaler,np.ones(12)*2,self.direction)

    def test_out_of_support_abstains_and_zero_policy_exact_identity(self):
        b=self.ctx['input']['controls']
        for features,w in ((self.ctx['features'],np.zeros(12)),(self.ctx['features']+10,np.ones(12))):
            out,_,m=policy_controls(b,.001,features,self.scaler,w,self.direction)
            np.testing.assert_array_equal(out,b)
        self.assertTrue(m['abstained'])

    def test_union_C_preserves_A_B_and_root_costs(self):
        b=self.ctx['baselines'][2][0]
        self.assertEqual(work_costs(b).shape,(31,))
        self.assertTrue(C_gate(b,b,2,2)['pass'])
        self.assertFalse(C_gate(b,b,2,2.001)['pass'])
        candidate=copy.deepcopy(b);candidate['total_net_work_j']+=.1
        self.assertFalse(C_gate(b,candidate,2,2)['pass'])
        candidate=copy.deepcopy(b);candidate['positive_work_per_dof_j'][6]+=.1
        self.assertFalse(C_gate(b,candidate,2,2)['pass'])

    def test_evaluate_baseline_independent_save_and_raw_constraints(self):
        contexts={p:self.ctx for p in TRAIN}
        with tempfile.TemporaryDirectory() as temp:
            record=evaluate(np.zeros(12),contexts,self.scaler,self.direction,temp)
            self.assertTrue((Path(temp)/TRAIN[0]/'selected-sub2.npz').exists())
        for pair in record['pitches'].values():
            self.assertFalse(pair['assessment']['pass']);self.assertTrue(pair['fine']['C']['pass'])
            self.assertTrue(pair['verification']['accounting_pass'])
        m=margins(contexts,record,0.)
        self.assertEqual(len(m),210);self.assertGreaterEqual(m.min(),-1e-12)
        record['pitches'][TRAIN[1]]['fine']['total_net_work_j']+=1e-6
        self.assertLess(margins(contexts,record,0.).min(),0)

    def test_assess_requires_C_even_if_legacy_accepts(self):
        pair={'coarse':{'speed_kmh':100.1,'paired_gain_kmh':.1,'policy_pose_pass':True,'C':{'pass':True}},
              'fine':{'speed_kmh':100.1,'paired_gain_kmh':.1,'policy_pose_pass':True,'C':{'pass':True}}}
        b={'coarse':{'speed_kmh':100},'fine':{'speed_kmh':100}}
        self.assertTrue(assess(pair,b)['pass']);pair['fine']['C']['pass']=False
        self.assertFalse(assess(pair,b)['pass'])

    def test_selection_needs_every_training_pitch_and_ignores_test_keys(self):
        base={'weights':[0.]*12,'pitches':{p:{'assessment':{'pass':False},'fine':{'paired_gain_kmh':0.}} for p in TRAIN}}
        good=copy.deepcopy(base);good['weights'][0]=.1
        for p in TRAIN:good['pitches'][p]={'assessment':{'pass':True},'fine':{'paired_gain_kmh':.1}}
        self.assertTrue(training_pass(good));self.assertIs(choose([base,good]),good)
        good['pitches'][TRAIN[0]]['assessment']['pass']=False
        self.assertIs(choose([base,good]),base)
        good['pitches'][EVALUATION[0]]={'assessment':{'pass':True}}
        self.assertFalse(training_pass(good))

    def test_load_context_rejects_changed_input_receipt(self):
        import gzip
        folder=Path(__file__).parents[1]/'pitch_transfer/results'
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'3034_1';dest.mkdir()
            (dest/'report.json').write_bytes(gzip.decompress((folder/'3034_1.json.gz').read_bytes()))
            (dest/'input.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Input receipt'):load_context(temp,'3034_1')

    def test_optimizer_has_no_evaluation_context_access(self):
        class Result:
            success=True;message='test-only zero candidate';nit=1;x=np.zeros(13)
        with tempfile.TemporaryDirectory() as temp:
            with patch('coach_hypotheses.shared_policy.study.minimize',return_value=Result()):
                selected,report=optimize({p:self.ctx for p in TRAIN},self.scaler,self.direction,temp)
        self.assertTrue(report['baseline_fallback']);self.assertEqual(report['policy_evaluations'],2)
        with self.assertRaises(ValueError):optimize({EVALUATION[0]:self.ctx},self.scaler,self.direction,'.')

    def test_run_freezes_weights_before_loading_evaluation(self):
        loaded=[]
        with tempfile.TemporaryDirectory() as temp:
            def loader(source,pitch):
                if pitch in EVALUATION:self.assertTrue((Path(temp)/'frozen-policy.json').exists())
                loaded.append(pitch);return self.ctx
            selected={'weights':[0.]*12,'evaluation_id':0}
            info={'baseline_fallback':True}
            with patch('coach_hypotheses.shared_policy.study.load_context',side_effect=loader),patch('coach_hypotheses.shared_policy.study.optimize',return_value=(selected,info)):
                r=run('unused',temp)
            self.assertEqual(loaded,list(TRAIN)+list(EVALUATION));self.assertFalse(r['independent_unseen_evaluation'])
            with self.assertRaises(FileExistsError):run('unused',temp)


if __name__=='__main__':unittest.main()
