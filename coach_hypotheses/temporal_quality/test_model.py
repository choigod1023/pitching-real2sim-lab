import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from . import model as m
from ..pitch_quality import conditional as old


def rows(date='2025-07-10', game_base=10, n=90):
    result = []
    for i in range(n):
        call = 'S' if i%3 == 0 else ('C' if i%5 == 0 else 'X')
        result.append(dict(gamePk=game_base+i%2, gameDate=date, atBatIndex=i, pitchIndex=0,
            pitcherId=1+i%2, batterId=100+i%7, pitchType='FF', identity_status='verified_within_play',
            pitchHand='R', batSide='R' if i%2 else 'L', startSpeed=90+i%9, pX=(i%11)/4-.5,
            pZ=1+(i%7)/3, spinRate=2100+i%15*20, breakVerticalInduced=12+i%5,
            breakHorizontal=-4+i%3, pre_count=dict(balls=0,strikes=i%2), callCode=call,detailsCode=call,
            official_pitch_code=dict(code=call,swingStatus=call!='C',swingMissStatus=call=='S',buntAttemptStatus=False)))
        # Both pitchers need multiple training games, independent of outcome.
        result[-1]['gamePk'] = game_base+(i//2)%3
    return result


class ModelTests(unittest.TestCase):
    def test_digest_save_and_date_exclusion_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.json'; rr=rows(); rr[1]['pitchType']='SI'; rr[2]['spinRate']=None
            rr[3]['detailsCode']='UNKNOWN'; rr[4]['identity_status']='uncertain_quarantined'
            rr[5]['batSide']=None; rr[6]['pre_count']=dict(balls=4,strikes=1)
            rr[7]['official_pitch_code']['buntAttemptStatus']=True
            m.save(p,rr); selected,a=m.read_rows(p,'2025-06-01','2025-07-15')
            self.assertEqual(a['raw'],90); self.assertEqual(len(selected),83)
            self.assertEqual(len(m.digest(p)),64)
            self.assertEqual(len(m.swings(selected)),sum(r['official_pitch_code']['swingStatus'] for r in selected))
            with self.assertRaises(ValueError): m.read_rows(p,'2025-08-01','2025-08-31')
            m.save(p,[rr[0],rr[0]])
            with self.assertRaises(ValueError): m.read_rows(p,'2025-06-01','2025-07-15')
            m.save(p,[])
            with self.assertRaises(ValueError): m.read_rows(p,'2025-06-01','2025-07-15')

    def test_targets_and_swings(self):
        rr=rows(); self.assertEqual(set(m.target(rr,'swing')),{0,1})
        for kind in m.TARGETS:
            np.testing.assert_array_equal(m.target(rr,kind),old.target_values(rr,kind))

    def test_spline_partition_constant_clip_and_training_invariance(self):
        s=m.Spline().fit(np.arange(12.)); knots=s.knots.copy()
        b,clip=s.transform([-100,0,5,11,100]); np.testing.assert_allclose(b.sum(axis=1),1)
        self.assertEqual(clip['below'],1);self.assertEqual(clip['above'],1)
        np.testing.assert_allclose(b[0],b[1]); np.testing.assert_allclose(b[-1],b[-2])
        np.testing.assert_array_equal(s.knots,knots)
        b,_=m.Spline().fit([3,3,3]).transform([1,3,5]);np.testing.assert_array_equal(b,np.ones((3,1)))
        with self.assertRaises(ValueError): m.Spline().fit([np.nan])
        with self.assertRaises(ValueError): s.transform([np.nan])

    def test_legacy_features_exactly_preserved(self):
        rr=m.swings(rows())
        for model,kind in [('L0','control'),('L1','both')]:
            x,n,clip=m.Features().fit(rr,model).transform(rr)
            original,names,nn=old.features(rr,kind)
            np.testing.assert_array_equal(x,original);self.assertEqual(n,nn);self.assertEqual(clip,{})
        with self.assertRaises(ValueError):m.Features().fit(rr,'XX')

    def test_nonlinear_design_is_shared_and_outcome_independent(self):
        rr=rows(); control=m.Features().fit(rr,'G0'); both=m.Features().fit(rr,'GB1')
        x,n,_=control.transform(rr); z,nz,_=both.transform(rr)
        np.testing.assert_array_equal(x[:,:n],z[:,:n]); self.assertGreater(nz,n)
        np.testing.assert_array_equal(x[:,n:],z[:,nz:])
        changed=copy.deepcopy(rr)
        for r in changed:r['callCode']='W';r['official_pitch_code']['swingMissStatus']=True
        np.testing.assert_array_equal(both.transform(changed)[0],z)
        for name in ('startSpeed','pX','pZ'):
            np.testing.assert_array_equal(control.splines[name].knots,both.splines[name].knots)
        self.assertEqual(len(both.names),z.shape[1])
        extreme=copy.deepcopy(rr[:2]);extreme[0]['pZ']=100
        xx,_,clips=both.transform(extreme)
        first=both.names.index('IVB_basis_0_raw_pZ')
        iv=both.splines['breakVerticalInduced'].transform([r['breakVerticalInduced'] for r in extreme])[0][:,1]
        self.assertAlmostEqual(xx[0,first],iv[0]*100)

    def test_indicator_unknown_is_zero(self):
        x=m.indicator([1,2,3],{1:0,3:1}).toarray()
        np.testing.assert_array_equal(x,[[1,0],[0,0],[0,1]])

    def test_logistic_legacy_predictions_match_original_solver(self):
        rr=m.swings(rows(n=150)); f=m.Logistic().fit(rr,m.TARGETS[0],'L0',1)
        x,names,n=old.features(rr,'control');e=old.Encoder().fit(x,old.group_ids(rr),n)
        design=e.transform(x,old.group_ids(rr));beta,fit=old.fit_logistic(design,old.target_values(rr,m.TARGETS[0]))
        self.assertTrue(f.meta['success']);self.assertTrue(fit['success'])
        np.testing.assert_allclose(f.predict(rr)[0],expit(design@beta),atol=1e-4,rtol=0)
        allone=copy.deepcopy(rr)
        for r in allone:r['callCode']='S'
        with self.assertRaises(ValueError):m.Logistic().fit(allone,m.TARGETS[0],'G0',1)

    def test_batter_term_and_no_test_outcome_training_leak(self):
        rr=m.swings(rows(n=180)); f=m.Logistic().fit(rr,m.TARGETS[0],'GB1',10)
        self.assertTrue(f.meta['success']); before=f.beta.copy()
        test=rows('2025-08-01',100,n=30)
        p,_=f.predict(test)
        for r in test:r['callCode']='S';r['batterId']=999999
        q,_=f.predict(test);self.assertTrue(np.isfinite(q).all())
        np.testing.assert_array_equal(f.beta,before);self.assertEqual(len(p),30)

    def test_crossed_batter_penalties_match_independent_dense_newton(self):
        rr=m.swings(rows(n=180));f=m.Logistic().fit(rr,m.TARGETS[0],'GB1',10)
        raw,n,_=f.features.transform(rr);g=old.group_ids(rr)
        centered=raw.copy()
        centered[:,:n]=(centered[:,:n]-np.array([f.encoder.means[v] for v in g]))/f.encoder.scale
        x=np.column_stack([np.ones(len(rr)),*(g==v for v in f.encoder.groups),centered,
                           *(np.array([r['batterId'] for r in rr])==v for v in f.batters)])
        penalty=np.r_[0,np.ones(len(f.encoder.groups)),np.full(raw.shape[1],10.),np.ones(len(f.batters))]
        y=m.target(rr,m.TARGETS[0]);reference=np.zeros(x.shape[1])
        for _ in range(30):
            p=expit(x@reference)
            gradient=x.T@(y-p)-penalty*reference
            hessian=x.T@((p*(1-p))[:,None]*x)+np.diag(penalty)
            step=np.linalg.solve(hessian,gradient);reference+=step
            if np.max(abs(step))<1e-10:break
        np.testing.assert_allclose(f.predict(rr)[0],expit(x@reference),atol=1e-4,rtol=0)
        self.assertEqual(len(f.audit['feature_names']),raw.shape[1])

    def test_equal_player_metrics_paired_bootstrap_and_verdict(self):
        rr=[dict(pitcherId=1,y=0,A=.1,B=.2)]*100+[dict(pitcherId=2,y=0,A=.9,B=.8)]
        metrics,pp=m.metrics(rr,('A','B'))
        self.assertAlmostEqual(metrics['A']['brier'],.41)
        c=m.contrast('A','B',pp,n=300);self.assertAlmostEqual(c['brier_gain'],.07)
        self.assertEqual(c,m.contrast('A','B',pp,n=300))
        s=dict(audit=dict(test_after_eligibility=2000,pitchers=50),fits=[dict(success=True)]*6,
               metrics=dict(A=dict(brier=.2,log_loss=.5),B=dict(brier=.19,log_loss=.49),pitcher_mean=dict(brier=.3)))
        comp=dict(reference='A',candidate='B',pitcher_bootstrap_95pct=[.001,.003])
        self.assertFalse(m.verdict(s,comp)['passed'])
        self.assertTrue(m.verdict(s,comp,True)['passed']);s['fits'][0]=dict(success=False)
        self.assertFalse(m.verdict(s,comp,True)['passed'])
        with self.assertRaises(ValueError):m.metrics([],('A',))

    def test_eligibility_and_game_leakage(self):
        train=m.swings(rows(n=180));test=m.swings(rows('2025-08-01',100,n=60))
        kept,a=m.eligible_test(train,test);self.assertEqual(len(kept),len(test))
        with contextlib.redirect_stdout(io.StringIO()):
            s,p,pp=m.evaluate(train,test,m.TARGETS[0],('GB0',),1)
        self.assertTrue(s['fits'][0]['success']);self.assertEqual(len(p),len(test))
        with self.assertRaises(ValueError):m.evaluate(train,train,m.TARGETS[0],('L0',),1)
        with self.assertRaises(ValueError):m.evaluate(train,[dict(test[0],pitcherId=99)],m.TARGETS[0],('L0',),1)

    def test_choose_final_provenance_and_fixed_learning_curve(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);a=root/'train.json';b=root/'validation.json';t=root/'test.json';choice=root/'choice.json'
            m.save(a,rows(n=180));m.save(b,rows('2025-07-20',100,n=80));m.save(t,rows('2025-08-10',200,n=80))
            selected=m.choose(a,b,choice);self.assertIn(selected['selected_lambda'],(1,10))
            self.assertFalse(selected['final_test_opened'])
            audit=root/'audit.json';m.save(audit,dict(collection_complete=True,interval=['2025-06-01','2025-08-31'],
                schedule_days_received=92,schedule_days_requested=92,failures=[],selected_game_count=9,parsed_game_count=9,
                split_exports={k:dict(sha256=m.digest(p)) for k,p in zip(('dev_train','dev_validation','final_test'),(a,b,t))}))
            result=m.final(a,b,t,choice,root/'out',root/'local',audit)
            self.assertEqual(len(result['analyses'][m.TARGETS[0]]['fits']),6)
            self.assertFalse(result['analyses'][m.TARGETS[0]]['verdict_A']['passed'])
            curve=result['learning_curve'];self.assertEqual(len(curve['periods']),3)
            self.assertEqual(len({r['audit']['test_after_eligibility'] for r in curve['periods']}),1)
            selected['code_sha256']='changed';m.save(choice,selected)
            with self.assertRaises(ValueError):m.final(a,b,t,choice,root/'out',root/'local',audit)
            broken=json.loads(audit.read_text());broken['collection_complete']=False;m.save(audit,broken)
            with self.assertRaises(ValueError):m.collection_integrity(audit,(a,b,t))
            broken['collection_complete']=True;broken['split_exports']['final_test']['sha256']='wrong';m.save(audit,broken)
            with self.assertRaises(ValueError):m.collection_integrity(audit,(a,b,t))
        self.assertEqual(m.learning_curve([],[],1)['status'],'no_common_eligible_players')

    def test_main_stage_dispatch(self):
        with patch('sys.argv',['model','choose','--train','a','--validation','b']),patch.object(m,'choose') as f:
            m.main();self.assertEqual(str(f.call_args[0][0]),'a')
        with patch('sys.argv',['model','final','--train','a','--validation','b','--test','c','--collection-audit','d']),patch.object(m,'final') as f:
            m.main();self.assertEqual(str(f.call_args[0][2]),'c')


if __name__=='__main__':unittest.main()
