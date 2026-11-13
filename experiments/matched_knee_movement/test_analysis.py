import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from .analyze import (read,digest,context,same_context,background_frame,background_features,
                      camera_gate,compare_background,weights,fit,design,model_gate,evaluate)
from .diagnose import complete_link
from .render import selected_pairs,measure,edges,source_frames,predictions


class AnalysisTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i in range(8):
            value=float(i-3)
            rows.append({'id':str(i),'game':str(i//2),
                         'delta':{'speed_kmh':i%3-1.,'rpm':(i%4-2)*50.,'horizontal_cm':1.5*value},
                         'count_delta':{'balls':i%2,'strikes':(i//2)%2,'outs_when_up':0},
                         'features':{'knee':{'grid':[value]*9}}})
        return rows

    def test_context_occupancy_and_strict_rules(self):
        row={'inning':'1','balls':'1','strikes':'2','outs_when_up':'1','batter':'123',
             'stand':'L','home_team':'PIT','inning_topbot':'Top','on_1b':'777','on_2b':'','on_3b':'nan'}
        a=context(row);b=context({**row,'on_1b':'888'})
        self.assertEqual(a['bases'],'100');self.assertTrue(same_context(a,b)['full_count'])
        b['balls']=2;self.assertTrue(same_context(a,b)['strict']);self.assertFalse(same_context(a,b)['full_count'])
        b['batter']=124;self.assertFalse(same_context(a,b)['strict'])

    def test_fixed_units_and_missing(self):
        p=self.fixture()[0];p['features']['knee']['grid'][4]=10
        np.testing.assert_allclose(design(p,'knee',4,False),[-1,-1,0,0,2])
        self.assertEqual(len(design(p,'knee',4,True)),6)
        np.testing.assert_allclose(design(p,'knee',4,False,'speed_rpm'),[-1,-1,2])
        with self.assertRaises(ValueError):design(p,'knee',4,False,'unknown')
        p['features']['knee']['grid'][4]=None;self.assertIsNone(design(p,'knee',4,True))

    def test_ridge_augmented_weighted_least_squares(self):
        x=np.array([[1,2],[2,2],[4,-1],[8,3]],float);y=np.array([2,4,1,9],float);games=['a','a','a','b']
        w=weights(games);np.testing.assert_allclose(w,[1/6,1/6,1/6,1/2])
        beta,intercept=fit(x,y,games)
        a=np.column_stack([np.ones(4),x]);penalty=np.diag([0,1,1])
        expected=np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w*y))
        np.testing.assert_allclose(np.r_[intercept,beta],expected,rtol=1e-12,atol=1e-12)

    def test_identical_support_and_no_leakage(self):
        rows=self.fixture();rows[0]['features']['knee']['grid'][4]=None
        report=evaluate(rows,'knee');self.assertEqual(report['evaluated_pairs'],7)
        self.assertFalse(report['pass'])
        for fold in report['folds']:
            self.assertNotIn(fold['game'],fold['train_games'])
            self.assertTrue(all(p['pair']!='0' for p in fold['predictions']))
        json.dumps(report,allow_nan=False)
        altered=json.loads(json.dumps(rows))
        for p in altered:
            if p['game']=='0':p['delta']['horizontal_cm']+=1000
        changed=evaluate(altered,'knee')
        a=next(f for f in report['folds'] if f['game']=='0');b=next(f for f in changed['folds'] if f['game']=='0')
        self.assertEqual(a['m1_coefficients'],b['m1_coefficients'])
        self.assertEqual(a['predictions'][0]['estimates'],b['predictions'][0]['estimates'])

    def test_small_support_does_not_become_zero_success(self):
        r=evaluate(self.fixture()[:4],'knee')
        self.assertEqual(r['evaluated_pairs'],0);self.assertIsNone(r['mae_cm']['m1']);self.assertFalse(r['pass'])
        self.assertFalse(model_gate([],0))

    def test_complete_link_blocks_chaining(self):
        keys=['a','b','c'];match={(a,b):True for a in keys for b in keys}
        match['a','c']=False
        self.assertEqual(complete_link(keys,match),[['a','b'],['c']])

    def test_ablation_keeps_support(self):
        rows=self.fixture();rows[0]['features']['knee']['grid'][4]=None
        full=evaluate(rows,'knee');small=evaluate(rows,'knee',control_mode='speed_rpm')
        self.assertEqual(full['evaluated_pairs'],small['evaluated_pairs'])
        for a,b in zip(full['folds'],small['folds']):
            self.assertEqual(a['train_games'],b['train_games'])
            self.assertEqual(a['train_pairs'],b['train_pairs'])
            self.assertEqual(len(a['m1_coefficients']),len(b['m1_coefficients'])+2)

    def test_display_selection_is_chronological_not_outcome(self):
        rows=[]
        for i in (2,0,1):
            rows.append({'id':str(i),'game':str(i),'date':str(i),'cohort':'speed_contrast',
                         'context_match':{'strict':True},'pair_camera':{'pass':True},
                         'features':{'lead_knee_change':{'nominal':i,'range':[i-1,i+1]}}})
        self.assertEqual([p['id'] for p in selected_pairs({'rows':rows})],['0','1','2'])
        self.assertIn('+1.00',measure(rows[-1]))
        rows[-1]['features']['lead_knee_change']['nominal']=None
        self.assertEqual(measure(rows[-1]),'관측 부족')
        p=np.zeros((33,2));s=np.zeros(33)
        self.assertEqual(edges(p,s,{'trunk_tilt':False,'lead_knee':False,'throwing_elbow':False}),[])
        data={'analyses':{'pair_camera_only/'+k:{'models':[{'nominal':{'folds':[{'predictions':[]}]}}]} for k in ('speed_contrast','matched_speed_spin')}}
        self.assertEqual(predictions(data),{})

    def test_camera_gate_and_bad_background(self):
        m={'matches':30,'inliers':25,'inlier_fraction':.8,'side_inliers':[10,15],
           'span_x':.7,'span_y':.5,'median_px':.3,'p90_px':.8,'scale':1.,'rotation_deg':0.}
        self.assertTrue(camera_gate(m));self.assertFalse(camera_gate({**m,'side_inliers':[24,1]}))
        self.assertFalse(camera_gate({**m,'scale':1.2}))
        empty=background_features(np.zeros((720,1280,3),np.uint8))
        self.assertFalse(compare_background(empty,empty)['pass'])

    def test_background_mask_and_similarity(self):
        rng=np.random.default_rng(42);im=rng.integers(0,256,(720,1280,3),dtype=np.uint8)
        first=background_features(im);points=first[0]
        self.assertTrue(np.all((points[:,0]<.26*1280)|(points[:,0]>=.74*1280)))
        second=background_features(cv2.warpAffine(im,np.array([[1,0,2],[0,1,1]],float),(1280,720)))
        match=compare_background(first,second);self.assertTrue(match['pass']);self.assertLess(match['median_px'],.2)

    def test_sequential_background_frame_and_io(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.mp4';w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(64,64))
            self.assertTrue(w.isOpened())
            for i in range(25):w.write(np.full((64,64,3),i*6,np.uint8))
            w.release();image,clock=background_frame({'video':str(path),'first_source_frame':20})
            self.assertEqual(clock['frame'],10);self.assertAlmostEqual(clock['pts_ms'],500,places=2)
            self.assertEqual(image.shape[:2],(1280,1280));self.assertEqual(len(digest(path)),64)
            j=Path(temp)/'result.json';j.write_text(json.dumps(clock));self.assertEqual(read(j),clock)
            frames=source_frames(path,[3,7]);self.assertEqual(set(frames),{3,7})
            with self.assertRaises(ValueError):source_frames(path,[100])


if __name__=='__main__':unittest.main()
