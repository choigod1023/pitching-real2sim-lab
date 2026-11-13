import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
from .analyze import (read,digest,wrap,weights,angular_mean,fit,circular_median,consecutive,
                      direction_trace,window_features,paired,design,evaluate,simulation_gate)
from .render import selected_pairs,observed_text,source_frames,edges


class ChainTests(unittest.TestCase):
    def fixture(self):
        out=[]
        for i in range(8):
            out.append({'id':str(i),'game':str(i//2),'delta':{'speed_kmh':i%3-1,'rpm':(i%4-2)*50},
                        'count_delta':{'balls':i%2,'strikes':(i//2)%2},
                        'measurements':{'knee':[float(i-3)]*9,'release_x':[float(2*i-3)]*9,'spin_axis':[179. if i%2 else -179.]*9}})
        return out

    def test_wrapped_difference_and_mean(self):
        self.assertEqual(float(wrap(359)), -1.)
        self.assertEqual(paired([359]*3,[1]*3,True),[2.]*9)
        self.assertAlmostEqual(abs(angular_mean([179,-179],np.array([.5,.5]))),180)
        self.assertIsNone(circular_median(np.array([0,180])))
        self.assertAlmostEqual(abs(circular_median(np.array([179,-179]))),180)
        with self.assertRaises(ValueError):angular_mean([0,180],np.array([.5,.5]))

    def test_ridge_matches_augmented_equation(self):
        x=np.array([[1,3],[2,4],[6,3],[1,5]],float);y=np.array([1,4,8,2]);g=['a','a','a','b'];w=weights(g)
        b,i,m=fit(x,y,g);a=np.column_stack([np.ones(4),x]);penalty=np.diag([0,1,1])
        expected=np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w*y))
        np.testing.assert_allclose(np.r_[i,b],expected,atol=1e-12);self.assertAlmostEqual(m,float(w@y))

    def test_circular_model_avoids_false_zero(self):
        x=np.zeros((4,1));g=['a','a','b','b']
        b,i,m=fit(x,[179,-179,178,-178],g,True)
        self.assertAlmostEqual(abs(float(wrap(i))),180);self.assertAlmostEqual(abs(m),180)
        self.assertEqual(float(b[0]),0)

    def test_window_four_samples_and_consecutive(self):
        t=np.arange(12)*1000/60;v=np.ones(12)*30
        self.assertEqual(window_features(v,t,9),[30.,30.,30.])
        v[6]=np.nan
        self.assertIsNone(window_features(v,t,9)[1])
        v[:]=30;v[5]=np.nan
        self.assertEqual(window_features(v,t,9)[1],30.)
        self.assertEqual(consecutive([True,False,True,True,True]),3)
        self.assertEqual(window_features(v[:4],t[:4],3),[None,None,None])

    def test_direction_exclusion_and_missing_hand(self):
        p=np.full((8,33,2),np.nan);s=np.full((8,33),np.nan)
        p[:,14]=[30,30];p[:,16]=[50,30];s[:,[14,16]]=1
        a,mask=direction_trace(p,s,[14,16],[100,100],np.arange(8)+10,[12])
        self.assertTrue(np.isnan(a[2]));self.assertEqual(int(mask.sum()),7);self.assertEqual(a[0],90)
        a,mask=direction_trace(p,s,[16,20],[100,100],np.arange(8),[])
        self.assertFalse(mask.any());self.assertTrue(np.isnan(a).all())
        p[0,16]=[32,30];_,mask=direction_trace(p,s,[14,16],[100,100],np.arange(8),[])
        self.assertFalse(mask[0])

    def test_no_label_leakage_and_shared_support(self):
        rows=self.fixture();rows[0]['measurements']['knee'][4]=None
        a=evaluate(rows,'knee','release_x');self.assertEqual(a['evaluated_pairs'],7);self.assertFalse(a['repeat_pass'])
        changed=json.loads(json.dumps(rows))
        for p in changed:
            if p['game']=='0':p['measurements']['release_x']=[10000]*9
        b=evaluate(changed,'knee','release_x')
        fa=next(x for x in a['folds'] if x['game']=='0');fb=next(x for x in b['folds'] if x['game']=='0')
        self.assertEqual(fa['m1_beta'],fb['m1_beta']);self.assertEqual(fa['predictions'][0]['estimates'],fb['predictions'][0]['estimates'])
        for f in a['folds']:self.assertNotIn(f['game'],f['train_games'])
        json.dumps(a,allow_nan=False)

    def test_angular_mae_and_tiny_support(self):
        r=evaluate(self.fixture(),'knee','spin_axis',mode='speed_rpm')
        self.assertLess(r['mae']['mean'],4)
        r=evaluate(self.fixture()[:4],'knee','release_x')
        self.assertEqual(r['evaluated_pairs'],0);self.assertIsNone(r['mae']['m1']);self.assertFalse(r['repeat_pass'])

    def test_units_missing_and_unknown_controls(self):
        p=self.fixture()[0];x,y=design(p,'knee','release_x',4,'full')
        np.testing.assert_allclose(x,[-1,-1,0,0,-.6]);self.assertEqual(y,-3)
        self.assertEqual(len(design(p,'knee','release_x',4,'speed_rpm')[0]),3)
        with self.assertRaises(ValueError):design(p,'knee','release_x',4,'unknown')
        p['measurements']['release_x'][4]=None;self.assertIsNone(design(p,'knee','release_x',4,'full'))

    def test_gate_never_substitutes_measured_speed(self):
        r=simulation_gate(True,True,False,True,False,False)
        self.assertFalse(r['allow_body_intervention']);self.assertIsNone(r['predicted_gain']['speed_kmh'])
        self.assertIn('same_pitch_3d',r['unmet'])
        self.assertTrue(simulation_gate(True,True,True,True,True,True)['allow_body_intervention'])

    def test_json_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'x.json';p.write_text('{"ok":true}',encoding='utf-8')
            self.assertEqual(read(p),{'ok':True});self.assertEqual(len(digest(p)),64)

    def test_overlay_keeps_missingness_and_source_order(self):
        rows=[{'id':str(i),'date':str(i),'game':str(i),'cohort':'speed_contrast','context_match':{'strict':True},
               'pair_camera':{'pass':True},'measurements':{'knee':[float(i)]*9,'forearm':[None]*9}} for i in (2,0,1)]
        self.assertEqual([p['id'] for p in selected_pairs({'rows':rows})],['0','1','2'])
        self.assertEqual(observed_text(rows[0],'forearm','°'),'관측 부족')
        self.assertIn('+2.00',observed_text(rows[0],'knee','°'))
        p=np.full((33,2),np.nan);s=np.full(33,np.nan)
        self.assertEqual(edges(p,s,{'trunk_tilt':False,'lead_knee':False},{'upperarm':False,'forearm':False}),[])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'raw.mp4';w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(64,64));self.assertTrue(w.isOpened())
            for i in range(10):w.write(np.full((64,64,3),i*15,np.uint8))
            w.release();images=source_frames(path,[2,7]);self.assertEqual(set(images),{2,7})
            with self.assertRaises(ValueError):source_frames(path,[15])


if __name__=='__main__':unittest.main()
