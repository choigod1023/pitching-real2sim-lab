import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .analyze import (angles, digest, evaluate, features, map_clock, original_clock,
                      outcomes, pair_feature, read, repetition, ridge, window_median, wrap)


class MotionTests(unittest.TestCase):
    def test_angles_and_visibility(self):
        p = np.zeros((5, 33, 2), float)
        for i, xy in {11:(30,20),12:(60,20),23:(30,60),24:(60,60),25:(30,90),27:(60,90),14:(80,20),16:(80,50)}.items():
            p[:, i] = xy
        s = np.ones((5, 33)); s[1, 16] = .2
        trace, valid = angles(p, s, [100,120], np.arange(5), {"knee":[2]})
        self.assertAlmostEqual(trace['trunk_tilt'][0], 0)
        self.assertAlmostEqual(trace['lead_knee'][0], 90)
        self.assertAlmostEqual(trace['throwing_elbow'][0], 90)
        self.assertFalse(valid['arm_direction'][1])
        self.assertTrue(np.isnan(trace['lead_knee'][2]))

    def test_circular_boundary_and_ambiguous_arc(self):
        self.assertEqual(float(wrap(358)), -2)
        t = np.arange(-160, 1, 10.)
        values = np.tile([179., -179.], 9)[:len(t)]
        self.assertAlmostEqual(abs(window_median(values,t,-150,-33,4)), 180)
        self.assertIsNone(window_median(np.linspace(-160,160,len(t)),t,-150,-33,4))

    def test_missing_and_incomplete_windows(self):
        t = np.arange(-160,1,10.); values = np.full(len(t),45.)
        self.assertEqual(window_median(values,t,-150,-33,4),45.)
        self.assertIsNone(window_median(values[4:],t[4:],-150,-33,4))
        values[1:7] = np.nan
        self.assertIsNone(window_median(values,t,-150,-33,4))

    def test_feature_grid_and_missing_event(self):
        times=np.arange(40)*16.6667
        traces={'trunk_tilt':np.linspace(0,39,40)}
        f=features(traces,times,20)
        self.assertEqual(len(f['trunk_tilt_level']),3)
        self.assertGreater(f['trunk_tilt_change'][1],0)
        self.assertTrue(all(v is None for v in features(traces,times,None)['trunk_tilt_level']))
        p=pair_feature([179,None,178],[-179,-178,-177])
        self.assertEqual(p['n'],6); self.assertIsNone(p['nominal']);self.assertEqual(p['stable_direction'],0)
        self.assertEqual(p['grid'][0],2)

    def test_ridge_game_weights_and_origin(self):
        slope, mean = ridge([1,1,3],[2,2,6],['a','a','b'])
        # weights .25,.25,.5; scale=1; numerator=10, denominator=6.
        self.assertAlmostEqual(slope,10/6);self.assertAlmostEqual(mean,4)
        self.assertEqual(ridge([0,0],[2,4],['a','b'])[0],0)

    def fixture(self):
        return [{'id':str(i),'game':g,'delta':{'rpm':y},'features':{'k':pair_feature([0]*3,[x]*3)}}
                for i,(g,x,y) in enumerate([('a',1,2),('a',2,4),('b',3,6),('c',4,8)])]

    def test_game_split_mean_baseline_and_missing_fallback(self):
        pairs=self.fixture(); result=evaluate(pairs,'k','rpm')
        for f in result['folds']:
            self.assertNotIn(f['game'],f['train_games'])
        fold=next(f for f in result['folds'] if f['game']=='c')
        self.assertEqual(fold['predictions'][0]['mean_baseline'],4.5)
        pairs[-1]['features']['k']=pair_feature([None]*3,[4]*3)
        result=evaluate(pairs,'k','rpm')
        # Missing the only pair in c also leaves a/b folds with one supported training game.
        self.assertEqual(result['fallback_count'],4)
        self.assertFalse(result['prediction_candidate'])
        fold=next(f for f in result['folds'] if f['game']=='a')
        self.assertEqual(fold['train_games'],['b'])
        self.assertEqual(fold['candidate_train_games'],['b','c'])
        self.assertEqual(fold['effective_train_pairs'],1)

    def test_repetition_requires_all_games_and_supported_pairs(self):
        pairs=self.fixture();self.assertTrue(repetition(pairs,'k')['descriptive_repeat'])
        pairs[-1]['features']['k']=pair_feature([0]*3,[-1,0,1])
        self.assertFalse(repetition(pairs,'k')['descriptive_repeat'])

    def test_source_clock_map_is_not_frame_name(self):
        original=np.arange(30)*16.6667
        frames,error=map_clock(original[8:18],original)
        np.testing.assert_array_equal(frames,np.arange(8,18));self.assertEqual(error,0)
        with self.assertRaises(ValueError):map_clock(original[[8,10]],original)
        with self.assertRaises(ValueError):map_clock(original[8:18]+1,original)

    def test_read_hash_units_and_sequential_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'source.json';p.write_text(json.dumps({'value':1}),encoding='utf-8')
            self.assertEqual(read(p),{'value':1});self.assertEqual(len(digest(p)),64)
            video=Path(temp)/'clock.mp4'
            w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(64,64))
            self.assertTrue(w.isOpened())
            for i in range(8):w.write(np.full((64,64,3),i*20,np.uint8))
            w.release();t,size=original_clock(video)
            self.assertEqual(size,[64,64]);self.assertEqual(len(t),8)
            self.assertTrue(np.all(np.diff(t)>0))
        actual=outcomes({'release_speed':'100','release_spin_rate':'2400','pfx_z':'1','pfx_x':'-2'})
        self.assertAlmostEqual(actual['speed_kmh'],160.9344)
        self.assertEqual(actual['horizontal_cm'],-60.96)


if __name__ == '__main__':
    unittest.main()
