import unittest
import numpy as np
import pandas as pd
from within_athlete_candidates import extract,comparisons,FEATURES,interval

class TestWithinAthleteCandidates(unittest.TestCase):
    def test_peak_is_relative_to_release_and_first_tie(self):
        f=pd.DataFrame({'time':[1.,1.1,1.2,1.3], 'fp_10_time':1.1,'BR_time':1.3})
        for c in ['pelvis_velo_z','torso_velo_z','elbow_velo_x','shoulder_velo_z','wrist_velo_x','wrist_velo_y','wrist_velo_z']:f[c]=[100,2,-4,4]
        r=extract(f)
        self.assertAlmostEqual(r['elbow_peak_ms'],-100)
        self.assertAlmostEqual(r['fp_to_br_ms'],200)
        self.assertAlmostEqual(r['wrist_peak_ms'],-100)

    def test_no_cross_session_or_athlete_pairing_and_equal_speeds_excluded(self):
        rows=[]
        for user,session,speeds in [(2,10,[80,82,81]),(2,11,[99]),(3,12,[70,70])]:
            for i,speed in enumerate(speeds):
                rows.append(dict(user=user,session=session,session_pitch=f'{session}_{i}',pitch_type='FF',split='discovery',pitch_speed_mph=speed,**{f:float(i) for f in FEATURES}))
        extreme,pairs,excluded=comparisons(pd.DataFrame(rows))
        self.assertEqual(len(extreme),1);self.assertEqual(len(pairs),3);self.assertEqual(len(excluded),2)
        self.assertEqual(extreme.iloc[0].fast_pitch,'10_1')
        self.assertAlmostEqual(extreme.iloc[0].delta_kmh,3.218688)
        self.assertEqual(extreme.iloc[0].elbow_peak_ms,1.)

    def test_invalid_events_and_bootstrap_repeatability(self):
        f=pd.DataFrame({'time':[0,1,2],'fp_10_time':2,'BR_time':1})
        with self.assertRaises(ValueError):extract(f)
        self.assertEqual(interval([1,2,3]),interval([1,2,3]))
        self.assertEqual(interval([4,4]),[4.,4.])

if __name__=='__main__':unittest.main()
