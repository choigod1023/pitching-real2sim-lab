import unittest
import numpy as np
import pandas as pd
from hand_geometry import release_times
from timing_stability import stability
from timing_calibration import evaluate


class TimingTests(unittest.TestCase):
    def test_frame_to_time_and_no_extrapolation(self):
        t=np.arange(100)/360
        np.testing.assert_allclose(release_times(.1,360,t,[-2,0,2]),[.1-2/360,.1,.1+2/360])
        with self.assertRaises(ValueError):release_times(0,360,t,[-2,0,2])
        with self.assertRaises(ValueError):release_times(.1,360,t,[0,.5])

    def test_independent_shifts_can_flip_when_common_shifts_do_not(self):
        rows=[dict(user=1,pitch=p,method='test',offset_frames=k,predicted_kmh=100+k+d)
              for p,d in [('a',0),('b',1)] for k in [-2,-1,0,1,2]]
        frame=pd.DataFrame(rows);r=stability(frame)[0]
        self.assertEqual(r['common_sign_flip_count'],0)
        self.assertEqual(r['independent_sign_flip_count'],1)
        with self.assertRaises(ValueError):stability(frame.iloc[1:])

    def test_heldout_targets_do_not_change_heldout_slope(self):
        rows=[dict(user=u,pitch=p,method='test',offset_frames=0,predicted_kmh=100+d,
                   observed_kmh=130+u*d) for u in [1,2,3] for p,d in [('a',0),('b',1)]]
        frame=pd.DataFrame(rows);before=evaluate(frame)[0]['folds'][0]['slope']
        frame.loc[frame.user.eq(1),'observed_kmh']=[1,1000]
        self.assertEqual(evaluate(frame)[0]['folds'][0]['slope'],before)
