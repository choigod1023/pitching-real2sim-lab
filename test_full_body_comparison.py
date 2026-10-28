import unittest
import numpy as np
import pandas as pd
from full_body_comparison import sample,peak,pair_summary

class TestFullBodyComparison(unittest.TestCase):
    def test_no_silent_extrapolation_and_vector_interpolation(self):
        np.testing.assert_allclose(sample([0,1],[[0,2],[2,4]],[.5]),[[1,3]])
        for q in [-.01,1.01]:
            with self.assertRaises(ValueError):sample([0,1],[0,1],q)
    def test_peak_keeps_signed_value_and_excludes_outside_event(self):
        self.assertEqual(peak(np.array([0,.1,.2,.3]),[100,-3,2,200],.1,.2),(.1,-3.))
        with self.assertRaises(ValueError):peak(np.array([0,1]),[1,2],.2,.3)
    def test_rounding_differences_not_counted_as_timing_evidence(self):
        f=pd.DataFrame([dict(session_pitch='a',speed_kmh=120,peak_ms=10),dict(session_pitch='b',speed_kmh=121,peak_ms=10.03),dict(session_pitch='c',speed_kmh=122,peak_ms=16)])
        r=pair_summary(f,['peak_ms'],'c','a')[0]
        self.assertEqual(r['same_direction_pairs'],2);self.assertEqual(r['unresolved_pairs'],1)
    def test_common_time_anchor_preserves_duration(self):
        t=np.linspace(0,1,361);p=np.column_stack([t,t*2,t*0]);fp=.4;br=.55
        np.testing.assert_allclose(sample(t,p,fp+.1)-sample(t,p,fp),[.1,.2,0])
        self.assertAlmostEqual((br-fp)*1000,150)

if __name__=='__main__':unittest.main()
