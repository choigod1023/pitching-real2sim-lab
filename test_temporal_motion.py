import unittest
import numpy as np
import pandas as pd
from temporal_motion_gate import extract,JOINTS

class TemporalTest(unittest.TestCase):
    def frame(self):
        t=np.linspace(0,1,11);d=dict(time=t,fp_10_time=np.full(11,.2),BR_time=np.full(11,.8))
        for j in JOINTS:
            for c,v in zip('xyz',[3,4,0]):d[j+'_velo_'+c]=np.full(11,v,dtype=float)
        return pd.DataFrame(d)
    def test_known_norm_and_duration(self):
        z=extract(self.frame());self.assertEqual(len(z),37);np.testing.assert_allclose(z[:5],5);self.assertAlmostEqual(z[-1],600)
    def test_no_extrapolation_or_missing_fill(self):
        g=self.frame();g['BR_time']=2
        with self.assertRaises(ValueError):extract(g)
        g=self.frame();g.loc[5,'wrist_velo_x']=np.nan
        with self.assertRaises(ValueError):extract(g)

if __name__=='__main__':unittest.main()
