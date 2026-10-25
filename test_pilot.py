import unittest
import numpy as np
import pandas as pd
from pilot import change,speed_at,regular_time

class CounterfactualChecks(unittest.TestCase):
    def setUp(self):
        self.t=np.linspace(0,1,361)
        self.s=np.zeros((361,3));self.e=np.tile([0,0,.3],(361,1))
        self.w=self.e+np.column_stack([.3*np.cos(self.t*3),.3*np.sin(self.t*3),np.zeros(361)])
        self.h=self.e+1.2*(self.w-self.e)
    def test_identity(self):
        w,h=change(self.t,self.s,self.e,self.w,self.h,.5,0,0)
        np.testing.assert_allclose(w,self.w,atol=1e-12);np.testing.assert_allclose(h,self.h,atol=1e-12)
    def test_geometry_preserved(self):
        w,h=change(self.t,self.s,self.e,self.w,self.h,.5,5,10)
        for a,b in [(w-self.e,self.w-self.e),(h-self.e,self.h-self.e),(h-w,self.h-self.w)]:
            np.testing.assert_allclose(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),atol=1e-12)
    def test_constant_velocity_and_translation(self):
        p=np.column_stack([3*self.t,4*self.t,np.zeros(361)])
        self.assertAlmostEqual(speed_at(self.t,p,.5),5)
        self.assertAlmostEqual(speed_at(self.t,p+10,.5),5)
    def test_rounded_timestamps_and_missing_frame(self):
        nominal=np.arange(100)/360
        df=pd.DataFrame({'time':nominal.round(4)})
        np.testing.assert_allclose(regular_time(df),nominal)
        with self.assertRaises(ValueError):regular_time(df.drop(index=50))

if __name__=='__main__':unittest.main()
