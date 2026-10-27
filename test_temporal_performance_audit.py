import unittest
import numpy as np
from temporal_performance_audit import phase_samples


class TestPhase(unittest.TestCase):
    def test_signed_linear_motion_and_unused_missing_values(self):
        t=np.arange(11,dtype=float)
        v=np.column_stack([2*t,-3*t]);v[0]=np.nan;v[-1]=np.nan
        result=phase_samples(t,v,2,8)
        np.testing.assert_allclose(result[:,0],2*np.linspace(2,8,21))
        np.testing.assert_allclose(result[:,1],-3*np.linspace(2,8,21))

    def test_missing_support_or_extrapolation_rejected(self):
        t=np.arange(11,dtype=float);v=t[:,None].copy();v[4]=np.nan
        with self.assertRaises(ValueError):phase_samples(t,v,2,8)
        with self.assertRaises(ValueError):phase_samples(t,np.zeros((11,1)),-1,8)


if __name__=='__main__':unittest.main()
