import unittest
import numpy as np
from quantization_reconstruction import smooth_rounding_bins

class TestQuantizationReconstruction(unittest.TestCase):
    def test_stays_inside_bins_and_reduces_acceleration_error(self):
        t=np.arange(240)/360.;truth=.25*np.sin(2*np.pi*2*t)+.4*t
        rounded=np.round(truth,4);restored,info=smooth_rounding_bins(rounded);restored=restored[:,0]
        self.assertTrue(info['converged']);self.assertLessEqual(np.max(abs(restored-rounded)),.00005+1e-12)
        exact=-.25*(2*np.pi*2)**2*np.sin(2*np.pi*2*t)
        accel=lambda p:np.gradient(np.gradient(p,t),t)
        before=np.mean((accel(rounded)[5:-5]-exact[5:-5])**2);after=np.mean((accel(restored)[5:-5]-exact[5:-5])**2)
        self.assertLess(after,before/3)
    def test_constant_signal_unchanged(self):
        values=np.full((80,3),.1234);out,info=smooth_rounding_bins(values);np.testing.assert_allclose(out,values,atol=1e-12)
    def test_invalid_samples_rejected(self):
        for x in [np.ones(4),np.array([0,1,np.nan,2,3])]:
            with self.assertRaises(ValueError):smooth_rounding_bins(x)

if __name__=='__main__':unittest.main()
