import unittest
import numpy as np
from personal_calibration import fit_population,predict


class PersonalTests(unittest.TestCase):
    def test_calibration_intercept_invariance(self):
        x=np.array([[0.],[1.],[2.]])
        m=fit_population([(x,np.array([0.,2.,4.]))],1)
        p=predict(m,x[:2],[0.,2.],x[2:],1)
        shifted=predict(m,x[:2],[100.,102.],x[2:],1)
        np.testing.assert_allclose(shifted,p+100)

    def test_constant_features_cannot_fake_motion_effect(self):
        x=np.zeros((3,2));m=fit_population([(x,np.array([0.,1.,2.]))],1)
        np.testing.assert_allclose(predict(m,x[:2],[10.,12.],x[2:],1),[11.])

    def test_individual_slope_helps_known_different_slope(self):
        m=(np.ones(1),np.array([1.]));x=np.array([[-1.],[1.]])
        base=predict(m,x,[-3.,3.],[[2.]])
        adapted=predict(m,x,[-3.,3.],[[2.]],1)
        self.assertLess(abs(adapted[0]-6),abs(base[0]-6))
        np.testing.assert_array_equal(m[1],[1.])
        with self.assertRaises(ValueError):predict(m,x,[-3.,3.],[[2.]],0)


if __name__=='__main__':unittest.main()
