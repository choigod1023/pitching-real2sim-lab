import unittest
import numpy as np
from paired_motion_baseline import outer_fold,train,balanced

class PairedTests(unittest.TestCase):
    def test_holdout_target_cannot_change_fit(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(24,3));u=np.repeat(np.arange(8),3);y=x@np.array([1.,2.,-1.])
        p,a=outer_fold(x,y,u,7);y[u==7]+=10000;q,b=outer_fold(x,y,u,7)
        np.testing.assert_array_equal(p,q);self.assertEqual(a,b)
    def test_linear_recovery(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(30,2));u=np.repeat(np.arange(10),3);coef=train(x,x@np.array([2.,-1.]),u,1e-10)
        np.testing.assert_allclose(coef,[2,-1],atol=1e-8)
    def test_balanced_and_zero_features(self):
        self.assertAlmostEqual(balanced(np.array([2,2,2,10]),np.array([1,1,1,2])),6)
        np.testing.assert_array_equal(train(np.zeros((4,2)),np.ones(4),np.array([1,1,2,2]),1),[0,0])

if __name__=='__main__':unittest.main()
