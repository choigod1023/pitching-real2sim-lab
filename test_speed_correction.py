import unittest
import numpy as np
import pandas as pd
from speed_correction import predict,balanced,evaluate,FEATURES
from expand_speed_check import scores


class CorrectionTests(unittest.TestCase):
    def test_bias_cannot_improve_changes(self):
        f=pd.DataFrame(dict(user=[1,1,2,2,2],pitch=['a','b','c','d','e'],
                            observed_kmh=[100,102,110,111,115],raw_kmh=[80,81,90,94,95]))
        f['bias_kmh']=f.raw_kmh+20;f['training_mean_kmh']=108.
        s=scores(f)
        self.assertAlmostEqual(s['raw_kmh_change_mae_kmh'],s['bias_kmh_change_mae_kmh'])
        self.assertAlmostEqual(s['bias_kmh_absolute_mae_kmh'],.75)

    def test_infinite_penalty_is_baseline(self):
        x=np.arange(12.).reshape(6,2);y=np.arange(6.);u=np.array([0,0,1,1,1,1])
        np.testing.assert_allclose(predict(x,y,u,x,np.inf,False),0)
        np.testing.assert_allclose(predict(x,y,u,x,np.inf,True),balanced(y,u))

    def test_linear_recovery(self):
        x=np.array([[1.],[2.],[4.],[5.]])
        np.testing.assert_allclose(predict(x,3*x[:,0]+7,np.arange(4),np.array([[9.]]),0.,True),34,atol=1e-10)

    def test_heldout_labels_do_not_change_predictions(self):
        rows=[]
        for user in range(4):
            for trial in range(3):
                rows.append(dict(user=user,pitch=f'{user}_{trial}',observed_kmh=100+user+trial,integrated_ball_kmh=80+user+trial,**{f:2+user*(j+1)+trial**(j+1) for j,f in enumerate(FEATURES)}))
        d=pd.DataFrame(rows);a,_=evaluate(d)
        d.loc[d.user.eq(0),'observed_kmh']+=np.array([17.,-8.,42.])
        b,_=evaluate(d)
        np.testing.assert_allclose(a[a.user.eq(0)].predicted,b[b.user.eq(0)].predicted,atol=0)


if __name__=='__main__':unittest.main()
