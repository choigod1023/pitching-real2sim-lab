import unittest
import numpy as np
from heldout_speed_comparison import fit_models, predict, evaluate


class HeldoutComparisonTests(unittest.TestCase):
    def setup_case(self, moving):
        t = np.arange(7)/360
        theta = 12*t
        rotations = np.array([[[np.cos(x),-np.sin(x),0],[np.sin(x),np.cos(x),0],[0,0,1]] for x in theta])
        origin = np.column_stack((2*t, t*t, -t))
        local = np.tile([.1,.2,.3],(7,1)) + (t[:,None]*np.array([3.,-2.,1.]) if moving else 0)
        ball = origin+np.einsum('nij,nj->ni',rotations,local)
        return t,ball,origin,rotations

    def test_rigid_case_both_local_models_exact(self):
        t,b,o,r=self.setup_case(False)
        models=fit_models(t[:4],b[:4],o[:4],r[:4])
        predicted=predict(models,t[4:],o[4:],r[4:])
        np.testing.assert_allclose(predicted['A'],b[4:],atol=1e-13)
        np.testing.assert_allclose(predicted['B'],b[4:],atol=1e-13)

    def test_affine_local_motion_forecast_exact(self):
        t,b,o,r=self.setup_case(True)
        models=fit_models(t[:4],b[:4],o[:4],r[:4])
        predicted=predict(models,t[4:],o[4:],r[4:])
        np.testing.assert_allclose(predicted['B'],b[4:],atol=1e-13)
        self.assertGreater(np.linalg.norm(predicted['A']-b[4:]),.01)

    def test_future_ball_labels_cannot_change_fit_or_prediction(self):
        t,b,o,r=self.setup_case(True)
        first=fit_models(t[:4],b[:4],o[:4],r[:4])
        b[4:]+=1000
        second=fit_models(t[:4],b[:4],o[:4],r[:4])
        for name in ('A','B','C'):
            np.testing.assert_array_equal(first[name],second[name])
            np.testing.assert_array_equal(predict(first,t[4:],o[4:],r[4:])[name],predict(second,t[4:],o[4:],r[4:])[name])

    def test_constant_world_velocity_forecast_and_units(self):
        t=np.arange(7)/360
        ball=np.column_stack((30*t,0*t,0*t))
        origins=np.zeros((7,3)); rotations=np.tile(np.eye(3),(7,1,1))
        fitted=fit_models(t[:4],ball[:4],origins[:4],rotations[:4])
        prediction=predict(fitted,t[4:],origins[4:],rotations[4:])['C']
        metrics=evaluate(prediction,ball[4:],1/360,100.)
        self.assertAlmostEqual(metrics['mean_predicted_speed_kmh'],108.)
        self.assertAlmostEqual(metrics['mean_absolute_speed_discrepancy_kmh'],0.)
        self.assertAlmostEqual(metrics['mean_absolute_radar_scalar_discrepancy_kmh'],8.)

    def test_reject_nonfinite_training(self):
        t,b,o,r=self.setup_case(True); b[0,0]=np.nan
        with self.assertRaises(ValueError):
            fit_models(t[:4],b[:4],o[:4],r[:4])


if __name__ == '__main__':
    unittest.main()
