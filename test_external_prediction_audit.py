import unittest
import numpy as np
from external_prediction_audit import score_arrays,check_labels


class ExternalPredictionTests(unittest.TestCase):
    def test_athlete_bias_cancels_in_pair_differences(self):
        y=np.array([[100.,101.,102.],[130.,132.,134.]])
        r=score_arrays(y,y+np.array([[10.],[-10.]]))
        self.assertAlmostEqual(r['absolute_mae_kmh'],10.)
        self.assertAlmostEqual(r['delta_mae_kmh'],0.)
        self.assertEqual(r['non_tie']['pairs'],6)

    def test_constant_prediction_equals_no_change(self):
        y=np.array([[100.,100.,102.],[130.,132.,134.]])
        r=score_arrays(y,np.full_like(y,120.))
        self.assertAlmostEqual(r['mae_difference_kmh'],0.)
        self.assertEqual(r['difference_bootstrap95'],[0.,0.])
        self.assertEqual(r['observed_tie_pairs'],1)

    def test_mismatched_label_order_rejected(self):
        check_labels([1.,2.,3.],[1.,2.,3.000001])
        with self.assertRaises(ValueError):check_labels([1.,3.,2.],[1.,2.,3.])
        with self.assertRaises(ValueError):score_arrays([[1.,float('nan')],[2.,3.]],[[1.,2.],[2.,3.]])
