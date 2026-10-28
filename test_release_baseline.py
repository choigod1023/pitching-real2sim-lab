import unittest
import numpy as np
import pandas as pd
from release_baseline import evaluate, ridge_predict


class ReleaseChecks(unittest.TestCase):
    def test_ridge_constant_feature_and_target_shift(self):
        x = np.array([[1., 1.], [2., 1.], [3., 1.]])
        y = np.array([10., 20., 30.])
        pred = ridge_predict(x, y, x)
        self.assertTrue(np.isfinite(pred).all())
        np.testing.assert_allclose(ridge_predict(x, y+50, x), pred+50)

    def test_heldout_target_does_not_change_predictions(self):
        rows = pd.DataFrame({'user': [1,1,2,2,3,3], 'session_pitch': list('abcdef'),
                            'observed_ball_kmh': [100.,105.,110.,115.,120.,125.],
                            'hand_release_kmh': [80.,85.,90.,95.,100.,105.],
                            'wrist_release_kmh': [70.,75.,80.,85.,90.,95.],
                            'shoulder_release_kmh': [10.]*6, 'session_height_m': [1.8]*6,
                            'session_mass_kg': [80.]*6})
        before, _ = evaluate(rows)
        rows.loc[rows.user==1, 'observed_ball_kmh'] += 999
        after, _ = evaluate(rows)
        np.testing.assert_allclose(before[before.user==1].predicted_ball_kmh, after[after.user==1].predicted_ball_kmh)
