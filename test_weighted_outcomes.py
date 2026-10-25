import unittest
import numpy as np
import pandas as pd
from weighted_outcomes import analyze,paired_interval


class OutcomeTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame([dict(subject=s,ball_weight=f'{w}oz',pitch_velocity=100+s-w)
                             for s in (1,2,3) for w in (3,4,5,6,7)])

    def test_pairing_units_and_reference_are_correct(self):
        r=analyze(self.data());c=[x for x in r['contrasts'] if x['weight_oz']==3][0]
        self.assertAlmostEqual(c['mean_difference'],2*1.609344)
        np.testing.assert_allclose(c['bootstrap_percentile_95'],[2*1.609344]*2)

    def test_repeated_identical_rows_do_not_reweight_athlete_mean(self):
        d=self.data();before=analyze(d)
        after=analyze(pd.concat([d,d[d.subject.eq(1)]]).sample(frac=1,random_state=1))
        self.assertEqual(before['contrasts'],after['contrasts'])

    def test_missing_outcome_rejected_and_bootstrap_deterministic(self):
        d=self.data();d.loc[0,'pitch_velocity']=np.nan
        with self.assertRaises(ValueError):analyze(d)
        self.assertEqual(paired_interval([1,2,5]),paired_interval([1,2,5]))
