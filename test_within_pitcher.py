import unittest
import numpy as np
import pandas as pd
from within_pitcher import evaluate_changes


class WithinPitcherChecks(unittest.TestCase):
    def test_change_is_independent_of_between_athlete_offset(self):
        frame=pd.DataFrame({'mode':['raw']*4,'user':[1,1,2,2],'session_pitch':['b','a','c','d'],
                            'observed_kmh':[102.,100.,120.,123.],
                            'predicted_kmh':[92.,90.,110.,113.],
                            'scalar_linear_kmh':[92.,90.,110.,113.],
                            'zero_kmh':[92.,90.,110.,113.]})
        rows,summary=evaluate_changes(frame)
        self.assertTrue(rows[rows.method=='predicted'].absolute_delta_error_kmh.eq(0).all())
        self.assertTrue(rows[(rows.user==1)].reference_pitch.eq('a').all())
        frame.loc[frame.user==1,['observed_kmh','predicted_kmh','scalar_linear_kmh','zero_kmh']]+=100
        _,shifted=evaluate_changes(frame)
        self.assertEqual(summary,shifted)
        baseline=[r for r in summary if r['method']=='no_change'][0]
        self.assertAlmostEqual(baseline['athlete_balanced_delta_mae_kmh'],2.5)
