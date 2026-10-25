import unittest
import pandas as pd
import numpy as np
from cohort_baseline import select_trials,heldout_mean,summarize


class CohortTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame({'user':[1,1,2,2],'pitch':['a','b','c','d'],
                             'cutoff_hz':[20.]*4,'observed_kmh':[100.,102.,120.,123.],
                             'predicted_kmh':[90.,92.,110.,113.]})

    def test_person_offsets_do_not_change_delta_error(self):
        a=self.data();before=summarize(a)[0]
        a.loc[a.user==1,'predicted_kmh']+=27
        self.assertEqual(before['change_mae_kmh'],summarize(a)[0]['change_mae_kmh'])
        self.assertEqual(before['change_mae_kmh'],0.)
        self.assertEqual(before['no_change_mae_kmh'],2.5)

    def test_mean_baseline_excludes_evaluated_athlete(self):
        a=self.data();before=heldout_mean(a)
        a.loc[a.user==1,'observed_kmh']+=1000
        np.testing.assert_array_equal(before[a.user==1],heldout_mean(a)[a.user==1])

    def test_selection_ignores_speeds_and_input_order(self):
        a=self.data().rename(columns={'pitch':'session_pitch'});a['p_throws']='R'
        first=select_trials(a,1).session_pitch.tolist()
        a['observed_kmh']=[999,0,-3,700]
        self.assertEqual(first,select_trials(a.iloc[::-1],1).session_pitch.tolist())
