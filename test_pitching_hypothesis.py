import unittest
from collections import Counter
import numpy as np
import pandas as pd
from pitching_hypothesis_test import motion_direction,TARGET,evaluation_membership
from pitching_trial_design import williams_order,assess_daily_contrasts,power_sensitivity,analyze_measurements


class HypothesisTests(unittest.TestCase):
    def test_evaluation_membership_depends_only_on_ids(self):
        np.testing.assert_array_equal(evaluation_membership([2,283,359,400]),[False,True,True,False])

    def test_direction_uses_motion_only_and_reverses_with_pair(self):
        for mask in ('A','B','C','AB','AC','BC','ABC'):
            self.assertEqual(motion_direction(TARGET,mask),1)
            self.assertEqual(motion_direction(-TARGET,mask),-1)

    def test_coupled_body_pattern_and_timing_resolution(self):
        wrong=TARGET.copy();wrong[2]*=-1
        self.assertEqual(motion_direction(wrong,'B'),0)
        tiny=TARGET.copy();tiny[3]=1000/360
        self.assertEqual(motion_direction(tiny,'C'),0)
        self.assertEqual(motion_direction(TARGET*2,'ABC'),0)
        with self.assertRaises(ValueError):motion_direction([1,np.nan,1,1],'A')

    def test_williams_balances_positions_and_every_ordered_transition(self):
        x=williams_order();self.assertEqual(x.shape,(8,8))
        labels=set(x.ravel());self.assertEqual(len(labels),8)
        for row in x:self.assertEqual(set(row),labels)
        for col in x.T:self.assertEqual(set(col),labels)
        counts=Counter((a,b) for row in x for a,b in zip(row[:-1],row[1:]))
        self.assertEqual(len(counts),56);self.assertEqual(set(counts.values()),{1})
        np.testing.assert_array_equal(x,williams_order())

    def test_non_significant_load_difference_is_not_proof_of_no_increase(self):
        x=np.column_stack((np.full(8,3.),np.tile([-1,1],4),np.tile([-1,1],4)))
        r=assess_daily_contrasts(x)
        self.assertFalse(r['limited_proxy_joint_criterion_pass']);self.assertGreater(r['upper'][1],0)

    def test_joint_criterion_and_minimum_meaningful_gain(self):
        base=np.array([[-.1,.1,-.1],[.1,-.1,.1],[0,0,0]])
        self.assertTrue(assess_daily_contrasts(base+[3,-5,-5])['limited_proxy_joint_criterion_pass'])
        self.assertFalse(assess_daily_contrasts(np.tile([1.,-5,-5],(8,1)))['limited_proxy_joint_criterion_pass'])
        self.assertFalse(assess_daily_contrasts(base+[3,-5,-5])['full_strength_work_hypothesis_verified'])

    def test_malformed_and_nonfinite_trials_rejected(self):
        for x in (np.ones((2,3)),np.ones((8,2)),np.full((8,3),np.nan)):
            with self.assertRaises(ValueError):assess_daily_contrasts(x)

    def test_simulation_is_not_actual_trial_and_strong_scenario_can_be_detected(self):
        p=power_sensitivity([1,1,1],24,[4,-5,-5],repetitions=1000)
        self.assertGreater(p,.99)
        self.assertEqual(power_sensitivity([1,1,1],24,[-5,5,5],repetitions=1000),0)

    def test_measurement_analysis_does_not_count_repeated_pitches_as_independent_days(self):
        rows=[]
        for day in range(3):
            for cond in ['000','111']:
                for pitch in range(4):
                    active=cond=='111'
                    rows.append(dict(athlete_id='p',day_id=day,paired_block_id=1,condition=cond,
                                     pitch_id=f'{cond}_{pitch}',measurement_qc='pass',
                                     speed_kmh=130+day+3*active,elbow_peak_nm=100-5*active,shoulder_peak_nm=100-5*active))
        f=pd.DataFrame(rows);r=analyze_measurements(f)
        self.assertEqual(r['independent_days_assumed'],3);self.assertEqual(r['complete_blocks'],3)
        self.assertEqual(r['inference']['mean'],[3.,-5.,-5.])
        self.assertFalse(r['muscle_force_total_work_and_causal_hypothesis_verified'])
        with self.assertRaises(ValueError):analyze_measurements(pd.concat([f,f.iloc[:1]]))
        f['day_id']=0;f['pitch_id']=[str(i) for i in range(len(f))]
        self.assertIsNone(analyze_measurements(f)['inference'])


if __name__=='__main__':unittest.main()
