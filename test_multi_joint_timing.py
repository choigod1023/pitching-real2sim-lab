import unittest
import numpy as np
from multi_joint_timing import shifted_controls, select_case


class TimingCombinationTests(unittest.TestCase):
    def test_combined_shift_preserves_other_inputs_and_excludes_terminal(self):
        loads=np.arange(66,dtype=float).reshape(6,11)
        loads[-1]=1e9
        controls=shifted_controls(loads,(-1,1,-2))
        np.testing.assert_array_equal(controls[:,:3],loads[:,:3])
        np.testing.assert_array_equal(controls[:,7],loads[:,7])
        np.testing.assert_array_equal(controls[-1],loads[-1])
        np.testing.assert_array_equal(controls[:4,3:6],loads[1:5,3:6])
        np.testing.assert_array_equal(controls[1:5,6],loads[:4,6])
        np.testing.assert_array_equal(controls[:3,8:11],loads[2:5,8:11])
        self.assertFalse(np.any(controls[:-1]==1e9))
        self.assertTrue(np.all(np.max(np.abs(controls[:-1]),axis=0)<=np.max(np.abs(loads[:-1]),axis=0)))

    def test_zero_shift_exact(self):
        loads=np.arange(55,dtype=float).reshape(5,11)
        np.testing.assert_array_equal(shifted_controls(loads,(0,0,0)),loads)

    def test_selection_does_not_promote_large_gain_that_fails_either_resolution(self):
        runs=[]
        for sub in (1,2):
            for name,speed,passed in [('s0_e0_w0',100.,True),('good',100.5,True),('bad',110.,sub==2)]:
                runs.append(dict(case_id=name,substeps=sub,speed_kmh=speed,speed_difference_from_baseline_kmh=speed-100.,
                    small_motion_pass=True,budget_gate={'strict_model_budget_pass':passed}))
        self.assertEqual(select_case(runs),('good','accepted_model_candidate',1))

    def test_numerically_unresolved_gain_not_candidate(self):
        runs=[]
        for sub,speed in [(1,99.9),(2,100.01)]:
            runs.extend([dict(case_id='s0_e0_w0',substeps=sub,speed_kmh=100.,speed_difference_from_baseline_kmh=0.,small_motion_pass=True,budget_gate={'strict_model_budget_pass':True}),
                dict(case_id='tiny',substeps=sub,speed_kmh=speed,speed_difference_from_baseline_kmh=speed-100.,small_motion_pass=True,budget_gate={'strict_model_budget_pass':True})])
        self.assertEqual(select_case(runs),('tiny','rejected_diagnostic',0))


if __name__=='__main__': unittest.main()
