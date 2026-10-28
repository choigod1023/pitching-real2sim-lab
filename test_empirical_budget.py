import unittest
import numpy as np
from empirical_budget_search import group_costs,metrics
class GroupBudgetTests(unittest.TestCase):
 def test_dot_power_allows_axis_redistribution(self):
  f=np.zeros((2,11));v=np.zeros((2,11));f[:,3:6]=[1,1,0];v[:,3:6]=[1,-1,0]
  r=group_costs({},f,v,1.)
  self.assertEqual(r['group_positive_work_j'][1],0)
  self.assertEqual(r['group_negative_work_j'][1],0)
 def test_group_power_and_peak_invariant_under_common_rotation(self):
  f=np.zeros((2,11));v=np.zeros((2,11));f[:,3:6]=[2,3,0];v[:,3:6]=[4,1,0]
  a=group_costs({},f,v,.1);rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
  f[:,3:6]=f[:,3:6]@rot;v[:,3:6]=v[:,3:6]@rot;b=group_costs({},f,v,.1)
  np.testing.assert_allclose(a['group_positive_work_j'],b['group_positive_work_j'])
  np.testing.assert_allclose(a['group_peak_loads'],b['group_peak_loads'])
 def test_unused_terminal_force_excluded(self):
  f=np.zeros((2,11));v=np.ones((2,11));f[-1]=1e9
  self.assertEqual(sum(group_costs({},f,v,.1)['group_peak_loads']),0)
if __name__=='__main__':unittest.main()
