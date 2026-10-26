import unittest
import numpy as np
from physical_release_comparison import launch_states, free_flight


class PhysicalComparisonTests(unittest.TestCase):
    def test_flight_matches_analytic_at_both_steps(self):
        p=np.array([1.,2.,3.]);v=np.array([30.,-2.,1.]);g=np.array([1.,-3.,-9.])
        times=np.arange(1,4)/360
        expected=p+times[:,None]*v+.5*times[:,None]**2*g
        for substeps in [1,2]:
            positions,velocities=free_flight(p,v,g,substeps=substeps)
            np.testing.assert_allclose(positions,expected,atol=2e-14)
            np.testing.assert_allclose(velocities,v+times[:,None]*g,atol=2e-14)

    def test_training_only_states_and_original_inputs_preserved(self):
        times=np.arange(7)/360
        ball=times[:,None]*np.array([30.,1.,2.])
        origin=np.zeros((7,3));rotation=np.tile(np.eye(3),(7,1,1))
        states=launch_states(times[:4],ball[:4],origin[:4],rotation[:4])
        ball[4:]=1e6;origin[4:]=-1e6;rotation[4:]=1e6
        modified=launch_states(times[:4],ball[:4],origin[:4],rotation[:4])
        for name in states:
            np.testing.assert_array_equal(states[name][0],modified[name][0])
            np.testing.assert_array_equal(states[name][1],modified[name][1])
        np.testing.assert_allclose(states['B'][1],[30.,1.,2.],atol=1e-12)
        np.testing.assert_allclose(states['C'][1],[30.,1.,2.],atol=1e-12)
        np.testing.assert_allclose(states['A'][1],0.)


if __name__=='__main__':unittest.main()
