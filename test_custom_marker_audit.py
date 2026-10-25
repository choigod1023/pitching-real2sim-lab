import unittest
import numpy as np
from custom_marker_audit import trajectory_stats


class CustomMarkerTests(unittest.TestCase):
    def test_gap_does_not_become_velocity(self):
        p=np.array([[0,1000,2,3],[0,0,0,0],[0,0,0,0],[0,-1,0,0]],dtype=float)
        r=trajectory_stats(p,1000,.001)
        self.assertEqual(r['valid_consecutive_pairs'],1)
        self.assertAlmostEqual(r['max_step_speed_kmh'],3.6)

    def test_missing_track_returns_no_velocity(self):
        p=np.zeros((4,3));p[3]=-1
        self.assertIsNone(trajectory_stats(p,360,1)['max_step_speed_kmh'])
