import unittest
import numpy as np
from pitch_outcomes import flight, evaluate


class FlightTests(unittest.TestCase):
    def test_vacuum_matches_ballistics(self):
        path,t=flight([0,0,1.8],[.2,40,0],cl=0,cd=0,dt=.001)
        self.assertAlmostEqual(t,16.8/40,places=10)
        np.testing.assert_allclose(path[-1,:3],[.2*t,16.8,1.8-9.80665*t*t/2],atol=2e-6)

    def test_backspin_lifts_and_drag_slows(self):
        a,_=flight([0,0,1.8],[0,40,0],cl=.15)
        b,_=flight([0,0,1.8],[0,40,0],cl=0)
        self.assertGreater(a[-1,2],b[-1,2])
        self.assertLess(np.linalg.norm(a[-1,3:]),40)

    def test_gyro_axis_no_lift_without_gravity(self):
        a,_=flight([0,0,1.8],[0,40,0],spin_axis=[0,1,0],cl=.2,gravity=0)
        b,_=flight([0,0,1.8],[0,40,0],cl=0,gravity=0)
        np.testing.assert_allclose(a,b)

    def test_convergence(self):
        a,_=flight([0,0,1.8],[.3,40,0],dt=.004)
        b,_=flight([0,0,1.8],[.3,40,0],dt=.001)
        self.assertLess(np.linalg.norm(a[-1,:3]-b[-1,:3]),1e-4)

    def test_common_noise_identity_and_no_recommendation(self):
        row=dict(name='base',position_m=[0,0,1.8],velocity_m_s=[0,40,0])
        r=evaluate([row,dict(row,name='same')],samples=5)
        self.assertEqual(r['records'][1]['target_error_reduction_cm'],0)
        self.assertEqual(r['records'][0]['endpoints_xz_m'],r['records'][1]['endpoints_xz_m'])
        self.assertFalse(r['human_trial_ready'])

    def test_invalid_and_ground(self):
        for v in ([0,-1,0],[0,1,-30],[0,float('nan'),0]):
            with self.assertRaises(ValueError):flight([0,0,1.8],v)
        with self.assertRaises(ValueError):flight([0,0,1.8],[0,40,0],dt=0)

    def test_failed_sample_not_hidden(self):
        r=evaluate([dict(name='bad',position_m=[0,0,1.8],velocity_m_s=[0,1,-30])])
        self.assertEqual(r['records'][0]['status'],'excluded')

    def test_direction_scenario_does_not_change_speed(self):
        row=dict(name='base',position_m=[0,0,1.8],velocity_m_s=[0,35,-10])
        raw=evaluate([row],samples=2)
        altered=evaluate([row],samples=2,elevation_offset_deg=20)
        self.assertEqual(raw['records'][0]['status'],'excluded')
        self.assertEqual(altered['records'][0]['status'],'ok')
        self.assertAlmostEqual(altered['records'][0]['release_speed_kmh'],np.linalg.norm(row['velocity_m_s'])*3.6)
        self.assertFalse(altered['records'][0]['human_trial_ready'])

    def test_grounded_no_lift_counterfactual_does_not_exclude_lift(self):
        r=evaluate([dict(name='base',position_m=[0,0,1],velocity_m_s=[0,35,0])],cl=.3,samples=2)
        self.assertEqual(r['records'][0]['status'],'ok')
        self.assertIsNone(r['records'][0]['lift_displacement_cm'])


if __name__=='__main__':unittest.main()
