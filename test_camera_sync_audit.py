import unittest
import numpy as np
from camera_sync_audit import timestamp_summary, epipolar_distance, shifted_track_audit


class CameraAuditTest(unittest.TestCase):
    def cameras(self):
        k=np.diag([1000.,1000.,1.])
        return [(k,np.column_stack((np.eye(3),np.zeros(3)))),
                (k,np.column_stack((np.eye(3),[-100.,0,0])))]

    def test_timestamp_drop_and_duplicate(self):
        result=timestamp_summary([0,1,2,4,4],1/360)
        self.assertFalse(result['strictly_increasing'])
        self.assertEqual(result['duplicate_pts'],1)
        self.assertEqual(result['intervals_above_1_5_median'],1)

    def test_timestamp_valid(self):
        r=timestamp_summary(range(1407),1/360)
        self.assertEqual(r['frames'],1407)
        self.assertAlmostEqual(r['interval_median_ms'],1000/360)
        self.assertTrue(r['strictly_increasing'])

    def test_epipolar_correspondence(self):
        self.assertAlmostEqual(epipolar_distance([[0,0],[-100,0]],self.cameras()),0)
        self.assertAlmostEqual(epipolar_distance([[0,0],[-100,2]],self.cameras()),2)

    def test_timestamp_invalid_input(self):
        for t in (np.nan,np.inf,0,-1):
            with self.assertRaises(ValueError):timestamp_summary([0,1],t)
        with self.assertRaises(ValueError):timestamp_summary([0,.5],1/360)

    def test_shift_does_not_silently_drop_missing(self):
        tracks={(19,0):{'pixel':[0,0]},(22,0):{'pixel':[-100,0]},
                (19,1):{'pixel':[1,1]},(22,1):{'pixel':[-99,1]}}
        a=shifted_track_audit(tracks,self.cameras(),[0,1],[0,1])
        self.assertTrue(a[0]['all_anchor_frames_available'])
        self.assertFalse(a[1]['all_anchor_frames_available'])
        self.assertNotIn('fit',a[1])
        self.assertFalse(a[0]['rows'][0]['visible_pair'])

    def test_uniform_scale_velocity(self):
        from ball_trajectory_fit import fit_trajectory
        x=np.array([[0.,0,1000],[10,2,1000],[20,4,1000]])
        t=np.arange(3)/360
        v=fit_trajectory(t,x)['midpoint_speed_kmh']
        self.assertAlmostEqual(fit_trajectory(t,1.01*x)['midpoint_speed_kmh'],1.01*v)


if __name__=='__main__':unittest.main()
