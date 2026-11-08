import unittest
import numpy as np
from analyze_new_throw_videos import bright_component,frame_slope
from read_paired_source_index import speed_label,require_same_trial


class PairedSourceTests(unittest.TestCase):
    def test_speed_label_is_label_not_measured_reference(self):
        r=speed_label('Baseball Throw - 62.7 MPH')
        self.assertAlmostEqual(r['kmh'],100.9058688)
        self.assertIn('not instrument',r['kind'])
        with self.assertRaises(ValueError):speed_label('unreported')

    def test_different_camera_system_trials_are_not_joined(self):
        require_same_trial('edge67','edge67')
        with self.assertRaises(ValueError):require_same_trial('Prime62.7','edge67')

    def test_image_slope_keeps_frame_units_and_no_kmh(self):
        f=np.arange(375,380);p=np.column_stack([200-3*(f-375),500+4*(f-375)])
        r=frame_slope(f,p);self.assertAlmostEqual(r['speed_px_per_frame'],5.)
        self.assertIsNone(r['speed_kmh'])
        with self.assertRaises(ValueError):frame_slope([0,0,1],p[:3])

    def test_complete_bright_ball_and_boundary_failure_are_distinct(self):
        image=np.zeros((200,220,3),np.uint8);image[90:110,75:125]=240
        r=bright_component(image,[100,100],160);self.assertEqual(r['status'],'candidate')
        np.testing.assert_allclose(r['pixel'],[99.5,99.5]);self.assertEqual(r['bbox'][2],50)
        clipped=bright_component(image,[55,100],160);self.assertEqual(clipped['status'],'roi_boundary')
        self.assertEqual(bright_component(np.zeros_like(image),[100,100],160)['status'],'no_component')


if __name__=='__main__':unittest.main()
