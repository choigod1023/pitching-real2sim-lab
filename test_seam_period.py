import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.linked_highspeed_release.seam_period import analyze_clip, estimate_period, track_ball


class SeamPeriodTests(unittest.TestCase):
    def test_estimate_period_distinguishes_two_synthetic_recurrences(self):
        yy, xx = np.mgrid[-1.1:1.1:48j, -1.1:1.1:48j]
        distances = xx**2 + yy**2
        times = np.arange(84)[:, None, None]
        texture1 = np.sin(3 * xx) * np.cos(2 * yy)
        texture2 = np.cos(2 * xx) * np.sin(3 * yy)
        for period in (13, 19):
            with self.subTest(period=period):
                phase = (2 * np.pi / period) * times
                chips = 127 + np.cos(phase) * texture1 + np.sin(phase) * texture2
                specs = estimate_period(chips, distances)
                self.assertEqual(len(specs), 15)
                self.assertEqual({spec['integer_period_frames'] for spec in specs}, {period})
                self.assertTrue(all(spec['peak_integer_lag_correlation'] > 0.99 for spec in specs))

    def test_estimate_period_rejects_textureless_input(self):
        yy, xx = np.mgrid[-1.1:1.1:48j, -1.1:1.1:48j]
        with self.assertRaisesRegex(ValueError, 'no visible texture variation'):
            estimate_period(np.ones((84, 48, 48)), xx**2 + yy**2)

    def test_track_ball_reports_unreadable_clip(self):
        with TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'Cannot open video'):
            track_ball(Path(folder) / 'missing.mp4', 4, (832, 626, 31))

    def test_analyze_clip_rejects_invalid_capture_rate(self):
        with self.assertRaisesRegex(ValueError, 'Nominal capture FPS'):
            analyze_clip(Path('missing.mp4'), 4, (832, 626, 31), 0)


if __name__ == '__main__':
    unittest.main()
