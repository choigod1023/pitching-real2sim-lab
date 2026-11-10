from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.optical_axis_flight.study import initial_state, simulate
from experiments.pitcher_active_spin_flight.study import match_efficiency, read_efficiency


class PitcherActiveSpinTests(unittest.TestCase):
    def test_reads_measured_ratio_and_counts_only_above_threshold(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaderboard.csv"
            path.write_text(
                "year,api_pitch_type,n_pitches,player_id,active_spin,alan_active_spin_pct\n"
                "2024,FF,30,7,.82,.41\n2024,FF,29,9,.99,.21\n", encoding="utf-8"
            )
            self.assertEqual(read_efficiency(path, 2024), ({7: (.82, 30)}, 2))
            with self.assertRaises(ValueError):
                read_efficiency(path, 2025)

    def test_pitcher_match_keeps_common_rate_and_marks_fallback(self):
        rows = [{"key": (1, 7, 3, 1)}, {"key": (1, 7, 4, 2)}, {"key": (1, 9, 4, 1)}]
        rates, known = match_efficiency(rows, {7: (.82, 30)})
        np.testing.assert_array_equal(rates, [.82, .82, 1.])
        np.testing.assert_array_equal(known, [True, True, False])
        self.assertEqual(np.sign(2300 * rates[0] - 2350 * rates[1]), -1)

    def test_prior_only_changes_flight_not_release(self):
        row = dict(release_speed=95., release_spin_rate=2300., spin_axis=215., release_pos_x=-2.,
                   release_pos_y=54., release_pos_z=6., vx0=7., vy0=-137., vz0=-6.)
        rows = [row, dict(row, spin_axis=220.)]
        launch = initial_state(rows, 1.)[:3]
        before = simulate(rows, True, keep=(0, 1))
        after = simulate(rows, True, np.array([.8, 1.]), fixed_launch=launch, keep=(0, 1))
        np.testing.assert_array_equal(before[3][0], after[3][0])
        np.testing.assert_array_equal(before[3][:, 1], after[3][:, 1])
        self.assertLess(after[0][0], before[0][0])
        self.assertLess(after[1][0], before[1][0])
