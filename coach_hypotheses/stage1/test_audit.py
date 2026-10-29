from pathlib import Path
import tempfile
import unittest

import numpy as np

from .audit import knee_sign_audit
from .test_metrics import fixture_tables


class AuditTests(unittest.TestCase):
    def test_outcome_free_sign_comparison_and_reversed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'full_sig').mkdir()
            tables = fixture_tables()
            time = tables['joint_angles'].time.to_numpy(float)
            tables['joint_angles']['lead_knee_angle_x'] = 60 - 20*time**2
            tables['joint_velos']['lead_knee_velo_x'] = 40*time
            for sign in (1, -1):
                for name in ('joint_angles', 'joint_velos'):
                    frame = tables[name].copy()
                    frame['session_pitch'] = 'example'
                    if name == 'joint_velos':
                        frame['lead_knee_velo_x'] *= sign
                    frame.to_csv(root/f'full_sig/{name}.csv', index=False)
                result = knee_sign_audit(root).iloc[0]
                self.assertAlmostEqual(result.r_raw_velocity_vs_negative_angle_derivative, sign, places=6)
                smaller = result.mae_raw_vs_extension_dps if sign == 1 else result.mae_negated_raw_vs_extension_dps
                self.assertLess(smaller, 1e-8)
            # No metadata or radar files exist in this synthetic directory.

    def test_nonincreasing_audit_time_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'full_sig').mkdir()
            for name, frame in fixture_tables().items():
                if name not in ('joint_angles', 'joint_velos'):
                    continue
                frame['session_pitch'] = 'example'
                frame.loc[400, 'time'] = .10001
                frame.to_csv(root/f'full_sig/{name}.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'invalid_sign_audit_time'):
                knee_sign_audit(root)


if __name__ == '__main__':
    unittest.main()
