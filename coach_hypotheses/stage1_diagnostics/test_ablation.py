import unittest

import numpy as np

from .ablation import matched_penalty
from .test_diagnostics import diagnostic_fixture


class AblationTests(unittest.TestCase):
    def test_all_fixed_penalties_reported_without_loss_unit_mixing(self):
        result=matched_penalty(diagnostic_fixture())
        self.assertEqual(len(result),8)
        self.assertNotIn('mae14',result.family.tolist())
        self.assertEqual(sorted(result.fixed_alpha.unique()),[100.,1000.])
        self.assertTrue(np.isfinite(result.mae_kmh).all())


if __name__=='__main__':unittest.main()
