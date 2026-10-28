import unittest
import numpy as np
from contact_sensitivity_audit import decompose


class SensitivityAuditTests(unittest.TestCase):
    def rows(self):
        return [dict(stiffness=k,release_ms=t,baseline=dict(speed_kmh=10*k+2*t)) for k in [1,2] for t in [1,2]]

    def test_additive_components_and_ranges(self):
        r=decompose(self.rows())
        np.testing.assert_allclose(r['squared_deviation_components'],[100,4,0])
        np.testing.assert_allclose(r['stiffness_range_at_fixed_release_kmh'],[10,10])
        np.testing.assert_allclose(r['release_range_at_fixed_stiffness_kmh'],[2,2])

    def test_reject_incomplete_and_duplicate_grid(self):
        rows=self.rows()
        for bad in [rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):decompose(bad)

    def test_nonfinite_and_constant(self):
        rows=self.rows();rows[0]['baseline']['speed_kmh']=float('nan')
        with self.assertRaises(ValueError):decompose(rows)
        for r in rows:r['baseline']['speed_kmh']=5
        self.assertEqual(decompose(rows)['grid_variation_percent'],[0,0,0])


if __name__=='__main__':unittest.main()
