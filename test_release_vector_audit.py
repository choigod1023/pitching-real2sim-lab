import unittest
import numpy as np
from release_vector_audit import field_frame,vector_metrics
from audit_prime_vectors import compare_vectors


class VectorTests(unittest.TestCase):
    def test_documented_frame(self):
        p,v=field_frame([2,3,1.7],[40,2,-1],[1,2])
        np.testing.assert_allclose(p,[-1,1,1.7]);np.testing.assert_allclose(v,[-2,40,-1])
        self.assertAlmostEqual(np.linalg.norm(v),np.linalg.norm([40,2,-1]))

    def test_common_frame_keeps_sideways_changes(self):
        _,a=field_frame([0,0,1.8],[40,0,0],[0,0])
        _,b=field_frame([0,0,1.8],[40,1,0],[0,0])
        self.assertEqual(b[0]-a[0],-1)

    def test_elevation_is_not_speed(self):
        a=vector_metrics([30,0,-10]);b=vector_metrics([30,0,10])
        self.assertEqual(a['speed_kmh'],b['speed_kmh'])
        self.assertEqual(a['elevation_deg'],-b['elevation_deg'])

    def test_equal_speed_can_have_large_direction_error(self):
        a=np.array([[0,0,0],[1,0,0],[2,0,0]])
        b=np.array([[0,0,0],[0,1,0],[0,2,0]])
        r=compare_vectors(a,b,1)
        self.assertEqual(r['speed_mae_kmh'],0)
        self.assertAlmostEqual(r['mean_direction_error_deg'],90)
        self.assertAlmostEqual(r['vector_error_kmh'],np.sqrt(2)*3.6)


if __name__=='__main__':unittest.main()
