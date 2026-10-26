import unittest
from hand_image_audit import region_distance


class RegionsTests(unittest.TestCase):
    def test_inside_and_boundary(self):
        self.assertEqual(region_distance([2,3],[0,0,4,6]),(0,0))
        self.assertEqual(region_distance([0,3],[0,0,4,6])[0],0)

    def test_outside_euclidean_distance(self):
        self.assertAlmostEqual(region_distance([7,10],[0,0,4,6])[0],5)

    def test_invalid(self):
        for box in ([0,0,0,1],[2,0,1,1],[0,0,float('nan'),1]):
            with self.assertRaises(ValueError):region_distance([1,1],box)


if __name__=='__main__':unittest.main()
