import unittest
from long_hand_alignment import combine_observations


class ObservationSplitTests(unittest.TestCase):
    def test_same_joint_frame_cannot_enter_both_splits(self):
        a = {'camera': 17, 'frame': 950, 'segment': 'l_hand', 'split': 'fit'}
        b = dict(a, split='eval')
        with self.assertRaises(ValueError):
            combine_observations([a], [b])

    def test_fit_excluded_frames_and_cameras_are_retained(self):
        a = {'camera': 17, 'frame': 950, 'segment': 'l_hand', 'split': 'fit'}
        b = {'camera': 21, 'frame': 950, 'segment': 'l_hand', 'split': 'eval'}
        c = {'camera': 17, 'frame': 955, 'segment': 'l_hand', 'split': 'eval'}
        result = combine_observations([a], [b, c])
        self.assertEqual([r['split'] for r in result], ['fit', 'eval', 'eval'])
        self.assertIsNot(result[0], a)


if __name__ == '__main__':
    unittest.main()
