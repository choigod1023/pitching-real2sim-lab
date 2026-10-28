import unittest
import numpy as np
from source_audit import event_summary, hand_quality
from hand_geometry import static_template


class SourceAuditTests(unittest.TestCase):
    def test_empty_header_slots_are_not_events(self):
        c = {'parameters': {}, 'header': {'events': {'size': 18,
             'events_label': ['']*18, 'events_time': [0.]*18}}}
        self.assertEqual(event_summary(c)['header_named_events'], 0)
        c['header']['events']['events_label'][0] = 'BR'
        self.assertEqual(event_summary(c)['header_named_events'], 1)

    def test_invalid_and_collinear_static_frames_excluded(self):
        xyz = np.array([[80., 0, 0], [0, 30, 0], [0, -30, 0]])
        points = np.zeros((4, 3, 3))
        points[:3] = np.repeat(xyz.T[:, :, None], 3, axis=2)
        points[3, 0, 1] = -1
        points[:3, :, 2] = 0
        c = {'parameters': {'POINT': {'LABELS': {'value': ['RFIN','RWRA','RWRB']},
             'UNITS': {'value': ['mm']}}}, 'data': {'points': points}}
        self.assertEqual(hand_quality(c, 'R')['valid_frames'], 1)
        self.assertEqual(hand_quality(c, 'L')['reason'], 'missing_label')
        expected=static_template(c)
        c['parameters']['POINT']['LABELS']['value']=['Skeleton_001_'+x for x in ['RFIN','RWRA','RWRB']]
        np.testing.assert_allclose(static_template(c),expected)
        c['data']['points'][3]=-1
        with self.assertRaises(ValueError):static_template(c)
