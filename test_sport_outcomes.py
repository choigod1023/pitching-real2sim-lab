import unittest
import pandas as pd
from sport_outcomes import attach_observed_speed, evidence_contract


class SportsOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({'user': [1], 'session_pitch': ['a'], 'pitch_speed_mph': [90.]})
        self.rows = pd.DataFrame({'user': [1, 1], 'session_pitch': ['a', 'a'],
                                  'delta_hand_speed_m_s': [0., 100.]})

    def test_proxy_cannot_change_observed_ball_speed(self):
        result = attach_observed_speed(self.rows, self.meta)
        self.assertEqual(len(result), 2)
        for value in result.observed_original_ball_speed_kmh:
            self.assertAlmostEqual(value, 144.84096)
        self.assertEqual(result.delta_hand_speed_m_s.tolist(), [0., 100.])

    def test_wrong_or_ambiguous_identity_rejected(self):
        with self.assertRaises(ValueError):
            attach_observed_speed(self.rows.assign(user=2), self.meta)
        with self.assertRaises(ValueError):
            attach_observed_speed(self.rows, pd.concat([self.meta, self.meta]))

    def test_missing_measurement_rejected(self):
        for value in [float('nan'), float('inf'), -1., 0.]:
            with self.assertRaises(ValueError):
                attach_observed_speed(self.rows, self.meta.assign(pitch_speed_mph=value))

    def test_no_sports_claim_from_kinematic_evidence(self):
        evidence = evidence_contract()
        self.assertIsNone(evidence['predicted_candidate_ball_speed_kmh'])
        self.assertIsNone(evidence['predicted_candidate_target_error_cm'])
        self.assertFalse(evidence['coaching_recommendation_ready'])
