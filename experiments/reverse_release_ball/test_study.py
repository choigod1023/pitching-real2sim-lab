import unittest

from experiments.release_conditioned_ball.study import SCORE_KEYS
from experiments.reverse_release_ball.study import corridor_coordinates, summarize


def result(id_, split, group, matched, false=0, wrong=0, preserved=True):
    old = dict.fromkeys(SCORE_KEYS, 0)
    old["matched"] = 24 if id_ == "prime-22" else 1
    new = {**old, "matched": matched, "false_on_negative": false, "wrong_on_positive": wrong}
    return {
        "id": id_, "split": split, "group": group, "existing_path_preserved": preserved,
        "existing_reference_frames_preserved": 24,
        "models": {
            "app": {"score": old, "early_score": old},
            "reverse": {"score": new, "early_score": new},
        },
    }


class ReverseGateTests(unittest.TestCase):
    def setUp(self):
        self.prime = result("prime-22", "regression", "prime", 24)
        self.mlb = result("pitch-a", "holdout", "mlb", 2)
        self.outdoor = result("pitch-b", "holdout", "outdoor", 2)

    def test_gate_requires_both_sources_and_an_unchanged_prime(self):
        rows = [self.prime, self.mlb, self.outdoor]
        _, gate = summarize(rows)
        self.assertTrue(gate["passes_app_gate"])
        self.assertFalse(summarize(rows[:2])[1]["passes_app_gate"])
        self.prime["existing_reference_frames_preserved"] = 23
        self.assertFalse(summarize(rows)[1]["passes_app_gate"])

    def test_gate_rejects_false_detections_and_any_existing_path_change(self):
        for error in ("false_on_negative", "wrong_on_positive"):
            with self.subTest(error=error):
                self.mlb["models"]["reverse"]["score"][error] = 1
                _, gate = summarize([self.prime, self.mlb, self.outdoor])
                self.assertFalse(gate["source_gates"]["mlb"])
                self.mlb["models"]["reverse"]["score"][error] = 0
        self.outdoor["existing_path_preserved"] = False
        self.assertFalse(summarize([self.prime, self.mlb, self.outdoor])[1]["passes_app_gate"])

    def test_gate_checks_gain_in_early_frames_separately(self):
        self.mlb["models"]["reverse"]["early_score"] = self.mlb["models"]["app"]["early_score"]
        _, gate = summarize([self.prime, self.mlb, self.outdoor])
        self.assertFalse(gate["source_gates"]["mlb"])

    def test_corridor_coordinates_scale_native_labels(self):
        row = {"anchors": [[10, 10], [110, 10]], "factors": [0.5, 0.5]}
        self.assertEqual(list(corridor_coordinates([(1, 60, 30)], row)), [(20, 5, 100)])
