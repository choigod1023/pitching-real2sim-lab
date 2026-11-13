import unittest
from compare import compare


class ComparisonModeTests(unittest.TestCase):
    def test_assisted_review_cannot_be_reported_as_blind_reference(self):
        task = {"id": "pitch", "group": "QA", "sha256": "a" * 64, "width": 100, "height": 100,
                "frames": [{"index": 1}, {"index": 2}], "assistance": {"method": "proposal_union"}}
        assignment = {"batch": "test", "tasks": [task]}
        clip = {"id": "pitch", "group": "QA", "sha256": "a" * 64, "width": 100, "height": 100,
                "window": [1, 2], "negative": [1], "unknown": [2], "positive": [],
                "release": None, "last_confirmed_contact": None, "first_confirmed_free": None}
        left = {"schema": "pitch-video-annotation/1", "batch": "test", "reader": "QA1", "practice": False, "clips": [clip]}
        right = {**left, "reader": "QA2"}
        self.assertFalse(compare(assignment, left, right)["independent_blind_reference"])
        del task["assistance"]
        self.assertTrue(compare(assignment, left, right)["independent_blind_reference"])
        clip["review_mode"] = "machine_assisted"
        self.assertFalse(compare(assignment, left, right)["independent_blind_reference"])


if __name__ == "__main__":
    unittest.main()
