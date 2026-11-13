import unittest
from suggest import suggestions


class SuggestionTests(unittest.TestCase):
    def test_native_coordinates_missing_frames_and_search_hint(self):
        task = {"width": 1280, "height": 720, "frames": [{"index": i} for i in range(10, 17)]}
        result = {"factors": [.75, .75], "models": {"proposal_union": {"points": [
            {"frame": 11, "x": 150, "y": 75}, {"frame": 12, "x": 160, "y": 80},
            {"frame": 13, "x": 170, "y": 85}, {"frame": 15, "x": float("nan"), "y": 90},
            {"frame": 9, "x": 10, "y": 10}, {"frame": 16, "x": 960, "y": 80}]}}}
        output = suggestions(task, result)
        self.assertEqual(output["points"]["11"], {"kind": "ball", "x": 200., "y": 100.})
        self.assertEqual(set(output["points"]), {"11", "12", "13"})
        self.assertEqual(output["release_hint"], [10, 11])
        result["models"]["proposal_union"]["points"] = []
        self.assertEqual(suggestions(task, result), {"points": {}, "release_hint": None})


if __name__ == "__main__":
    unittest.main()
