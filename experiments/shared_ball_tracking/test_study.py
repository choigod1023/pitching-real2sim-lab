import unittest

import cv2
import numpy as np

from experiments.shared_ball_tracking import study


class SharedBallTrackingTests(unittest.TestCase):
    def test_disk_proposals_follow_centres_with_shared_settings(self):
        frames = []
        for i in range(9):
            frame = np.full((120, 240, 3), 35, dtype=np.uint8)
            cv2.circle(frame, (40 + 12*i, 60), 4, (240, 240, 240), -1)
            frames.append(frame)
        candidates = study.round_candidates(frames, [[20, 60], [200, 60]])
        for i, frame in enumerate(candidates):
            self.assertTrue(any(np.hypot(p["x"]-(40+12*i), p["y"]-60) <= 2. for p in frame))
        blank = [np.zeros_like(frames[0]) for _ in range(5)]
        self.assertEqual(study.round_candidates(blank, [[20, 60], [200, 60]]), [[]]*5)

    def test_score_respects_coordinate_scale_and_missingness(self):
        points = [{"frame":1,"x":20,"y":30},{"frame":2,"x":48,"y":40},
                  {"frame":4,"x":20,"y":30},{"frame":5,"x":0,"y":0}]
        result = study.score(points, [[1,40,60],[2,80,80],[3,45,70]], [4], [.5,.5], 960)
        self.assertEqual(result["matched"],2)
        self.assertEqual(result["missed_positive"],1)
        self.assertEqual(result["false_on_negative"],1)
        self.assertEqual(result["mean_px_960"],4.)

    def test_source_gate_does_not_accept_a_false_positive_tradeoff(self):
        def row(hits, false):
            return {"positive":3,"matched":hits,"returned_on_positive":hits,"wrong_on_positive":0,
                    "missed_positive":3-hits,"negative":2,"false_on_negative":false}
        clips = [{"group":domain,"models":{"app":{"score":row(1,0)},
                                          "better":{"score":row(2,0)},
                                          "worse":{"score":row(3,1)}}}
                 for domain in ("a","b")]
        self.assertEqual(study.summarize(clips)["passes_source_gate"],{"better":True,"worse":False})


if __name__ == "__main__":
    unittest.main()
