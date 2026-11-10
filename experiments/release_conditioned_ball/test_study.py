import unittest

from experiments.release_conditioned_ball.study import aggregate, proposal_recall


class ReleaseConditionedStudyTests(unittest.TestCase):
    def test_candidate_recall_uses_either_list_and_preserves_missing_frame(self):
        frames = [
            {"frame":1,"app":[[40,20],[200,10]],"round":[]},
            {"frame":2,"app":[[200,10]],"round":[[45,22]]},
            {"frame":3,"app":[],"round":[]},
        ]
        report = proposal_recall(frames,[[1,80,40],[2,90,44],[3,100,45]],[.5,.5],960)
        self.assertEqual(report,{"positive":3,"available":{"app":1,"round":1,"either":2}})

    def test_same_pitch_camera_transfer_cannot_pass_the_heldout_pitch_gate(self):
        def clip(split, group, base, new):
            scores = lambda n: dict(positive=3,matched=n,returned_on_positive=n,wrong_on_positive=0,
                                    missed_positive=3-n,negative=1,false_on_negative=0)
            return {"split":split,"group":group,"models":{"app":{"score":scores(base)},
                    "release_beam":{"score":scores(new)}}}
        clips = [clip("heldout_pitch","a",0,2),clip("camera_transfer_same_pitch","b",0,3)]
        self.assertFalse(aggregate(clips)[1])
        self.assertTrue(aggregate([*clips,clip("heldout_pitch","c",1,2)])[1])


if __name__ == "__main__":
    unittest.main()
