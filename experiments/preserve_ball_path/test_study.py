import unittest

from experiments.release_conditioned_ball.study import SCORE_KEYS
from experiments.preserve_ball_path.study import summarize


class PreservePathStudyTests(unittest.TestCase):
    @staticmethod
    def clip(id, split, group, old, new, wrong=0, false=0):
        def score(matched, w=0, f=0):
            return {key:{"positive":24,"matched":matched,"wrong_on_positive":w,"false_on_negative":f}.get(key,0)
                    for key in SCORE_KEYS}
        return {"id":id,"split":split,"group":group,
                "existing_positive_positions_preserved":{"preserved":24,"anchored":24},
                "models":{"app":{"score":score(old)},"preserved":{"score":score(new,wrong,false)},
                          "anchored":{"score":score(new,wrong,false)}}}

    def test_holdout_gate_requires_gain_on_both_sources_and_no_more_false_points(self):
        prime = self.clip("prime-22","regression","prime",24,24)
        mlb = self.clip("mlb","new_holdout","mlb",2,3)
        unchanged = self.clip("out","new_holdout","outdoor",4,4)
        wrong = self.clip("out","new_holdout","outdoor",4,5,wrong=1)
        self.assertFalse(summarize([prime,mlb,unchanged])[1]["passes_app_gate"])
        self.assertFalse(summarize([prime,mlb,wrong])[1]["passes_app_gate"])
        self.assertTrue(summarize([prime,mlb,self.clip("out","new_holdout","outdoor",4,5)])[1]["passes_app_gate"])

    def test_followup_cannot_use_the_first_holdout_as_independent(self):
        clips = [self.clip("prime-22","regression","prime",24,24),
                 self.clip("mlb","new_holdout","mlb",2,3),
                 self.clip("out","new_holdout","outdoor",4,5)]
        self.assertFalse(summarize(clips)[1]["anchored_passes_app_gate"])
        for row in clips[1:]:
            row["split"] = "reserve_holdout"
        self.assertTrue(summarize(clips)[1]["anchored_passes_app_gate"])


if __name__ == "__main__":
    unittest.main()
