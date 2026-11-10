"""Test protected ball paths on frozen regression and unseen baseball clips."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.release_conditioned_ball.study import SCORE_KEYS, proposal_recall
from experiments.shared_ball_tracking.study import execute


def summarize(rows):
    groups = {}
    for clip in rows:
        group = groups.setdefault(clip["split"], {}).setdefault(clip["group"], {})
        for name, model in clip["models"].items():
            scores = group.setdefault(name, {key:0 for key in SCORE_KEYS})
            for key in SCORE_KEYS:
                scores[key] += model["score"][key]
    prime = next(row for row in rows if row["id"] == "prime-22")
    identical = prime["existing_positive_positions_preserved"]["preserved"]
    anchored_identical = prime["existing_positive_positions_preserved"]["anchored"]
    def prime_gate(mode, count):
        return count == prime["models"]["app"]["score"]["matched"] == prime["models"][mode]["score"]["matched"] == 24
    prime_ok = prime_gate("preserved",identical)
    def source_gate(split, mode):
        holdout = list(groups.get(split,{}).values())
        return (len(holdout) >= 2
                and all(row[mode]["matched"] > row["app"]["matched"]
                        and row[mode]["false_on_negative"] <= row["app"]["false_on_negative"]
                        and row[mode]["wrong_on_positive"] <= row["app"]["wrong_on_positive"]
                        for row in holdout))
    heldout_ok = source_gate("new_holdout","preserved")
    reserve_ok = source_gate("reserve_holdout","anchored")
    return groups, {"prime_positions_identical":identical,"prime_gate":prime_ok,
                    "new_source_gate":heldout_ok,"passes_app_gate":prime_ok and heldout_ok,
                    "anchored_prime_positions_identical":anchored_identical,
                    "anchored_new_source_gate":reserve_ok,
                    "anchored_passes_app_gate":prime_gate("anchored",anchored_identical) and reserve_ok}


def compare(path):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    runner = Path(__file__).with_name("run_app.mjs")
    results = []
    for item in manifest["clips"]:
        row = execute(item,path.parent,runner=runner,extra_config={"release":item["release"]})
        positive = {frame for frame,_,_ in item["positive"]}
        old = {p["frame"]:(p["x"],p["y"]) for p in row["models"]["app"]["points"] if p["frame"] in positive}
        preserved = {}
        for name in ("preserved","anchored"):
            kept = {p["frame"]:(p["x"],p["y"]) for p in row["models"][name]["points"] if p["frame"] in positive}
            preserved[name] = sum(kept.get(frame)==point for frame,point in old.items())
        origin, end = np.asarray(row["anchors"],dtype=float)
        direction = end-origin
        length = np.linalg.norm(direction)
        direction /= length
        labels = np.asarray(item["positive"])[:,1:3]*row["factors"]-origin
        along = labels@direction
        cross = labels@[-direction[1],direction[0]]
        inside = (-.12*length <= along) & (along <= 1.12*length) & (abs(cross)<=max(12.,.13*length))
        evidence = row.pop("proposal_evidence")
        missing = [point for point in item["positive"] if point[0] not in old]
        results.append({**row,"split":item["split"],"release":item["release"],
                        "positive_outside_corridor":int((~inside).sum()),
                        "existing_positive_positions_preserved":preserved,
                        "candidate_availability_on_app_misses":proposal_recall(
                            evidence,missing,row["factors"],row["width"])})
    groups, gates = summarize(results)
    with path.open("rb") as stream:
        digest = file_digest(stream,"sha256").hexdigest()
    return {"manifest_sha256":digest,"by_split":groups,"gates":gates,"clips":results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.write_text(json.dumps(compare(args.manifest),indent=2,allow_nan=False),encoding="utf-8")
