"""Evaluate release gating and multi-hypothesis tracking on frozen baseball clips."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.shared_ball_tracking.study import TOLERANCE_960, execute, score


SCORE_KEYS = ("positive", "matched", "returned_on_positive", "wrong_on_positive",
              "missed_positive", "negative", "false_on_negative")


def aggregate(clips):
    groups = {}
    for clip in clips:
        group = groups.setdefault(clip["split"], {}).setdefault(clip["group"], {})
        for mode, result in clip["models"].items():
            row = group.setdefault(mode, {key:0 for key in SCORE_KEYS})
            for key in SCORE_KEYS:
                row[key] += result["score"][key]
    heldout = [value for value in groups.get("heldout_pitch", {}).values() if value["app"]["positive"]]
    supported = (
        sum(row["release_beam"]["matched"] > row["app"]["matched"] for row in heldout) >= 2
        and all(row["release_beam"]["matched"] >= row["app"]["matched"]
                and row["release_beam"]["false_on_negative"] <= row["app"]["false_on_negative"] for row in heldout)
    )
    return groups, supported


def proposal_recall(evidence, labels, factors, width):
    """Measure candidate availability using references after inference."""
    lookup = {row["frame"]:row for row in evidence}
    counts = dict(app=0,round=0,either=0)
    for frame, x, y in labels:
        reference = np.asarray([x,y]) * factors
        row = lookup[frame]
        matches = {}
        for mode in ("app","round"):
            values = np.asarray(row[mode]).reshape(-1,2)
            distances = np.linalg.norm(values-reference,axis=1)*960/width
            matches[mode] = bool(np.any(distances <= TOLERANCE_960))
            counts[mode] += matches[mode]
        counts["either"] += any(matches.values())
    return {"positive":len(labels),"available":counts}


def compare(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    runner = Path(__file__).with_name("run_app.mjs")
    results = []
    for item in manifest["clips"]:
        start, stop = item["window"]
        low, high = item["release"]
        if not start <= low <= high < stop:
            raise ValueError(f"Release bracket outside reviewed window: {item['id']}")
        row = execute(item,manifest_path.parent,runner=runner,extra_config={"release":item["release"]})
        after = [point for point in item["positive"] if point[0] > high]
        for result in row["models"].values():
            result["score_after_release_bracket"] = score(
                result["points"],after,[],row["factors"],row["width"])
        proposals = proposal_recall(row.pop("proposal_evidence"),after,row["factors"],row["width"])
        results.append({**row,"split":item["split"],"release":item["release"],
                        "positives_in_or_before_bracket":len(item["positive"])-len(after),
                        "proposal_availability_after_bracket":proposals})
    groups, supported = aggregate(results)
    with manifest_path.open("rb") as stream:
        digest = file_digest(stream,"sha256").hexdigest()
    return {"manifest_sha256":digest,"by_split":groups,"passes_holdout_source_gate":supported,"clips":results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.write_text(json.dumps(compare(args.manifest),indent=2,allow_nan=False),encoding="utf-8")
