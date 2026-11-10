"""Measure early optical recovery without using reference centers for tracking."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2

from experiments.release_conditioned_ball.study import SCORE_KEYS, proposal_recall
from experiments.shared_ball_tracking.study import execute, score


SOURCES = ("mlb", "outdoor")


def summarize(rows):
    counts = {}
    for row in rows:
        split = counts.setdefault(row["split"], {})
        group = split.setdefault(row["group"], {})
        for mode, model in row["models"].items():
            summary = group.setdefault(mode, {
                scope: {key: 0 for key in SCORE_KEYS} for scope in ("all", "early")
            })
            for scope, field in (("all", "score"), ("early", "early_score")):
                for key in SCORE_KEYS:
                    summary[scope][key] += model[field][key]

    prime = next(row for row in rows if row["id"] == "prime-22")
    identical = prime["existing_reference_frames_preserved"]
    prime_ok = identical == prime["models"]["app"]["score"]["matched"] == prime["models"]["reverse"]["score"]["matched"] == 24
    holdouts = counts.get("holdout", {})
    source_ok = {}
    for source in SOURCES:
        if source not in holdouts:
            source_ok[source] = False
            continue
        old, new = (holdouts[source][mode] for mode in ("app", "reverse"))
        source_ok[source] = (
            new["early"]["matched"] > old["early"]["matched"]
            and new["all"]["wrong_on_positive"] <= old["all"]["wrong_on_positive"]
            and new["all"]["false_on_negative"] <= old["all"]["false_on_negative"]
        )
    preserved = all(row["existing_path_preserved"] for row in rows)
    return counts, {
        "prime_positions_identical": identical, "prime_gate": prime_ok,
        "existing_paths_preserved": preserved, "source_gates": source_ok,
        "passes_app_gate": prime_ok and preserved and all(source_ok.values()),
    }


def compare(manifest_path):
    data = manifest_path.read_bytes()
    manifest = json.loads(data.decode("utf-8-sig"))
    items = manifest["clips"]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Clip IDs must be unique")
    runner = Path(__file__).with_name("run_app.mjs")
    rows = []
    for item in items:
        low, high = item["release"]
        if not item["window"][0] <= low <= high <= item["window"][1] - 2:
            raise ValueError(f"Release outside analyzable window: {item['id']}")
        positive = item["positive"]
        early = [p for p in positive if high < p[0] <= high + 2]
        early_negative = [f for f in item["negative"] if high < f <= high + 2]
        row = execute(item, manifest_path.parent, runner=runner, extra_config={"release": [low, high]})
        for model in row["models"].values():
            model["early_score"] = score(model["points"], early, early_negative, row["factors"], row["width"])
        old, new = ({p["frame"]: p for p in row["models"][mode]["points"]} for mode in ("app", "reverse"))
        preserved = {frame for frame, point in old.items() if new.get(frame) == point}
        evidence = row.pop("proposal_evidence")
        row.update(
            split=item["split"], release=[low, high],
            existing_path_preserved=len(preserved) == len(old),
            existing_reference_frames_preserved=sum(p[0] in preserved for p in positive),
            early_miss_proposals=proposal_recall(
                evidence, [p for p in early if p[0] not in old], row["factors"], row["width"]
            ),
            early_outside_corridor=sum(
                not (-.12 * item_length <= along <= 1.12 * item_length and abs(cross) <= max(12., .13 * item_length))
                for along, cross, item_length in corridor_coordinates(early, row)
            ),
        )
        rows.append(row)
    counts, gates = summarize(rows)
    return {"manifest_sha256": sha256(data).hexdigest(), "by_split": counts, "gates": gates, "clips": rows}


def corridor_coordinates(labels, row):
    (x0, y0), (x1, y1) = row["anchors"]
    length = ((x1 - x0)**2 + (y1 - y0)**2)**.5
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    sx, sy = row["factors"]
    for _, x, y in labels:
        dx, dy = x * sx - x0, y * sy - y0
        yield dx * ux + dy * uy, dy * ux - dx * uy, length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.write_text(json.dumps(compare(args.manifest), indent=2, allow_nan=False), encoding="utf-8")
