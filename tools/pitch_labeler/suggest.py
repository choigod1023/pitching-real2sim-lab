"""Run the existing image tracker and prepare unconfirmed suggestions for review."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import subprocess


def suggestions(task, result):
    """Preserve measured candidates; never interpolate or label missed frames absent."""
    frame_ids = {frame["index"] for frame in task["frames"]}
    sx, sy = result["factors"]
    points = {}
    for point in result["models"]["proposal_union"]["points"]:
        frame = point["frame"]
        x, y = point["x"] / sx, point["y"] / sy
        if frame not in frame_ids or not all(map(math.isfinite, [x, y])):
            continue
        if not (0 <= x < task["width"] and 0 <= y < task["height"]):
            continue
        points[str(frame)] = {"kind": "ball", "x": round(x, 1), "y": round(y, 1)}
    ordered = sorted(map(int, points))
    # A search hint, not a measured contact or a confirmed release event.
    first = next((f for f in ordered if f - 1 in frame_ids and
                  all(str(f + k) in points for k in (1, 2))), None)
    return {"points": points, "release_hint": None if first is None else [first - 1, first]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment", type=Path)
    parser.add_argument("spec", type=Path)
    parser.add_argument("tracker_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier proposals; choose a new output name")
    assignment = json.loads(args.assignment.read_text(encoding="utf-8-sig"))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    loader = importlib.util.spec_from_file_location("existing_tracker", args.tracker_root /
                                                   "experiments/shared_ball_tracking/study.py")
    tracker = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(tracker)
    tracker.cv2.setNumThreads(2)
    revision = subprocess.check_output(["git", "-C", str(args.tracker_root), "rev-parse", "HEAD"],
                                       text=True).strip()
    provenance = {"method": "proposal_union", "tracker_commit": revision,
                  "generated_at": datetime.now(timezone.utc).isoformat(),
                  "spec_sha256": sha256(args.spec.read_bytes()).hexdigest(),
                  "corridor_source": "operator_scene_bounds", "confirmed": False}
    for task in assignment["tasks"]:
        item = next(row for row in spec["clips"] if row["id"] == task["id"])
        result = tracker.execute({**item, "sha256": task["sha256"], "positive": [], "negative": []},
                                 args.spec.parent)
        proposal = suggestions(task, result)
        task["assistance"] = {**provenance, **proposal, "anchors": item["anchors"]}
        print(json.dumps({"id": task["id"], "suggested": len(proposal["points"]),
                          "frames": len(task["frames"]), "release_hint": proposal["release_hint"]}))
    args.output.write_text(json.dumps(assignment, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
