"""Check two independent submissions and list decisions for an adjudicator."""
import argparse
import json
from math import hypot, isfinite
from pathlib import Path


def validate(assignment, submission):
    if (submission["schema"] != "pitch-video-annotation/1" or submission["practice"] is not False
            or submission["batch"] != assignment["batch"] or not submission["reader"]):
        raise ValueError("Submission does not belong to this research assignment")
    tasks = {task["id"]: task for task in assignment["tasks"]}
    clips = submission["clips"]
    if len(clips) != len(tasks) or {clip["id"] for clip in clips} != tasks.keys():
        raise ValueError("Wrong or duplicate task IDs")
    output = {}
    for clip in clips:
        task = tasks[clip["id"]]
        frames = [frame["index"] for frame in task["frames"]]
        if (clip["sha256"], clip["width"], clip["height"], clip["window"], clip["group"]) != (
            task["sha256"], task["width"], task["height"], [frames[0], frames[-1]], task["group"]
        ):
            raise ValueError(f"Mismatched source for {clip['id']}")
        marks = {frame: ("absent",) for frame in clip["negative"]}
        for frame in clip["unknown"]:
            if frame in marks:
                raise ValueError(f"Duplicate frame {frame}")
            marks[frame] = ("unknown",)
        for frame, x, y in clip["positive"]:
            if (frame in marks or not isfinite(x) or not isfinite(y)
                    or not 0 <= x < task["width"] or not 0 <= y < task["height"]):
                raise ValueError(f"Invalid ball mark at {frame}")
            marks[frame] = ("ball", x, y)
        if len(marks) != len(clip["positive"]) + len(clip["negative"]) + len(clip["unknown"]) or marks.keys() != set(frames):
            raise ValueError(f"Missing or duplicate frames for {clip['id']}")
        release = clip["release"]
        contact, free = clip["last_confirmed_contact"], clip["first_confirmed_free"]
        if release is None:
            if contact is not None or free is not None:
                raise ValueError(f"Partial release for {clip['id']}")
        elif (contact not in marks or free not in marks or contact >= free or release != [contact, free - 1]
              or marks[contact][0] != "absent" or marks[free][0] != "ball"
              or any(frame < free and mark[0] == "ball" for frame, mark in marks.items())):
            raise ValueError(f"Inconsistent release for {clip['id']}")
        output[clip["id"]] = (release, marks)
    return output


def compare(assignment, left, right):
    if left["reader"] == right["reader"]:
        raise ValueError("Use two distinct reader codes")
    left_tasks = validate(assignment, left)
    right_tasks = validate(assignment, right)
    rows = []
    for task in assignment["tasks"]:
        left_release, left_marks = left_tasks[task["id"]]
        right_release, right_marks = right_tasks[task["id"]]
        status_conflicts, centres = [], []
        for frame in sorted(left_marks):
            a, b = left_marks[frame], right_marks[frame]
            if a[0] != b[0]:
                status_conflicts.append([frame, a[0], b[0]])
            elif a[0] == "ball":
                centres.append([frame, round(hypot(a[1]-b[1], a[2]-b[2])*960/task["width"], 3)])
        rows.append({"id":task["id"], "release":[left_release,right_release],
                     "status_conflicts":status_conflicts, "ball_distances_px960":centres})
    return {"batch":assignment["batch"], "readers":[left["reader"],right["reader"]], "clips":rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assignment", "left", "right", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    data = [json.loads(path.read_text(encoding="utf-8-sig")) for path in (args.assignment,args.left,args.right)]
    args.output.write_text(json.dumps(compare(*data), ensure_ascii=False, indent=2), encoding="utf-8")
