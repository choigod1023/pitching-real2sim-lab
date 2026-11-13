"""Prepare exact decoded pitch frames and a blind annotation assignment."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import cv2


def prepare(spec_path, destination):
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    ids = [row["id"] for row in spec["clips"]]
    if len(ids) != len(set(ids)) or not ids or any(
        not item or len(item) > 64 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in item)
        for item in ids
    ):
        raise ValueError("Invalid or duplicate task IDs")
    destination.mkdir(parents=True, exist_ok=False)
    tasks = []
    for clip in spec["clips"]:
        source = spec_path.parent / clip["path"]
        with source.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        capture = cv2.VideoCapture(str(source))
        first, last = clip["window"]
        if not capture.isOpened() or not 0 <= first < last:
            capture.release()
            raise ValueError(f"Unreadable source or window for {clip['id']}")
        folder = destination / "frames" / clip["id"]
        folder.mkdir(parents=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        try:
            for frame in range(last + 1):
                if frame < first:
                    if not capture.grab():
                        raise ValueError(f"Missing source frame {frame}")
                    continue
                ok, image = capture.read()
                if not ok or image.shape[:2] != (height, width):
                    raise ValueError(f"Invalid source frame {frame}")
                time_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                if frames and time_ms <= frames[-1]["time_ms"]:
                    raise ValueError(f"Non-monotonic PTS at {frame}")
                output = folder / f"{frame}.png"
                ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not ok:
                    raise ValueError(f"Cannot encode frame {frame}")
                output.write_bytes(encoded.tobytes())
                frames.append({"index": frame, "time_ms": round(time_ms, 6),
                               "file": output.relative_to(destination).as_posix()})
        finally:
            capture.release()
        tasks.append({"id": clip["id"], "group": clip["group"], "sha256": digest,
                      "width": width, "height": height, "frames": frames})
    assignment = {"version": 1, "batch": spec["batch"], "tasks": tasks}
    (destination / "assignment.json").write_text(
        json.dumps(assignment, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return assignment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    data = prepare(args.spec, args.destination)
    for task in data["tasks"]:
        print(task["id"], len(task["frames"]), task["sha256"])
