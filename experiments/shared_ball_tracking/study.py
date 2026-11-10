"""Compare two proposal extractors on an identical app tracker and unseen video sources."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np


SCALES = (1.2, 2.5, 5.0, 10.0)
OFFSETS = (-4, -2, -1, 1, 2, 4)
TOLERANCE_960 = 8.


def decode(path, window):
    cap = cv2.VideoCapture(str(path))
    first, last = window
    if not cap.isOpened() or not 0 <= first < last:
        raise ValueError(f"Unreadable video or window: {path.name}")
    images, frames = [], []
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        factor = min(1., 960 / max(width, height))
        size = (round(width * factor), round(height * factor))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for index in range(last + 1):
            if index < first:
                if not cap.grab():
                    raise ValueError(f"Source ended before frame {first}: {path.name}")
                continue
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Source ended before frame {last}: {path.name}")
            frames.append({"frame": index, "time": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000})
            images.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if any(b["time"] <= a["time"] for a, b in zip(frames, frames[1:])):
        raise ValueError(f"Nonmonotonic decoded PTS: {path.name}")
    return images, frames, (size[0] / width, size[1] / height), fps


def round_candidates(images, anchors):
    """Propose the center of a bright disk using RGB context and two-sided motion."""
    height, width = images[0].shape[:2]
    a, b = np.asarray(anchors, dtype=float)
    delta = b - a
    length = np.linalg.norm(delta)
    if length < width * .1:
        raise ValueError("Detection corridor too short")
    along = delta / length
    across = np.array([-along[1], along[0]])
    band = max(12., .13 * length)
    margin = round(band + .12 * length + 24)
    left, top = np.maximum(np.minimum(a, b).astype(int) - margin, 0)
    right, bottom = np.minimum(np.maximum(a, b).astype(int) + margin, [width, height])
    gray = [cv2.cvtColor(im[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(np.float32) for im in images]
    y, x = np.mgrid[top:bottom, left:right]
    s = (x-a[0])*along[0] + (y-a[1])*along[1]
    v = (x-a[0])*across[0] + (y-a[1])*across[1]
    inside = (s >= -.12*length) & (s <= 1.12*length) & (abs(v) <= band)
    scales = np.asarray(SCALES) * width / 960
    output = []
    for index, frame in enumerate(gray):
        other = [gray[index+k] for k in OFFSETS if 0 <= index+k < len(gray)]
        motion = np.maximum(frame - np.minimum.reduce(other), 0)
        layers = []
        for scale in scales:
            core = cv2.GaussianBlur(frame, (0, 0), max(.55, .55 * scale))
            surround = cv2.GaussianBlur(frame, (0, 0), max(1.1, 1.60 * scale))
            contrast = core - surround
            support = cv2.GaussianBlur(motion, (0, 0), max(.55, .55 * scale))
            score = np.maximum(contrast, 0) * np.sqrt(support)
            score[~inside | (contrast < 4.) | (support < 3.) | (core < 74.)] = 0
            layers.append(score)
        best = np.max(layers, axis=0)
        scale_id = np.argmax(layers, axis=0)
        peaks = (best > 10.) & (best == cv2.dilate(best, np.ones((5, 5), np.uint8)))
        py, px = np.nonzero(peaks)
        ranks = np.lexsort((px, py, -best[py, px]))[:24]
        output.append([
            {"x": int(x[cy, cx]), "y": int(y[cy, cx]), "s": float(s[cy, cx]), "v": float(v[cy, cx]),
             "diameter": float(2 * scales[scale_id[cy, cx]]),
             "strength": float(best[cy, cx])}
            for cy, cx in zip(py[ranks], px[ranks])
        ])
    return output


def score(points, labels, negative, factors, width):
    native_scale = np.asarray(factors)
    by_frame = {point["frame"]: point for point in points}
    deltas = []
    for frame, x, y in labels:
        point = by_frame.get(frame)
        if point:
            difference = np.asarray([point["x"], point["y"]]) - np.asarray([x, y]) * native_scale
            deltas.append((frame, float(np.linalg.norm(difference) * 960 / width)))
    errors = [distance for _, distance in deltas]
    correct = sum(distance <= TOLERANCE_960 for distance in errors)
    return {"positive": len(labels), "returned_on_positive": len(errors), "matched": correct,
            "wrong_on_positive": len(errors) - correct, "missed_positive": len(labels) - len(errors),
            "negative": len(negative), "false_on_negative": sum(frame in by_frame for frame in negative),
            "mean_px_960": float(np.mean(errors)) if errors else None,
            "distances_px_960": deltas}


def execute(item, directory, *, runner=None, extra_config=None):
    path = directory / item["path"]
    with path.open("rb") as stream:
        sha = file_digest(stream, "sha256").hexdigest()
    if sha != item["sha256"]:
        raise ValueError(f"Unexpected source hash: {path.name}")
    images, frames, factors, fps = decode(path, item["window"])
    height, width = images[0].shape[:2]
    anchors = (np.asarray(item["anchors"]) * factors).tolist()
    proposal = round_candidates(images, anchors)
    with TemporaryDirectory() as temp:
        cfg = Path(temp) / "config.json"
        cfg.write_text(json.dumps({"width":width,"height":height,"anchors":anchors,"frames":frames,
                                  "candidates":proposal, **(extra_config or {})}), encoding="utf-8")
        rgba = b"".join(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).tobytes() for frame in images)
        output = subprocess.run(["node", str(runner or Path(__file__).with_name("run_app.mjs")), str(cfg)],
                                input=rgba, capture_output=True, check=True)
    payload = json.loads(output.stdout)
    models = payload.get("models",payload)
    for model in models.values():
        model["score"] = score(model["points"], item["positive"], item["negative"], factors, width)
    result = {"id":item["id"],"group":item["group"],"sha256":sha,"width":width,"height":height,"factors":factors,
              "nominal_playback_fps":fps,"frames":frames,"anchors":anchors,"models":models}
    if "proposalEvidence" in payload:
        result["proposal_evidence"] = payload["proposalEvidence"]
    return result


def summarize(results):
    totals = {}
    for clip in results:
        for name, model in clip["models"].items():
            row = totals.setdefault(clip["group"], {}).setdefault(name, {})
            for key in ("positive", "returned_on_positive", "matched", "wrong_on_positive", "missed_positive",
                        "negative", "false_on_negative"):
                row[key] = row.get(key, 0) + model["score"][key]
    groups = [group for group in totals.values() if group["app"]["positive"]]
    decisions = {}
    for mode in results[0]["models"].keys() - {"app"}:
        decisions[mode] = (
            sum(group[mode]["matched"] > group["app"]["matched"] for group in groups) >= 2
            and all(group[mode]["matched"] >= group["app"]["matched"] for group in groups)
            and all(group[mode]["false_on_negative"] <= group["app"]["false_on_negative"]
                    for group in totals.values())
        )
    return {"by_source": totals, "passes_source_gate": decisions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    results = [execute(item,args.manifest.parent) for item in manifest["clips"]]
    with args.manifest.open("rb") as stream:
        manifest_hash = file_digest(stream, "sha256").hexdigest()
    report = {"manifest_sha256":manifest_hash,"clips":results,**summarize(results)}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
