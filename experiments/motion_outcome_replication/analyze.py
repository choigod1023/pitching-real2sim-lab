"""Retrospective within-pitcher comparisons; never a coaching prescription."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

JOINTS = {"trunk_tilt": [11, 12, 23, 24], "lead_knee": [23, 25, 27],
          "throwing_elbow": [12, 14, 16], "arm_direction": [12, 16]}
TARGETS = ("speed_kmh", "rpm", "vertical_cm", "horizontal_cm")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wrap(value):
    return (np.asarray(value) + 180) % 360 - 180


def original_clock(path):
    cap = cv2.VideoCapture(str(path))
    times, size = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = [frame.shape[1], frame.shape[0]]
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    times = np.asarray(times)
    if len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source clock is missing or nonmonotonic")
    return times, size


def map_clock(cached, original):
    indices = np.abs(original[:, None] - cached[None, :]).argmin(axis=0)
    error = np.abs(original[indices] - cached)
    if len(indices) < 2 or np.any(np.diff(indices) != 1) or error.max() > 0.1:
        raise ValueError("Track does not map uniquely to sequential source frames")
    return indices, float(error.max())


def angles(points, scores, size, nominal, excluded=None):
    valid = np.isfinite(points).all(-1) & np.isfinite(scores) & (scores >= .5)
    valid &= (points[:, :, 0] >= 0) & (points[:, :, 0] < size[0])
    valid &= (points[:, :, 1] >= 0) & (points[:, :, 1] < size[1])
    result, masks = {}, {}
    excluded = excluded or {}
    for name, ids in JOINTS.items():
        p = points[:, ids]
        good = valid[:, ids].all(-1)
        if name == "trunk_tilt":
            u = p[:, :2].mean(1) - p[:, 2:].mean(1)
            links = np.stack([p[:, 0]-p[:, 1], p[:, 2]-p[:, 3], u], axis=1)
            value = np.degrees(np.arctan2(u[:, 0], -u[:, 1]))
        elif name == "arm_direction":
            u = p[:, 1]-p[:, 0]
            links = u[:, None, :]
            value = np.degrees(np.arctan2(u[:, 0], -u[:, 1]))
        else:
            u, v = p[:, 0]-p[:, 1], p[:, 2]-p[:, 1]
            links = np.stack([u, v], axis=1)
            value = np.degrees(np.arctan2(np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]), (u*v).sum(-1)))
        length = np.linalg.norm(links, axis=-1)
        if good.any():
            typical = np.median(length[good], axis=0)
            good &= (length >= np.maximum(8, .35*typical)).all(-1) & (length <= 2*typical).all(-1)
        alias = {"lead_knee": "knee", "trunk_tilt": "trunk", "throwing_elbow": "arm", "arm_direction": "arm"}[name]
        good &= ~np.isin(nominal, excluded.get(alias, []))
        result[name] = np.where(good, value, np.nan)
        masks[name] = good
    return result, masks


def window_median(values, relative, low, high, minimum):
    if relative[0]>low or relative[-1]<high:
        return None
    selected = values[(relative >= low-1e-6) & (relative <= high+1e-6)]
    finite = selected[np.isfinite(selected)]
    if len(selected) < minimum or len(finite) < max(minimum, int(np.ceil(.8*len(selected)))):
        return None
    # Unwrap relative to an observed value, not across missing time samples.
    local = finite[0] + wrap(finite-finite[0])
    if np.ptp(local)>=180:
        return None
    return float(wrap(np.median(local)))


def features(traces, times, first_index):
    result = {}
    for name, values in traces.items():
        level, change = [], []
        for offset in (-1, 0, 1):
            index = first_index+offset if first_index is not None else -1
            if index < 0 or index >= len(times):
                level.append(None); change.append(None); continue
            relative = times-times[index]
            level.append(window_median(values, relative, -150, -33, 4))
            early = window_median(values, relative, -150, -100, 2)
            late = window_median(values, relative, -83, -33, 2)
            change.append(float(wrap(late-early)) if early is not None and late is not None else None)
        result[name+"_level"] = level
        result[name+"_change"] = change
    return result


def pair_feature(low, high):
    grid = [float(wrap(b-a)) if a is not None and b is not None else None for a in low for b in high]
    values = [v for v in grid if v is not None]
    nominal = float(wrap(high[1]-low[1])) if low[1] is not None and high[1] is not None else None
    return {"nominal": nominal, "grid": grid, "n": len(values), "range": [min(values), max(values)] if values else None,
            "stable_direction": (1 if min(values)>0 else -1 if max(values)<0 else 0) if len(values)>=8 else 0}


def ridge(x, y, games):
    x, y = np.asarray(x, float), np.asarray(y, float)
    counts = Counter(games)
    weights = np.asarray([1/counts[g] for g in games], float)
    weights /= weights.sum()
    scale = max(1., float(np.median(np.abs(x))))
    coefficient = float(np.sum(weights*x*y)/(np.sum(weights*x*x)+scale*scale))
    return coefficient, float(np.sum(weights*y))


def evaluate(pairs, feature, target, scenario=4):
    games = sorted({p["game"] for p in pairs})
    folds = []
    for game in games:
        all_train = [p for p in pairs if p["game"] != game]
        train = [p for p in all_train if p["features"][feature]["grid"][scenario] is not None]
        test = [p for p in pairs if p["game"] == game]
        # The selected-group mean baseline uses all training pairs, not a feature-selected subset.
        counts = Counter(p["game"] for p in all_train)
        mean = sum(p["delta"][target]/counts[p["game"]] for p in all_train)/len(counts) if counts else 0.
        enough = len(train)>=2 and len({p["game"] for p in train})>=2
        slope = ridge([p["features"][feature]["grid"][scenario] for p in train],
                      [p["delta"][target] for p in train], [p["game"] for p in train])[0] if enough else None
        predictions = []
        for p in test:
            x = p["features"][feature]["grid"][scenario]
            fallback = slope is None or x is None
            pred = 0. if fallback else slope*x
            y = p["delta"][target]
            bounds = p["features"][feature]["range"]
            predictions.append({"pair": p["id"], "observed": y, "predicted": pred, "fallback": fallback,
                                "mean_baseline": mean, "absolute_error": abs(pred-y), "zero_error": abs(y),
                                "mean_error": abs(mean-y), "prediction_timing_range": sorted(slope*np.asarray(bounds)) if not fallback and bounds else None})
        errors = {k: float(np.mean([p[k] for p in predictions])) for k in ("absolute_error", "zero_error", "mean_error")}
        folds.append({"game": game, "train_games": sorted({p['game'] for p in train}),
                      "candidate_train_games": sorted(counts), "effective_train_pairs": len(train),
                      "slope": slope, "predictions": predictions, **errors})
    metrics = {k: float(np.mean([f[k] for f in folds])) for k in ("absolute_error", "zero_error", "mean_error")}
    slopes = [f["slope"] for f in folds]
    sign_ok = all(x is not None and x>0 for x in slopes) or all(x is not None and x<0 for x in slopes)
    fallback = sum(p["fallback"] for f in folds for p in f["predictions"])
    passes = len(games)>=3 and fallback==0 and sign_ok and all(f["absolute_error"]<min(f["zero_error"],f["mean_error"]) for f in folds)
    return {"feature": feature, "target": target, "folds": folds, "fallback_count": fallback,
            "prediction_candidate": passes, "beats_both_game_average": metrics["absolute_error"]<min(metrics["zero_error"],metrics["mean_error"]), **metrics}


def repetition(pairs, feature):
    games = []
    for game in sorted({p["game"] for p in pairs}):
        all_rows = [p["features"][feature] for p in pairs if p["game"] == game]
        rows = [p for p in all_rows if p["n"]>=8]
        # Every selected pair in a game must support the timing comparison.
        bounds = [float(np.median([p["range"][i] for p in rows])) for i in (0, 1)] if rows else None
        direction = (1 if bounds[0]>0 else -1 if bounds[1]<0 else 0) if bounds and len(rows)==len(all_rows) else 0
        games.append({"game": game, "supported_pairs": len(rows), "total_pairs": len(all_rows),
                      "median_range": bounds, "direction": direction})
    repeat = len(games)>=3 and games[0]["direction"]!=0 and all(g["direction"]==games[0]["direction"] for g in games)
    return {"feature": feature, "games": games, "descriptive_repeat": repeat}


def outcomes(row):
    return {"speed_kmh": float(row["release_speed"])*1.609344, "rpm": float(row["release_spin_rate"]),
            "vertical_cm": float(row["pfx_z"])*30.48, "horizontal_cm": float(row["pfx_x"])*30.48}


def load_inputs(lab):
    spin = lab/"skenes-ff-motion-cohort-20261110"
    speed = lab/"skenes-cross-game-motion-20261106/artifacts/skenes-cross-game"
    group = read(spin/"cohort.json"); event = read(spin/"motion-review.json"); prov = read(spin/"pose-provenance.json")
    clips, pairs, sources = {}, [], {}
    for folder, names in ((spin, ["cohort.json", "motion-review.json", "pose.npz", "pose-provenance.json"]),
                          (speed, ["selection.private.json", "review.private.json", "pose.private.npz", "pose-source.private.json", "skenes-2024.private.csv"])):
        for name in names:
            sources[str((folder/name).relative_to(lab))] = digest(folder/name)
    with np.load(spin/"pose.npz", allow_pickle=False) as z:
        for game in group["games"]:
            if game["game"] not in group["media_attempt_first_games"]:
                continue
            for p in game["chosen"]:
                keys = []
                for side in ("low", "high"):
                    meta = p[side]; key = meta["play_id"][:8]; keys.append(key)
                    clips[key] = {"video": str(spin/"clips"/(key+".mp4")), "source_sha256": prov["video_hashes"][key],
                                  "points": z[key+"_point"], "scores": z[key+"_score"], "times": z[key+"_time"],
                                  "nominal_frames": z[key+"_frame"], "first": event[key]["nominal_first_separated_ball"],
                                  "record": outcomes(meta["row"]), "play_id": meta["play_id"], "exclude": {}, "meta": meta}
                pairs.append({"id": "spin-"+keys[0]+"-"+keys[1], "cohort": "matched_speed_spin", "game": str(game["game"]), "date": game["date"], "keys": keys})
    selection = read(speed/"selection.private.json"); review = read(speed/"review.private.json"); prov = read(speed/"pose-source.private.json")
    with (speed/"skenes-2024.private.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            if int(row["pitcher"])!=694973:
                continue
            identity = (int(row["game_pk"]), int(row["at_bat_number"]), int(row["pitch_number"]))
            if identity in rows:
                raise ValueError("Nonunique same-pitch row")
            rows[identity] = row
    with np.load(speed/"pose.private.npz", allow_pickle=False) as z:
        for keys in selection["pairs"]:
            meta = selection["pitches"][keys[0]]
            pairs.append({"id": "speed-"+"-".join(keys), "cohort": "speed_contrast", "game": str(meta["game"]), "date": meta["date"], "keys": keys})
        for key in sorted(k[:-6] for k in z.files if k.endswith("_point")):
            if key in clips:
                raise ValueError("Duplicate pitch across selected groups")
            meta = selection["pitches"][key]; row = rows[(meta["game"], meta["at_bat"], meta["pitch_number"])]
            record = outcomes(row)
            if row["pitch_type"]!="FF" or abs(record["speed_kmh"]-meta["speed_kmh"])>1e-5 or record["rpm"]!=meta["spin_rpm"] or abs(record["vertical_cm"]-meta["vertical_break_cm"])>1e-5:
                raise ValueError("Pitch record mismatch")
            free = review[key].get("free")
            if free is not None and free[0]!=free[1]:
                raise ValueError("Non-point free-ball reference needs a separate protocol")
            clips[key] = {"video": str(speed/(key+".mp4")), "source_sha256": prov["clips"][key],
                          "points": z[key+"_point"], "scores": z[key+"_score"], "times": z[key+"_time"],
                          "nominal_frames": np.arange(159, 208), "first": free[0] if free else None,
                          "record": record, "play_id": meta["play_id"], "exclude": review[key].get("exclude", {}), "meta": meta}
    return clips, pairs, sources


def run(lab, out):
    cv2.setNumThreads(2)
    clips, pairs, sources = load_inputs(lab)
    if len({c["source_sha256"] for c in clips.values()})!=len(clips) or len({c["play_id"] for c in clips.values()})!=len(clips):
        raise ValueError("Source pitch duplicated")
    cache = {}; public_clips = {}
    for key, clip in clips.items():
        if digest(clip["video"])!=clip["source_sha256"]:
            raise ValueError("Source video changed: "+key)
        source_times, size = original_clock(clip["video"])
        frames, error = map_clock(clip["times"], source_times)
        first_rows = np.flatnonzero(clip["nominal_frames"]==clip["first"])
        first_index = int(first_rows[0]) if len(first_rows)==1 else None
        traces, masks = angles(clip["points"], clip["scores"], size, clip["nominal_frames"], clip["exclude"])
        values = features(traces, clip["times"], first_index)
        public_clips[key] = {k: v for k, v in clip.items() if k not in ("points", "scores", "times", "nominal_frames", "meta")}
        public_clips[key].update({"features": values, "first_index": first_index, "first_source_frame": int(frames[first_index]) if first_index is not None else None,
                                 "source_size": size, "clock_max_error_ms": error, "frame_offsets": sorted(set((frames-clip["nominal_frames"]).tolist()))})
        for name, value in {"points": clip["points"], "scores": clip["scores"], "frames": frames, "times": clip["times"],
                            **traces, **{k+"_mask": v for k, v in masks.items()}}.items():
            cache[key+"_"+name] = value
        print("source", key, public_clips[key]["frame_offsets"], "first", public_clips[key]["first_source_frame"], flush=True)
    kept, exclusions = [], []
    for pair in pairs:
        missing = [key for key in pair["keys"] if key not in public_clips or public_clips[key]["first_index"] is None]
        if missing:
            exclusions.append({**pair, "reason": "missing_video_or_first_free_ball", "missing": missing}); continue
        a, b = [public_clips[k] for k in pair["keys"]]
        kept.append({**pair, "delta": {k: b["record"][k]-a["record"][k] for k in TARGETS},
                     "features": {k: pair_feature(a["features"][k], b["features"][k]) for k in a["features"]}})
    groups = {}
    for cohort in sorted({p["cohort"] for p in kept}):
        subset = [p for p in kept if p["cohort"]==cohort]
        feature_names = list(subset[0]["features"])
        repeats = [repetition(subset, f) for f in feature_names]
        models = [evaluate(subset, f, t) for f in feature_names for t in TARGETS]
        for model in models:
            cases = [evaluate(subset, model["feature"], model["target"], s) for s in range(9)]
            model["timing_cases"] = [{k: c[k] for k in ("absolute_error", "zero_error", "mean_error", "fallback_count", "prediction_candidate")} for c in cases]
            model["timing_stable_prediction"] = all(c["prediction_candidate"] for c in cases)
            model["followup_candidate"] = model["timing_stable_prediction"] and next(x["descriptive_repeat"] for x in repeats if x["feature"]==model["feature"])
        groups[cohort] = {"n_pairs": len(subset), "n_games": len({p["game"] for p in subset}), "repetition": repeats, "models": models}
    output = {"protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")), "sources": sources,
              "interpretation": "retrospective_development_2d_observations_not_causal_or_real2sim_intervention",
              "n_clips": len(clips), "clips": public_clips, "pairs": kept, "exclusions": exclusions, "cohorts": groups}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"mapped-tracks.private.npz", **cache)
    (out/"results.private.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("complete", len(clips), "clips", len(kept), "pairs", "excluded", len(exclusions), flush=True)
    for cohort, value in groups.items():
        print(cohort, "repeat", [x["feature"] for x in value["repetition"] if x["descriptive_repeat"]],
              "predictive", [(x["feature"], x["target"]) for x in value["models"] if x["prediction_candidate"]], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.lab.resolve(), args.out)
