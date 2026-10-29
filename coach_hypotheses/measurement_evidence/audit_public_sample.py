"""Read-only inspection of downloaded Ozkaya 2018 samples; no model calibration."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct

import ezc3d
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "artifacts"
SAMPLES = ("S1/1w2d/S1_DP_05_O", "S1/1w2d/S1_DP_06_O", "S1/1w1d/S1_NDP_07_O")


def valid_points(xyz, residual):
    """Coordinates must be finite AND C3D residual must mark them valid."""
    return np.isfinite(xyz).all(axis=0) & np.isfinite(residual) & (residual >= 0)


def event_index(seconds, rate, raw_first_frame, convention):
    """Return a declared candidate; neither convention proves producer timing."""
    frame_float = seconds * rate
    if abs(frame_float - round(frame_float)) > 1e-3:
        raise ValueError("Event is not on the declared sampling grid")
    if convention == "c3d_first_sample_zero":
        raw_frame = int(round(frame_float)) + 1
    elif convention == "legacy_first_frame_at_1_over_rate":
        raw_frame = int(round(frame_float))
    else:
        raise ValueError("Explicit C3D-standard or producer-legacy convention required")
    return raw_frame - raw_first_frame


def fixed_window_speed(xyz_m, valid, index, rate, count=5):
    """Fit only index+1..index+count; never bridge missing samples."""
    indices = np.arange(index + 1, index + count + 1)
    if indices[0] < 0 or indices[-1] >= xyz_m.shape[1] or not valid[indices].all():
        return None
    time = np.arange(count, dtype=float) / rate
    time -= time.mean()
    velocity = xyz_m[:, indices] @ time / (time @ time)
    return float(np.linalg.norm(velocity))


def digest(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest()}


def inspect_trial(stem):
    path = RAW / (stem + ".c3d")
    c = ezc3d.c3d(str(path))
    first, last = struct.unpack_from("<HH", path.read_bytes(), 6)
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    rate = float(c["header"]["points"]["frame_rate"])
    assert c["parameters"]["POINT"]["UNITS"]["value"] == ["mm"]
    xyz = c["data"]["points"][:3] / 1000.0
    residual = c["data"]["meta_points"]["residuals"][0]
    ev = c["parameters"]["EVENT"]
    used = int(ev["USED"]["value"][0])
    times = np.asarray(ev["TIMES"]["value"])
    events = [{"label": ev["LABELS"]["value"][i], "context": ev["CONTEXTS"]["value"][i],
               "generic_flag": int(ev["GENERIC_FLAGS"]["value"][i]),
               "seconds": float(times[0, i] * 60 + times[1, i])} for i in range(used)]
    generic = [e for e in events if e["label"] == "Event" and e["context"] == "General"
               and e["generic_flag"] == 1]
    if len(generic) != 1:
        raise ValueError("Expected exactly one author generic release proxy")
    indices = {name: event_index(generic[0]["seconds"], rate, first, name) for name in
               ["c3d_first_sample_zero", "legacy_first_frame_at_1_over_rate"]}
    side = "L" if "_NDP_" in stem else "R"  # S1 is right dominant in author Table 1.
    selected = ["Ball", side + "3MC", side + "5MC"]
    markers = {}
    for name in selected:
        i = labels.index(name)
        valid = valid_points(xyz[:, i], residual[i])
        idx = np.flatnonzero(valid)
        markers[name] = {"valid_count": int(valid.sum()), "total_count": len(valid),
                         "valid_ratio": float(valid.mean()),
                         "first_valid_array_index": int(idx[0]) if len(idx) else None,
                         "last_valid_array_index": int(idx[-1]) if len(idx) else None}
    ball = labels.index("Ball")
    hand = labels.index(side + "3MC")
    pair_valid = valid_points(xyz[:, ball], residual[ball]) & valid_points(xyz[:, hand], residual[hand])
    candidates = []
    compare_indices = set()
    for convention, index in indices.items():
        window = np.arange(max(0, index - 5), min(xyz.shape[2], index + 6))
        compare_indices.update(window.tolist())
        distances = [float(np.linalg.norm(xyz[:, ball, i] - xyz[:, hand, i]) * 1000)
                     if pair_valid[i] else None for i in window]
        event_markers = {}
        for name in selected:
            i = labels.index(name)
            valid = valid_points(xyz[:, i], residual[i])
            event_markers[name] = {"event_pm5_valid_count": int(valid[window].sum()),
                                   "fixed_after_proxy_5_sample_speed_m_s": fixed_window_speed(xyz[:, i], valid, index, rate)}
        candidates.append({"convention": convention, "array_index": index, "raw_frame": first + index,
                           "pm5_offsets": (window-index).tolist(), "ball_to_3MC_distance_mm": distances,
                           "markers": event_markers})
    csv_path = RAW / (stem + ".csv")
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    cols = [j for j, label in enumerate(rows[1]) if label == "Ball"]
    assert len(cols) == 3
    numeric_rows = {int(row[0]): row for row in rows[5:] if row and row[0].isdigit()}
    compare = []
    for i in sorted(compare_indices):
        row = numeric_rows[first + int(i)]
        if pair_valid[i] and all(row[j] for j in cols):
            compare.append(float(np.max(np.abs(np.array([float(row[j]) for j in cols]) - xyz[:, ball, i]))))
    return {"input": digest(path), "csv_input": digest(csv_path), "raw_first_frame": first,
            "raw_last_frame": last, "ezc3d_first_frame": c["header"]["points"]["first_frame"],
            "frame_rate_hz": rate, "point_units": "mm", "point_channels": len(labels),
            "events": events, "source_event_mapping": "UNKNOWN",
            "standard_reference_convention": "c3d_first_sample_zero",
            "selected_source_mapping": None, "release_proxy_candidates": candidates,
            "event_mapping_source": "https://www.c3d.org/HTML/Documents/eventtimes1.htm",
            "csv_matching_proves": "point numbering and coordinate agreement only, not event-time origin",
            "markers": markers,
            "csv_ball_m_vs_c3d_mm_max_difference_m": max(compare) if compare else None,
            "csv_event_seconds": rows[5][rows[1].index("Event")],
            "csv_unit_header_present": False,
            "full_hand_orientation_observed": False,
            "ball_center_definition_verified": False,
            "independent_contact_release_verified": False}


def inspect_static(path):
    c = ezc3d.c3d(str(path))
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    xyz = c["data"]["points"][:3]
    residual = c["data"]["meta_points"]["residuals"][0]
    result = {}
    for side in ["R", "L"]:
        names = [side + name for name in ["WRL", "WRM", "FR1", "FR2", "FR3", "FR4"]]
        ids = [labels.index(n) for n in names]
        valid = np.logical_and.reduce([valid_points(xyz[:, i], residual[i]) for i in ids])
        frames = np.flatnonzero(valid)
        cluster = np.median(xyz[:, ids[2:], :][:, :, frames], axis=2).T
        singular = np.linalg.svd(cluster - cluster.mean(axis=0), compute_uv=False)
        result[side] = {"markers": names, "all_six_valid_count": int(valid.sum()),
                        "cluster_singular_values_mm": singular.tolist(),
                        "noncollinear_rank_at_least_2": bool(singular[1] > 1e-6),
                        "virtual_wrist_is_estimate_not_bone_center": True}
    return {"input": digest(path), "frames": xyz.shape[2], "sides": result}


def main():
    result = {"purpose": "source feasibility audit; no calibration or athlete intervention",
              "revision": "C3D-standard and unverified producer-legacy timing candidates both retained",
              "supersedes": "history_initial_mapping_20261029/sample_audit.json",
              "selection": "three exploratory samples listed before audit script, not population evidence",
              "trials": [inspect_trial(s) for s in SAMPLES],
              "statics": [inspect_static(RAW / name) for name in
                          ["S1/1w2d/S1_static_01_D.c3d", "S1/1w2d/S1_static_02_N.c3d",
                           "S1/1w1d/S1_static_03_N.c3d"]],
              "downloads": [digest(RAW / n) for n in ["S1.rar", "S1_csv.rar", "Trialinfo.xlsx"]],
              "unchanged": {"stage2_distributions": True, "scientific_gate": True, "routine_paused": True}}
    (HERE / "sample_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps([{ "file": Path(t["input"]["path"]).name, "mapping": t["source_event_mapping"],
                        "candidates": [{"convention": c["convention"], "index": c["array_index"],
                                        "ball": c["markers"]["Ball"]} for c in t["release_proxy_candidates"]]}
                      for t in result["trials"]], indent=2))


if __name__ == "__main__":
    main()
