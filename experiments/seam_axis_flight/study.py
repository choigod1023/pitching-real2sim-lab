"""Compare video-derived axes and instrument-conditioned flights on the linked example pitches."""
import argparse
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.linked_highspeed_release.study import launch_basis, predict_movement, read_pitches


def optical_inputs(camera_axis, basis, yaw=0., pitch=0.):
    forward, right, up = np.asarray(basis, dtype=float)
    axis = np.asarray(camera_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.linalg.norm(axis):
        raise ValueError("Expected a finite nonzero axis in camera coordinates")
    axis = axis / np.linalg.norm(axis)
    field = Rotation.from_euler("zx", [yaw, pitch], degrees=True).apply([axis[0], -axis[2], axis[1]])
    transverse = field - np.dot(field, forward) * forward
    active = np.linalg.norm(transverse)
    if active < 1e-9:
        raise ValueError("Indeterminate lift direction for fully gyro-aligned spin")
    transverse /= active
    lift = np.cross(transverse, forward)
    tilt = np.rad2deg(np.arctan2(np.dot(lift, right), np.dot(lift, up)))
    return field, transverse, float(active), float(tilt)


def summarize_flight(pitch, movement, path):
    ivb = float(movement[1])
    plate = float(100 * path[-1, 2])
    return {
        "ivb_cm": ivb, "plate_height_cm": plate,
        "ivb_error_cm": ivb - pitch["ivb_in"] * 2.54,
        "plate_height_error_cm": plate - pitch["plate_height_in"] * 2.54,
        "path_yz_m": path[:, 1:3].tolist(),
    }


def compare(workbook, observed):
    pitches = [row for row in read_pitches(workbook) if row["pitch_type"] == "FB" and row["video"]]
    selected = [row for row in pitches if str(row["pitch_id"]) in observed]
    if len(selected) != 2:
        raise ValueError("This exploratory comparison requires both linked video pitches")
    shared_rate = float(np.mean([row["true_rpm"] / row["rpm"] for row in selected]))
    rows = []
    for pitch in selected:
        rotation = observed[str(pitch["pitch_id"])]
        camera = rotation["camera_axes"]["fit_0.60"]
        basis = launch_basis(pitch)
        field, axis, active, tilt = optical_inputs(camera, basis)
        alternatives = np.array([
            optical_inputs(camera, basis, yaw, angle)[2:]
            for yaw, angle in product((-5, 0, 5), repeat=2)
        ])
        runs = {
            "shared_efficiency_recorded_direction": predict_movement(pitch, active_rpm=shared_rate * pitch["rpm"]),
            "recorded_direction_efficiency": predict_movement(pitch),
            "video_direction_efficiency": predict_movement(pitch, spin_axis=axis, active_rpm=active * pitch["rpm"]),
        }
        rows.append({
            "pitch_id": pitch["pitch_id"], "speed_kmh": pitch["speed_mph"] * 1.609344,
            "instrument_rpm": pitch["rpm"], "video_rpm_at_nominal_fps": rotation["optical_rpm_at_nominal_capture_fps"],
            "recorded_ivb_cm": pitch["ivb_in"] * 2.54, "recorded_plate_height_cm": pitch["plate_height_in"] * 2.54,
            "recorded_active_fraction": pitch["true_rpm"] / pitch["rpm"],
            "recorded_spin_clock": pitch["direction"],
            "camera_axis_right_up_towards_camera": camera, "assumed_field_axis_right_forward_up": field.tolist(),
            "active_fraction_using_camera_view_only": float(np.hypot(*camera[:2])),
            "active_fraction_using_recorded_release_direction": active, "video_lift_tilt_deg": tilt,
            "active_fraction_camera_geometry_range": [float(alternatives[:, 0].min()), float(alternatives[:, 0].max())],
            "lift_tilt_camera_geometry_range_deg": [float(alternatives[:, 1].min()), float(alternatives[:, 1].max())],
            "flights": {name: summarize_flight(pitch, *result) for name, result in runs.items()},
        })
    pairs = [
        {"pitch_ids": [a["pitch_id"], b["pitch_id"]], "gap_kmh": abs(a["speed_kmh"] - b["speed_kmh"])}
        for a, b in combinations(rows, 2)
    ]
    modes = list(rows[0]["flights"])
    absolute_errors = {
        mode: {
            f"{key}_mae_cm": float(np.mean([abs(row["flights"][mode][f"{key}_error_cm"]) for row in rows]))
            for key in ("ivb", "plate_height")
        }
        for mode in modes
    }
    ivb_changes = {"recorded": rows[1]["recorded_ivb_cm"] - rows[0]["recorded_ivb_cm"]}
    for mode in modes:
        ivb_changes[mode] = rows[1]["flights"][mode]["ivb_cm"] - rows[0]["flights"][mode]["ivb_cm"]
    return {
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "four_seam_pitches_with_linked_video": len(pitches), "shared_instrument_efficiency": shared_rate,
        "same_pitcher_verified_by_workbook": False,
        "pairs": pairs, "eligible_pairs_for_direction_test": 0,
        "exploratory_ivb_change_8_minus_4_cm": ivb_changes, "exploratory_two_pitch_errors": absolute_errors,
        "rpm_baseline_accuracy": None, "video_axis_direction_accuracy": None,
        "scope": "Exploratory flights with matched endpoint records; no measured flight path or camera calibration",
        "pitches": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook", "rotation", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    estimate = json.loads(args.rotation.read_text(encoding="utf-8"))
    args.output.write_text(json.dumps(compare(args.workbook, estimate), indent=2, allow_nan=False), encoding="utf-8")
