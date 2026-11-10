"""Attribute the change in flight metrics on the two linked Rapsodo pitches."""
import argparse
import hashlib
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np

from experiments.linked_highspeed_release.study import launch_basis, read_pitches
from experiments.rapsodo_camera_calibration.study import PRINCIPAL, positions, project_path, revised_inputs
from experiments.seam_axis_flight.study import optical_inputs
from pitch_outcomes import flight


DISTANCE_M = 16.8
PLATE_DEPTH_M = 17 * .0254
FACTOR_NAMES = ("efficiency", "lift_direction", "launch_direction")
FIELDS = ("ivb_cm", "plate_height_cm")
PRIMARY = ("before", "calibrated", "old_tilt", "recorded_tilt", "lyu_lift", "near_plate", "far_plate")


def lift_coefficient(spin_factor, law):
    if law == "sawicki":
        return 1.5 * spin_factor if spin_factor <= .1 else .09 + .6 * spin_factor
    if law == "lyu_four_seam" and .05 <= spin_factor <= .20:
        return float(np.interp(spin_factor, [.05, .10, .15, .20], [.15, .16, .17, .21]))
    raise ValueError(f"Unsupported lift law or spin factor: {law}, {spin_factor}")


def lift_tilt(axis, basis):
    forward, right, up = basis
    lift = np.cross(axis, forward)
    return float(np.rad2deg(np.arctan2(lift @ right, lift @ up)))


def run_flight(pitch, efficiency, tilt_deg, distance=DISTANCE_M, law="sawicki", dt=.001):
    forward, right, up = launch_basis(pitch)
    theta = np.deg2rad(tilt_deg)
    axis = np.cross(forward, np.cos(theta) * up + np.sin(theta) * right)
    speed = pitch["speed_mph"] * .44704
    spin_factor = .0366 * (2 * np.pi * pitch["rpm"] * efficiency / 60) / speed
    cl = lift_coefficient(spin_factor, law)
    position = np.array([0., 0., pitch["release_height_ft"] * .3048])
    velocity = speed * forward
    path, flight_time = flight(position, velocity, axis, cl, .35, distance, dt)
    no_lift, _ = flight(position, velocity, axis, 0., .35, distance, dt)
    vacuum_time = distance / velocity[1]
    vacuum_height = position[2] + velocity[2] * vacuum_time - .5 * 9.80665 * vacuum_time**2
    movement = 100 * (path[-1, [0, 2]] - no_lift[-1, [0, 2]])
    return {
        "distance_m": distance, "time_s": flight_time, "efficiency_pct": 100 * efficiency,
        "spin_factor": spin_factor, "cl": cl, "lift_law": law, "tilt_deg": tilt_deg,
        "initial_vertical_lift_cl": float(cl * np.cos(theta) * up[2]),
        "horizontal_break_cm": float(movement[0]), "ivb_cm": float(movement[1]),
        "plate_height_cm": float(100 * path[-1, 2]),
        "no_lift_height_cm": float(100 * no_lift[-1, 2]),
        "gravity_only_height_cm": float(100 * vacuum_height),
        "ivb_gravity_only_cm": float(100 * (path[-1, 2] - vacuum_height)),
        "path_m": path[:, :3], "no_lift_path_m": no_lift[:, :3],
    }


def averaged_contributions(factorial):
    """Average marginal changes across all orders of replacing the three inputs."""
    totals = np.zeros((len(FACTOR_NAMES), len(FIELDS)))
    orders = tuple(permutations(range(len(FACTOR_NAMES))))
    for order in orders:
        state = [0] * len(order)
        for factor in order:
            previous = factorial[tuple(state)]
            state[factor] = 1
            following = factorial[tuple(state)]
            totals[factor] += [following[field] - previous[field] for field in FIELDS]
    return {factor: dict(zip(FIELDS, values.tolist()))
            for factor, values in zip(FACTOR_NAMES, totals / len(orders))}


def check_previous(modes, prior):
    for here, earlier in (("before", "before"), ("calibrated", "rotation_and_launch")):
        for field in FIELDS:
            if not np.isclose(modes[here][field], prior["flights"][earlier][field], rtol=0, atol=1e-8):
                raise ValueError(f"Previous pitch {prior['pitch_id']} does not reproduce for {here} {field}")


def late_projection(modes, prior, trace):
    late = prior["late"]
    observed = np.asarray(late["observed_centers_px"])
    origin = positions(trace[:3], prior["focal_px"], PRINCIPAL)[0]
    basis = np.asarray(prior["camera_basis"])
    projected = {
        key: project_path(mode["path_m"], late["frames"], trace[0][0], origin, basis, prior["focal_px"], 500)
        for key, mode in modes.items()
    }
    common = np.logical_and.reduce([np.isfinite(values).all(axis=1) for values in projected.values()])
    if not common.any():
        raise ValueError("No common late frames across flight assumptions")
    return {
        "frames": np.asarray(late["frames"])[common].tolist(),
        "observed_px": observed[common].tolist(),
        "modes": {
            key: {"projected_px": values[common].tolist(),
                  "mean_px": float(np.linalg.norm(values[common] - observed[common], axis=1).mean())}
            for key, values in projected.items()
        },
    }


def scenarios(pitch, prior, rotation, dt=.001):
    yaw, elevation = prior["image_horizontal_vertical_deg"]
    optical_launch = dict(pitch, horizontal_angle_deg=yaw, vertical_angle_deg=elevation)
    original_cam_axis = rotation["camera_axes"]["fit_0.60"]
    _, _, old_eta, old_tilt = optical_inputs(original_cam_axis, launch_basis(pitch))
    u, v, towards = original_cam_axis
    field_axis = np.asarray(prior["camera_basis"]) @ [u, -towards, v]
    new_eta, _, transverse = revised_inputs(pitch, optical_launch, field_axis)
    new_tilt = lift_tilt(transverse, launch_basis(optical_launch))
    hours, minutes = map(int, pitch["direction"].split(":"))
    recorded_tilt = 30 * (hours % 12) + .5 * minutes

    etas = (old_eta, new_eta)
    tilts = (old_tilt, new_tilt)
    launches = (pitch, optical_launch)
    factorial = {
        (i, j, k): run_flight(launches[k], etas[i], tilts[j], dt=dt)
        for i, j, k in product(range(2), repeat=3)
    }
    modes = {
        "before": factorial[0, 0, 0],
        "calibrated": factorial[1, 1, 1],
        "old_tilt": factorial[1, 0, 1],
        "recorded_tilt": run_flight(optical_launch, new_eta, recorded_tilt, dt=dt),
        "lyu_lift": run_flight(optical_launch, new_eta, new_tilt, law="lyu_four_seam", dt=dt),
        "near_plate": run_flight(optical_launch, new_eta, new_tilt, DISTANCE_M - PLATE_DEPTH_M, dt=dt),
        "far_plate": run_flight(optical_launch, new_eta, new_tilt, DISTANCE_M + PLATE_DEPTH_M, dt=dt),
    }
    return modes, factorial


def compare(workbook, prior, rotations, traces):
    raw = {pitch["pitch_id"]: pitch for pitch in read_pitches(workbook)}
    if {row["pitch_id"] for row in prior["pitches"]} != {4, 8} or prior["capture_fps_assumed"] != 500:
        raise ValueError("Expected the previous two-pitch, nominal 500 fps calibration")
    rows = []
    for previous in prior["pitches"]:
        key = previous["pitch_id"]
        pitch = raw[key]
        modes, factorial = scenarios(pitch, previous, rotations[str(key)])
        check_previous(modes, previous)
        fine, _ = scenarios(pitch, previous, rotations[str(key)], dt=.0005)
        reference = {"ivb_cm": pitch["ivb_in"] * 2.54, "plate_height_cm": pitch["plate_height_in"] * 2.54}
        reference["implied_no_lift_height_cm"] = reference["plate_height_cm"] - reference["ivb_cm"]
        for name, mode in modes.items():
            mode["errors"] = {field: mode[field] - reference[field] for field in FIELDS}
            mode["ivb_gravity_only_error_cm"] = mode["ivb_gravity_only_cm"] - reference["ivb_cm"]
            mode["half_step_delta_cm"] = {field: fine[name][field] - mode[field] for field in FIELDS}
        rows.append({
            "pitch_id": key, "reference": reference, "speed_kmh": pitch["speed_mph"] * 1.609344,
            "video_path_yz_m_registered": previous["video_path_yz_m_registered"],
            "late_video_path_yz_m_registered": previous["late"]["video_path_yz_m_registered"],
            "late_projection": late_projection(modes, previous, traces[str(key)]),
            "contributions": averaged_contributions(factorial),
            "factorial": {
                "".join(map(str, indices)): {field: run[field] for field in FIELDS}
                for indices, run in factorial.items()
            },
            "modes": {name: {**mode, "path_m": mode["path_m"].tolist(), "no_lift_path_m": mode["no_lift_path_m"].tolist()}
                      for name, mode in modes.items()},
        })
    mae = {
        name: {**{field: float(np.mean([abs(row["modes"][name]["errors"][field]) for row in rows])) for field in FIELDS},
               "ivb_gravity_only_cm": float(np.mean([abs(row["modes"][name]["ivb_gravity_only_error_cm"]) for row in rows]))}
        for name in PRIMARY
    }
    return {"workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(), "mae": mae, "pitches": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook", "calibration", "rotation", "track_4", "track_8", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    paths = (args.calibration, args.rotation, args.track_4, args.track_8)
    prior, rotations, track_4, track_8 = (json.loads(path.read_text(encoding="utf-8-sig")) for path in paths)
    report = compare(args.workbook, prior, rotations, {"4": track_4["ball_trace"], "8": track_8["ball_trace"]})
    report["input_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
