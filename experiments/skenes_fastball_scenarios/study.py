"""Paired Skenes flight scenarios initialized from recorded ball states."""
import argparse
import json
from math import atan2, cos, degrees, hypot, pi, radians, sin, sqrt
from pathlib import Path

import numpy as np

from pitch_outcomes import flight

FT = .3048
MPH = 1.609344
FRONT_FT = 17 / 12
NOMINAL = (.65, .85, .35)


def release_state(pitch):
    row = pitch["row"]
    x, y, z = (float(row[f"release_pos_{dim}"]) for dim in "xyz")
    v = np.array([float(row[f"v{dim}0"]) for dim in "xyz"])
    a = np.array([float(row[f"a{dim}"]) for dim in "xyz"])
    speed = float(row["release_speed"]) * MPH / 3.6
    if not np.isfinite(np.r_[x, y, z, v, a, speed]).all() or not (50 < y < 61 and v[1] < 0 and speed > 0):
        raise ValueError("Invalid release or 50 ft state")
    discriminant = v[1] ** 2 + 2 * a[1] * (y - 50)
    if discriminant <= 0:
        raise ValueError("Cannot back-project to the release plane")
    t = 2 * (y - 50) / (v[1] - sqrt(discriminant))
    back = (v + t * a) * [1, -1, 1]
    velocity = back / np.linalg.norm(back) * speed
    return np.array([x * FT, 0., z * FT]), velocity, (y - FRONT_FT) * FT


def slot_offset(low_deg, high_deg, radius_m):
    low, high = radians(low_deg), radians(high_deg)
    return radius_m * np.array([cos(low) - cos(high), 0., sin(high) - sin(low)])


def lift_model(rpm, active, axis_deg, speed):
    """Fixed launch-spin coefficient from Sawicki et al., Eq. 13."""
    s = .0366 * (rpm * active * 2 * pi / 60) / speed
    cl = 1.5 * s if s <= .1 else .09 + .6 * s
    angle = radians(axis_deg - 180)
    return cl, (cos(angle), 0., sin(angle))


def scenario(low, high, mode, radius, active, cd, dt):
    p, v, distance = release_state(low)
    rpm = float(low["row"]["release_spin_rate"])
    speed = np.linalg.norm(v)
    if mode == "slot_only":
        p += slot_offset(low["arm_angle_deg"], high["arm_angle_deg"], radius)
    elif mode == "direction_only":
        _, direction, _ = release_state(high)
        v = direction / np.linalg.norm(direction) * speed
    elif mode == "rpm_only":
        rpm = float(high["row"]["release_spin_rate"])
    elif mode != "baseline":
        raise ValueError(f"Unrecognized mode {mode!r}")

    cl, axis = lift_model(rpm, active, float(low["row"]["spin_axis"]), speed)
    path, elapsed = flight(p, v, axis, cl, cd, distance, dt)
    without_lift, _ = flight(p, v, axis, 0., cd, distance, dt)
    return {
        "initial_xyz_m": p.tolist(),
        "initial_velocity_m_s": v.tolist(),
        "initial_speed_kmh": float(np.linalg.norm(v) * 3.6),
        "direction_deg": [degrees(atan2(v[0], v[1])), degrees(atan2(v[2], hypot(v[0], v[1])))],
        "rpm_input": rpm,
        "cl": cl,
        "flight_s": elapsed,
        "plate_xz_cm": (path[-1, [0, 2]] * 100).tolist(),
        "lift_displacement_xz_cm": ((path[-1, [0, 2]] - without_lift[-1, [0, 2]]) * 100).tolist(),
        "path_xyz_m": path[:, :3].tolist(),
        "t_s": [*(i * dt for i in range(len(path) - 1)), elapsed],
    }


def compare_pair(pair, slot_delta_deg=None):
    low, high = pair["low"], pair["high"]
    slot_input = high if slot_delta_deg is None else {**high, "arm_angle_deg": low["arm_angle_deg"] + slot_delta_deg}
    modes = ("baseline", "slot_only", "direction_only", "rpm_only")

    def run(mode, radius, active, cd, dt):
        return scenario(low, slot_input if mode == "slot_only" else high, mode, radius, active, cd, dt)

    nominal = {m: run(m, *NOMINAL, .0005) for m in modes}
    base = nominal["baseline"]

    def delta(reference, changed, metric):
        return (np.asarray(changed[metric]) - reference[metric]).tolist()

    grid = []
    for cd in (.30, .35, .40):
        for active in (.70, .85, 1.0):
            ref = run("baseline", .65, active, cd, .001)
            for mode in modes[1:]:
                for radius in ((.55, .65, .75) if mode == "slot_only" else (.65,)):
                    varied = run(mode, radius, active, cd, .001)
                    grid.append({
                        "mode": mode, "radius_m": radius, "active_spin_assumed": active, "cd": cd,
                        "delta_plate_xz_cm": delta(ref, varied, "plate_xz_cm"),
                        "delta_lift_xz_cm": delta(ref, varied, "lift_displacement_xz_cm"),
                    })
    return {
        "scope": "conditional_ball_flight_no_identified_joint_to_ball_dynamics",
        "game": low["game"], "date": low["date"],
        "play_ids": [low["play_id"], high["play_id"]],
        "slot_change_deg": slot_input["arm_angle_deg"] - low["arm_angle_deg"],
        "slot_source": "matched_pair" if slot_delta_deg is None else "external_scenario",
        "observed": {
            "release_speed_kmh": [low["speed_kmh"], high["speed_kmh"]],
            "rpm": [low["spin_rpm"], high["spin_rpm"]],
            "pfx_z_cm": [low["vertical_break_cm"], high["vertical_break_cm"]],
            "arm_angle_deg": [low["arm_angle_deg"], high["arm_angle_deg"]],
            "plate_xz_cm": [[float(p["row"][f"plate_{d}"]) * FT * 100 for d in "xz"] for p in (low, high)],
        },
        "nominal": nominal,
        "delta": {m: {
            "plate_xz_cm": delta(base, nominal[m], "plate_xz_cm"),
            "lift_xz_cm": delta(base, nominal[m], "lift_displacement_xz_cm"),
            "initial_xyz_cm": (100 * np.array(delta(base, nominal[m], "initial_xyz_m"))).tolist(),
        } for m in modes[1:]},
        "sensitivity": grid,
        "max_timestep_delta_cm": max(
            float(np.max(np.abs(delta(nominal[m], run(m, *NOMINAL, .001), "plate_xz_cm"))))
            for m in modes
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--slot-delta-deg", type=float)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8-sig"))
    pair = next(pair for game in cohort["games"] for pair in game["chosen"])
    result = compare_pair(pair, args.slot_delta_deg)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("observed", "delta", "max_timestep_delta_cm")}, indent=2))


if __name__ == "__main__":
    main()
