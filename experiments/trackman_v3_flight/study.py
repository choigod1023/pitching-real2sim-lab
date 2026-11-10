"""Compare matched TrackMan V3 pitches using conditional ball flight."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from decimal import Decimal
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from pitch_outcomes import flight_acceleration


FT = .3048
IN = .0254
MPH = .44704
G = 9.80665
PLATE_FT = 17 / 12
DISTANCE_FT = 60.5 - PLATE_FT
INPUTS = ("RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
          "VertRelAngle", "HorzRelAngle")
VARIANTS = {
    "backspin": (False, False, 1.),
    "release_angles": (True, False, 1.),
    "inferred_axis": (False, True, 1.),
    "both": (True, True, 1.),
    "both_active_80pct": (True, True, .8),
}


def read_cohort(source):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    groups, pitches = defaultdict(list), []
    game = previous = 0
    for row in raw:
        number = int(row["PitchNo"])
        if number == 1:
            game += 1
            previous = 0
        if number != previous + 1:
            raise ValueError("Noncontiguous PitchNo in the exported game block")
        previous = number
        if (row["TaggedPitchType"], row["AutoPitchType"]) != ("Fastball", "Four-Seam"):
            continue
        measured = np.array([float(row[k]) for k in (*INPUTS, "InducedVertBreak", "HorzBreak")])
        if not np.isfinite(measured).all() or measured[0] <= 0 or measured[1] <= 0:
            raise ValueError(f"Invalid model input or comparison in game {game}, pitch {number}")
        groups[(game, row["PitcherId"])].append(len(pitches))
        pitches.append(dict(row, game_block=game))
    speeds = [Decimal(row["RelSpeed"]) * Decimal("1.609344") for row in pitches]
    pairs = [(i, j) for indices in groups.values() for i, j in combinations(indices, 2)
             if abs(speeds[i] - speeds[j]) <= Decimal("1.0")]
    return pitches, np.array(pairs, dtype=int).reshape(-1, 2), len(raw), game


def release_inputs(rows, use_angles, use_axis, active):
    speed = MPH * np.array([float(row["RelSpeed"]) for row in rows])
    rpm = np.array([float(row["SpinRate"]) for row in rows])
    xz = FT * np.array([[float(row["RelSide"]), float(row["RelHeight"])] for row in rows])
    distance = FT * (DISTANCE_FT - np.array([float(row["Extension"]) for row in rows]))
    if use_angles:
        azimuth, elevation = np.deg2rad([
            [float(row[key]) for row in rows] for key in ("HorzRelAngle", "VertRelAngle")
        ])
    else:
        azimuth = elevation = np.zeros(len(rows))
    unit = np.column_stack((np.cos(elevation) * np.sin(azimuth),
                            np.cos(elevation) * np.cos(azimuth), np.sin(elevation)))
    right = np.column_stack((np.cos(azimuth), -np.sin(azimuth), np.zeros(len(rows))))
    up = np.cross(right, unit)
    tilt = (np.deg2rad([float(row["SpinAxis"]) - 180. for row in rows])
            if use_axis else np.zeros(len(rows)))
    axis = np.cross(unit, np.cos(tilt)[:, None] * up + np.sin(tilt)[:, None] * right)
    spin_parameter = .0366 * (2 * np.pi * active * rpm / 60) / speed
    cl = np.where(spin_parameter <= .1, 1.5 * spin_parameter, .09 + .6 * spin_parameter)
    return xz, speed[:, None] * unit, distance, axis, cl


def integrate_to_plate(xz, velocity, distance, axis, cl, *, steps=256, cd=.35, keep=()):
    """Integrate all pitches in normalized forward distance with RK4."""
    state = np.column_stack((xz, velocity, np.zeros(len(xz))))
    if steps <= 0 or np.any(distance <= 0) or np.any(velocity[:, 1] <= 0):
        raise ValueError("Need positive steps, distance and forward velocity")
    kept = list(keep)
    path = [state[kept].copy()] if kept else []
    below_ground = state[:, 1] < 0
    step = 1 / steps

    def derivative(values):
        vel = values[:, 2:5]
        acc = flight_acceleration(vel, axis, cl[:, None], cd, G)
        return np.column_stack((vel[:, [0, 2]], acc, np.ones(len(vel)))) * (
            distance[:, None] / vel[:, 1:2]
        )

    for _ in range(steps):
        a = derivative(state)
        b = derivative(state + a * step / 2)
        c = derivative(state + b * step / 2)
        d = derivative(state + c * step)
        state += (a + 2 * b + 2 * c + d) * step / 6
        below_ground |= state[:, 1] < 0
        if kept:
            path.append(state[kept].copy())
    if not np.isfinite(state).all() or np.any(state[:, 3] <= 0):
        raise ValueError("Invalid terminal flight state")
    return state, np.asarray(path), below_ground


def simulate(rows, config, *, steps=256, keep=()):
    xz, velocity, distance, axis, cl = release_inputs(rows, *config)
    end, trace, ground = integrate_to_plate(xz, velocity, distance, axis, cl, steps=steps, keep=keep)
    straight_height = xz[:, 1] + distance * velocity[:, 2] / velocity[:, 1]
    ivb_cm = 100 * (end[:, 1] - straight_height + .5 * G * end[:, 5]**2)
    return ivb_cm, distance, end, trace, ground


def reference_trajectory(row, count=257):
    coefficients = np.array([[float(row[f"PitchTrajectory{axis}c{i}"]) for i in range(3)]
                             for axis in "XYZ"])
    a, b, c = coefficients[0, ::-1].copy()
    roots = np.roots([a, b, c - PLATE_FT])
    times = [float(t.real) for t in roots if np.isreal(t) and 0 < t.real < 1.]
    if len(times) != 1:
        raise ValueError("No unique forward pass through the front of home plate")
    t = np.linspace(0., times[0], count)
    x, y, z = coefficients @ np.vstack((np.ones(count), t, t*t))
    return np.column_stack((-z, PLATE_FT - x, y)) * FT, times[0]


def metrics(values, observed, pairs, cm=True):
    i, j = pairs.T
    estimate, actual = values[j] - values[i], observed[j] - observed[i]
    correct = np.sign(estimate) == np.sign(actual)
    summary = {"correct": int(correct.sum()), "pairs": len(pairs),
               "accuracy_pct": 100 * float(correct.mean()), "ties": int(np.sum(estimate == 0))}
    if cm:
        summary["pair_delta_mae_cm"] = float(np.mean(np.abs(estimate - actual)))
    return summary


def run(source):
    rows, pairs, total, games = read_cohort(source)
    observed = np.array([float(row["InducedVertBreak"]) * 2.54 for row in rows])
    rpm = np.array([float(row["SpinRate"]) for row in rows])
    keep = [next(i for i, row in enumerate(rows) if row["game_block"] == 1 and int(row["PitchNo"]) == n)
            for n in (17, 18)]
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as pool:
        jobs = {name: pool.submit(simulate, rows, config, keep=keep if name == "both" else ())
                for name, config in VARIANTS.items()}
        simulations = {name: job.result() for name, job in jobs.items()}
    results = {"rpm_only": metrics(rpm, observed, pairs, cm=False)}
    for name, simulation in simulations.items():
        summary = metrics(simulation[0], observed, pairs)
        summary["pitch_mae_cm"] = float(np.mean(np.abs(simulation[0] - observed)))
        summary["below_ground_pitches"] = int(simulation[-1].sum())
        results[name] = summary
    raw_below = np.array([float(row["PlateLocHeight"]) < 0 for row in rows])
    invalid = np.logical_or.reduce([
        raw_below, *(simulation[-1] for name, simulation in simulations.items() if name != "both_active_80pct")
    ])
    physical_pairs = pairs[~invalid[pairs].any(axis=1)]
    physical = {"rpm_only": metrics(rpm, observed, physical_pairs, cm=False)}
    physical.update({name: metrics(simulations[name][0], observed, physical_pairs)
                     for name in VARIANTS if name != "both_active_80pct"})
    ivb, distance, end, paths, _ = simulations["both"]
    fine = simulate(rows, VARIANTS["both"], steps=512)[0]
    examples = []
    for k, idx in enumerate(keep):
        row = rows[idx]
        reference, reference_time = reference_trajectory(row, len(paths))
        progress = np.linspace(-distance[idx], 0, len(paths))
        sim = np.column_stack((paths[:, k, 0], progress, paths[:, k, 1]))
        examples.append({
            "pitch_number": int(row["PitchNo"]),
            "speed_kmh": float(row["RelSpeed"]) * 1.609344,
            "rpm": rpm[idx],
            "spin_axis_deg": float(row["SpinAxis"]),
            "release_angle_h_v_deg": [float(row[key]) for key in ("HorzRelAngle", "VertRelAngle")],
            "recorded_ivb_cm": observed[idx], "model_ivb_cm": ivb[idx],
            "recorded_plate_height_cm": 100 * FT * float(row["PlateLocHeight"]),
            "recorded_plate_lateral_cm": 100 * FT * float(row["PlateLocSide"]),
            "model_plate_height_cm": 100 * end[idx, 1],
            "model_plate_lateral_cm": 100 * end[idx, 0],
            "recorded_zone_time_s": float(row["ZoneTime"]),
            "reference_time_s": reference_time, "model_time_s": end[idx, 5],
            "reference_xyz_m": reference.tolist(), "model_xyz_m": sim.tolist(),
        })
    theta = np.array([float(row["SpinAxis"]) - 180 for row in rows])
    clock = np.array([30 * (int(row["Tilt"].split(":")[0]) % 12)
                      + .5 * int(row["Tilt"].split(":")[1]) for row in rows])
    hb = np.array([float(row["HorzBreak"]) * 2.54 for row in rows])
    ivb_reference = np.array([
        float(row["VertBreak"]) * 2.54 + 50 * G * float(row["ZoneTime"])**2 for row in rows
    ])
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": total, "game_blocks": games, "four_seam_pitches": len(rows),
        "pairs": len(pairs), "paired_pitchers": len({rows[i]["PitcherId"] for i in pairs[:, 0]}),
        "direction_results": results,
        "physical_subset": {"raw_plate_below_ground_pitches": int(raw_below.sum()),
                            "excluded_pairs": len(pairs) - len(physical_pairs),
                            "direction_results": physical},
        "axis_provenance": "TrackMan V3 movement-inferred; no independent axis or gyro angle in CSV",
        "max_axis_clock_difference_deg": float(np.max(np.abs((theta - clock + 180) % 360 - 180))),
        "median_axis_movement_angle_difference_deg": float(np.median(
            np.abs((theta - np.rad2deg(np.arctan2(hb, observed)) + 180) % 360 - 180))),
        "max_csv_gravity_convention_residual_cm": float(np.max(np.abs(observed - ivb_reference))),
        "max_step_halving_ivb_difference_cm": float(np.max(np.abs(fine - ivb))),
        "step_halving_sign_matches": metrics(fine, observed, pairs)["correct"],
        "examples": examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.csv), indent=2, allow_nan=False), encoding="utf-8")
