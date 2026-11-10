"""Compare paired Statcast four-seams using the public optical spin projection."""
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

from experiments.trackman_v3_flight.study import FT, G, MPH, PLATE_FT, integrate_to_plate
from pitch_outcomes import flight_acceleration


FIELDS = ("release_speed", "release_spin_rate", "spin_axis", "release_pos_x", "release_pos_y",
          "release_pos_z", "vx0", "vy0", "vz0", "pfx_x", "pfx_z", "plate_z", "ax", "ay", "az")
KEY = ("game_pk", "pitcher", "at_bat_number", "pitch_number")


def read_cohort(source):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    candidates = [row for row in raw if row["game_year"] == "2025"
                  and row["game_type"] == "R" and row["pitch_type"] == "FF"]
    rows = []
    for row in candidates:
        try:
            values = {field: float(row[field]) for field in FIELDS}
            key = tuple(int(row[field]) for field in KEY)
        except ValueError:
            continue
        if not np.isfinite(list(values.values())).all() or not (
            values["release_speed"] > 0 and values["release_spin_rate"] > 0
            and values["release_pos_y"] > 50 and values["vy0"] < 0
            and 0 <= values["spin_axis"] <= 360
        ):
            continue
        rows.append({**values, "key": key, "name": row["player_name"],
                     "speed_exact": Decimal(row["release_speed"]) * Decimal("1.609344"),
                     "date": row["game_date"]})
    rows.sort(key=lambda row: row["key"])
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate pitch key")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["key"][:2]].append(i)
    pairs = np.array([
        (i, j) for group in groups.values() for i, j in combinations(group, 2)
        if abs(rows[i]["speed_exact"] - rows[j]["speed_exact"]) <= Decimal("1.0")
    ], dtype=int).reshape(-1, 2)
    if not len(pairs):
        raise ValueError("No complete pairs within 1 km/h")
    return rows, pairs, len(raw), len(candidates)


def column(rows, name):
    return np.array([row[name] for row in rows], dtype=float)


def transverse_axis(velocity, tilt):
    x, z = np.cos(tilt), -np.sin(tilt)
    y = -(x * velocity[:, 0] + z * velocity[:, 2]) / velocity[:, 1]
    axis = np.column_stack((x, y, z))
    return axis / np.linalg.norm(axis, axis=1, keepdims=True)


def lift_coefficient(rows, active):
    speed = MPH * column(rows, "release_speed")
    spin_parameter = .0366 * (2 * np.pi * active * column(rows, "release_spin_rate") / 60) / speed
    return np.where(spin_parameter <= .1, 1.5 * spin_parameter, .09 + .6 * spin_parameter)


def initial_state(rows, active):
    xz = FT * np.column_stack((-column(rows, "release_pos_x"), column(rows, "release_pos_z")))
    distance = FT * (column(rows, "release_pos_y") - PLATE_FT)
    segment = FT * (column(rows, "release_pos_y") - 50)
    v50 = FT * np.column_stack((-column(rows, "vx0"), -column(rows, "vy0"), column(rows, "vz0")))
    speed = MPH * column(rows, "release_speed")
    cl = lift_coefficient(rows, active)
    zero = np.zeros(len(rows))

    def derivative(vel):
        return flight_acceleration(vel, transverse_axis(vel, zero), cl[:, None], .35) / vel[:, 1:2]

    mid = v50 - .5 * segment[:, None] * derivative(v50)
    launch = v50 - segment[:, None] * derivative(mid)
    launch *= speed[:, None] / np.linalg.norm(launch, axis=1, keepdims=True)
    return xz, launch, distance, cl


def simulate(rows, use_axis, active=1., *, keep=(), steps=256, fixed_launch=None):
    if fixed_launch is None:
        xz, velocity, distance, cl = initial_state(rows, active)
    else:
        xz, velocity, distance = fixed_launch
        cl = lift_coefficient(rows, active)
    tilt = np.deg2rad(column(rows, "spin_axis") - 180) if use_axis else np.zeros(len(rows))
    end, trace, ground = integrate_to_plate(
        xz, velocity, distance, transverse_axis(velocity, tilt), cl, keep=keep, steps=steps
    )
    tangent_z = xz[:, 1] + distance * velocity[:, 2] / velocity[:, 1]
    ivb = 100 * (end[:, 1] - tangent_z + .5 * G * end[:, 5]**2)
    return ivb, 100 * end[:, 1], distance, trace, ground


def reference_trajectory(rows, count=257):
    v50 = FT * np.column_stack((-column(rows, "vx0"), -column(rows, "vy0"), column(rows, "vz0")))
    acceleration = FT * np.column_stack((-column(rows, "ax"), -column(rows, "ay"), column(rows, "az")))
    release_pos = FT * np.column_stack((-column(rows, "release_pos_x"), -column(rows, "release_pos_y"),
                                         column(rows, "release_pos_z")))
    plane = -FT * PLATE_FT

    def time_at(y):
        step = y + 50 * FT
        return 2 * step / (v50[:, 1] + np.sqrt(v50[:, 1]**2 + 2 * acceleration[:, 1] * step))

    t0, t1 = time_at(release_pos[:, 1]), time_at(plane)
    elapsed = t1 - t0
    v0 = v50 + acceleration * t0[:, None]
    time = elapsed[:, None] * np.linspace(0., 1., count)
    path = (release_pos[:, None, :] + v0[:, None, :] * time[..., None]
            + .5 * acceleration[:, None, :] * time[..., None]**2)
    path[..., 1] -= plane
    tangent = release_pos[:, 2] + (plane - release_pos[:, 1]) * v0[:, 2] / v0[:, 1]
    ivb = 100 * (path[:, -1, 2] - tangent + .5 * G * elapsed**2)
    return path, ivb


def compare(predicted, measured, pairs, *, physical_units=True):
    i, j = pairs.T
    observed = measured[j] - measured[i]
    changes = predicted[j] - predicted[i]
    signed = observed != 0
    correct = (np.sign(observed) == np.sign(changes)) & signed
    result = {
        "correct": int(correct.sum()), "non_tie_pairs": int(signed.sum()),
        "accuracy_pct": float(100 * correct.sum() / signed.sum()) if signed.any() else None,
        "observed_ties": int((~signed).sum()), "predicted_ties_on_nonzero_pairs": int((signed & (changes == 0)).sum()),
    }
    if physical_units:
        result["pair_delta_mae_cm"] = float(np.mean(np.abs(changes - observed)))
    return result


def run(source, examples=False):
    rows, pairs, total, ff_count = read_cohort(source)
    unique = np.unique(pairs)
    observed_ivb = 100 * FT * column(rows, "pfx_z")
    observed_height = 100 * FT * column(rows, "plate_z")
    rpm = column(rows, "release_spin_rate")
    keep = next(((i, j) for i, j in pairs if j == i + 1), ()) if examples else ()
    settings = {"backspin": (False, 1.), "optical_2d": (True, 1.),
                "backspin_active_80pct": (False, .8), "optical_2d_active_80pct": (True, .8)}
    with ThreadPoolExecutor(max_workers=len(settings)) as pool:
        jobs = {key: pool.submit(simulate, rows, axis, active, keep=keep if active == 1 else ())
                for key, (axis, active) in settings.items()}
        simulations = {key: job.result() for key, job in jobs.items()}
    results = {"rpm_only": compare(rpm, observed_ivb, pairs, physical_units=False)}
    invalid = (observed_height < 0) | simulations["backspin"][-1] | simulations["optical_2d"][-1]
    physical_pairs = pairs[~invalid[pairs].any(axis=1)]
    physical_unique = np.unique(physical_pairs)
    physical = {"rpm_only": compare(rpm, observed_ivb, physical_pairs, physical_units=False)}
    for label, (ivb, height, _, _, ground) in simulations.items():
        results[label] = compare(ivb, observed_ivb, pairs) | {
            "ivb_mae_cm": float(np.mean(np.abs(ivb[unique] - observed_ivb[unique]))),
            "height_mae_cm": float(np.mean(np.abs(height[unique] - observed_height[unique]))),
            "below_ground_pitches": int(ground.sum()),
        }
        if settings[label][1] == 1:
            physical[label] = compare(ivb, observed_ivb, physical_pairs) | {
                "ivb_mae_cm": float(np.mean(np.abs(ivb[physical_unique] - observed_ivb[physical_unique]))),
                "height_mae_cm": float(np.mean(np.abs(height[physical_unique] - observed_height[physical_unique]))),
            }
    ref_end, ref_ivb = reference_trajectory(rows, count=2)
    xz, velocity, distance, cl = initial_state(rows, 1.)
    no_lift, _, _ = integrate_to_plate(
        xz, velocity, distance, transverse_axis(velocity, np.zeros(len(rows))), np.zeros_like(cl)
    )
    alternate = {}
    for name in ("backspin", "optical_2d"):
        ivb = simulations[name][1] - 100 * no_lift[:, 1]
        alternate[name] = compare(ivb, observed_ivb, pairs) | {
            "ivb_mae_cm": float(np.mean(np.abs(ivb[unique] - observed_ivb[unique])))
        }
    phi = np.rad2deg(np.arctan2(-column(rows, "pfx_x"), column(rows, "pfx_z"))) + 180
    axis_gap = (column(rows, "spin_axis") - phi + 180) % 360 - 180
    refined = simulate(rows, True, steps=512)[0]
    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "date": rows[0]["date"], "raw_rows": total, "ff_rows": ff_count,
        "complete_ff_rows": len(rows), "pairs": len(pairs), "unique_paired_pitches": len(unique),
        "pitchers": len({rows[i]["key"][1] for i in unique}),
        "games": len({rows[i]["key"][0] for i in unique}),
        "results": results, "physical_pair_count": len(physical_pairs), "physical_results": physical,
        "no_lift_reference_ivb_results": alternate,
        "reference_plate_z_mae_cm": float(np.mean(np.abs(100 * ref_end[:, -1, 2] - observed_height))),
        "reference_ivb_vs_csv_mae_cm": float(np.mean(np.abs(ref_ivb - observed_ivb))),
        "axis_to_movement_abs_median_deg": float(np.median(np.abs(axis_gap))),
        "step_halving_max_ivb_diff_cm": float(np.max(np.abs(refined - simulations["optical_2d"][0]))),
        "examples": [],
    }
    if keep:
        reference, _ = reference_trajectory([rows[i] for i in keep])
        report["overlay"] = {
            "title": "Hawk-Eye 회전 방향으로 계산한 포심 궤적",
            "subtitle": f"{rows[keep[0]]['name']}  ·  {rows[keep[0]]['date']}",
            "reference_label": "Statcast 적합 궤적", "model_label": "광학 2D 회전 방향",
            "delta_label": "A→B", "note": "실선은 Statcast 적합 궤적입니다. 두 비행 모델의 유효 회전 비율은 100%로 가정했습니다.",
        }
        for k, i in enumerate(keep):
            ivb, height, distance, trace, _ = simulations["optical_2d"]
            y = np.linspace(-distance[i], 0, len(trace))
            base_ivb, base_z, _, base, _ = simulations["backspin"]
            report["examples"].append({
                "pitch_number": rows[i]["key"][-1], "key": rows[i]["key"], "label": "AB"[k],
                "player_name": rows[i]["name"], "speed_kmh": float(rows[i]["speed_exact"]),
                "rpm": rpm[i], "spin_axis_deg": rows[i]["spin_axis"],
                "recorded_ivb_cm": observed_ivb[i], "model_ivb_cm": ivb[i], "backspin_ivb_cm": base_ivb[i],
                "recorded_height_cm": observed_height[i], "model_height_cm": height[i],
                "backspin_height_cm": base_z[i],
                "reference_xyz_m": reference[k].tolist(),
                "model_xyz_m": np.column_stack((trace[:, k, 0], y, trace[:, k, 1])).tolist(),
                "backspin_xyz_m": np.column_stack((base[:, k, 0], y, base[:, k, 1])).tolist(),
            })
    return report


def combine(*reports):
    summary = {"pairs": sum(row["pairs"] for row in reports),
               "unique_paired_pitches": sum(row["unique_paired_pitches"] for row in reports)}
    for method in reports[0]["results"]:
        rows = [report["results"][method] for report in reports]
        correct = sum(row["correct"] for row in rows)
        count = sum(row["non_tie_pairs"] for row in rows)
        metrics = {"correct": correct, "non_tie_pairs": count,
                   "accuracy_pct": 100 * correct / count if count else None}
        for key, weights in (("pair_delta_mae_cm", "pairs"), ("ivb_mae_cm", "unique_paired_pitches"),
                             ("height_mae_cm", "unique_paired_pitches")):
            if key in rows[0]:
                metrics[key] = sum(row[key] * report[weights] for row, report in zip(rows, reports)) / summary[weights]
        summary[method] = metrics
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("replication", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary = pool.submit(run, args.primary, examples=True)
        replication = pool.submit(run, args.replication)
        first, second = primary.result(), replication.result()
    report = {"primary": first, "replication": second, "combined": combine(first, second)}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
