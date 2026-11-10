"""Compare a pitcher-season optical efficiency prior on the existing Statcast pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.optical_axis_flight.study import (
    column, combine, compare, initial_state, read_cohort, reference_trajectory, simulate,
)
from experiments.trackman_v3_flight.study import FT


def read_efficiency(source, year, min_pitches=30):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    valid = {}
    for row in records:
        if int(row["year"]) != year or row["api_pitch_type"] != "FF":
            raise ValueError("Expected a single season of four-seam leaderboard data")
        count = int(row["n_pitches"])
        if count < min_pitches or not row["active_spin"].strip():
            continue
        rate = float(row["active_spin"])
        pitcher = int(row["player_id"])
        if not 0 < rate <= 1 or pitcher in valid:
            raise ValueError("Invalid or duplicate pitcher efficiency")
        valid[pitcher] = (rate, count)
    return valid, len(records)


def match_efficiency(rows, records):
    pitcher_ids = [row["key"][1] for row in rows]
    rates = np.array([records.get(pid, (1., 0))[0] for pid in pitcher_ids])
    known = np.array([pid in records for pid in pitcher_ids])
    return rates, known


def evaluate(rows, pairs, simulations, rates):
    if not len(pairs):
        raise ValueError("Empty set of comparable pairs")
    unique = np.unique(pairs)
    ivb = 100 * FT * column(rows, "pfx_z")
    height = 100 * FT * column(rows, "plate_z")
    rpm = column(rows, "release_spin_rate")
    results = {"rpm_only": compare(rpm, ivb, pairs, physical_units=False),
               "active_rpm_only": compare(rpm * rates, ivb, pairs, physical_units=False)}
    for name, (pred_ivb, pred_height, _, _, ground) in simulations.items():
        results[name] = compare(pred_ivb, ivb, pairs) | {
            "ivb_mae_cm": float(np.mean(np.abs(pred_ivb[unique] - ivb[unique]))),
            "height_mae_cm": float(np.mean(np.abs(pred_height[unique] - height[unique]))),
            "below_ground_pitches": int(ground[unique].sum()),
        }
    return {"pairs": len(pairs), "unique_paired_pitches": len(unique),
            "pitchers": len({rows[i]["key"][1] for i in unique}), "results": results}


def run(source, prior, current, show_example=False):
    rows, pairs, _, _ = read_cohort(source)
    prior_rates, prior_known = match_efficiency(rows, prior)
    current_rates, current_known = match_efficiency(rows, current)
    known_pairs = pairs[prior_known[pairs].all(axis=1)]
    current_pairs = pairs[current_known[pairs].all(axis=1)]
    keep = next(((i, j) for i, j in pairs if j == i + 1), ()) if show_example else ()
    launch = initial_state(rows, 1.)[:3]
    settings = {
        "optical_100pct": (True, 1.),
        "optical_2024_mean": (True, prior_rates),
        "backspin_2024_mean": (False, prior_rates),
        "optical_2025_mean": (True, current_rates),
    }
    with ThreadPoolExecutor(max_workers=len(settings)) as pool:
        jobs = {
            key: pool.submit(simulate, rows, use_axis, active, fixed_launch=launch,
                             keep=keep if key in ("optical_100pct", "optical_2024_mean") else ())
            for key, (use_axis, active) in settings.items()
        }
        simulations = {key: job.result() for key, job in jobs.items()}
    output = {
        "date": rows[0]["date"], "pitch_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "all_pairs": evaluate(rows, pairs, simulations, prior_rates),
        "known_2024_pairs": evaluate(rows, known_pairs, simulations, prior_rates),
        "known_2025_pair_count": len(current_pairs),
        "pairs_with_100pct_fallback_2024": len(pairs) - len(known_pairs),
        "pairs_with_100pct_fallback_2025": len(pairs) - len(current_pairs),
    }
    if keep:
        pitcher = rows[keep[0]]["key"][1]
        rate, count = prior[pitcher]
        paths, _ = reference_trajectory([rows[i] for i in keep])
        nominal_ivb, nominal_height, _, nominal, _ = simulations["optical_100pct"]
        adjusted_ivb, adjusted_height, distance, adjusted, _ = simulations["optical_2024_mean"]
        output["overlay"] = {
            "title": "투수별 유효 회전율을 넣은 포심 궤적",
            "subtitle": f"{rows[keep[0]]['name']}  ·  {rows[keep[0]]['date']}",
            "reference_label": "Statcast 적합 궤적", "model_label": f"2024 평균 {rate:.2%}",
            "comparison_label": "유효 회전율 100%", "comparison_field": "comparison_xyz_m",
            "delta_label": "A→B",
            "note": f"두 공 모두 광학 2D 회전 방향을 사용했습니다. {rate:.2%}는 2024년 포심 {count}구의 평균입니다.",
        }
        output["examples"] = []
        for k, i in enumerate(keep):
            y = np.linspace(-distance[i], 0., len(adjusted))
            output["examples"].append({
                "pitch_number": rows[i]["key"][-1], "key": rows[i]["key"], "label": "AB"[k],
                "speed_kmh": float(rows[i]["speed_exact"]), "season_active_spin": rate, "season_pitch_count": count,
                "recorded_ivb_cm": 100 * FT * rows[i]["pfx_z"],
                "recorded_height_cm": 100 * FT * rows[i]["plate_z"],
                "nominal_ivb_cm": nominal_ivb[i], "nominal_height_cm": nominal_height[i],
                "model_ivb_cm": adjusted_ivb[i], "model_height_cm": adjusted_height[i],
                "reference_xyz_m": paths[k].tolist(),
                "model_xyz_m": np.column_stack((adjusted[:, k, 0], y, adjusted[:, k, 1])).tolist(),
                "comparison_xyz_m": np.column_stack((nominal[:, k, 0], y, nominal[:, k, 1])).tolist(),
            })
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("april", "september", "spin_2024", "spin_2025", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prior, prior_rows = read_efficiency(args.spin_2024, 2024)
    current, current_rows = read_efficiency(args.spin_2025, 2025)
    with ThreadPoolExecutor(max_workers=2) as pool:
        april = pool.submit(run, args.april, prior, current, show_example=True)
        september = pool.submit(run, args.september, prior, current)
        first, second = april.result(), september.result()
    report = {
        "leaderboards": {
            "2024": {"rows": prior_rows, "usable": len(prior),
                     "sha256": hashlib.sha256(args.spin_2024.read_bytes()).hexdigest()},
            "2025": {"rows": current_rows, "usable": len(current),
                     "sha256": hashlib.sha256(args.spin_2025.read_bytes()).hexdigest()},
        },
        "april": first, "september": second,
        "combined_all": combine(first["all_pairs"], second["all_pairs"]),
        "combined_known_2024": combine(first["known_2024_pairs"], second["known_2024_pairs"]),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
