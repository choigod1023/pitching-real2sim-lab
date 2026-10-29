"""Freeze outcome-free scenario summaries before separate recorded comparison."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from ..stage2.uncertainty import PITCHES, prepare_case, file_hash, recorded_comparison
from .targets import (N, SEEDS, conditions, common_samples, evaluate, summarize,
                      numerical_difference, width_shapley, local_distance_sensitivity, scenario_goal)

PACKAGE = Path(__file__).parent
ROOT = PACKAGE.parents[1]
PROTOCOL_COMMIT = 'ba5c9c816971d3d83f8597777860ed6ba53b7a6d'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def tracked_hashes():
    names = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    return {name: file_hash(ROOT/name) for name in names
            if not name.startswith('coach_hypotheses/measurement_targets/')}


def run(data, output):
    output = Path(output).resolve()
    if not output.is_relative_to(PACKAGE.resolve()):
        raise ValueError('output_must_remain_in_measurement_targets')
    output.mkdir(parents=True, exist_ok=True)
    before = tracked_hashes()
    rows, sensitivities, shapley, sources, processing = [], [], [], {}, []
    designs = {seed: common_samples(N, seed) for seed in SEEDS}
    for pitch in PITCHES:
        case = prepare_case(data, pitch)
        sources.update(case['sources'])
        sensitivity = local_distance_sensitivity(case)
        sensitivities.append({'pitch': pitch, **sensitivity})
        nominal = sensitivity['nominal_kmh']
        # Hold distance AND relative motion exactly at nominal to expose residual processing differences.
        u = np.full((9, 7), .5)
        u[:, 2] = np.repeat([1/6, .5, 5/6], 3)
        u[:, 3] = np.tile([1/6, .5, 5/6], 3)
        processing.append({'pitch': pitch, 'nominal_distance_release_filter_grid_kmh':
                           evaluate(case, u, ('distance', 'relative_motion'), 0.).reshape(3,3).tolist()})
        for seed, design in designs.items():
            samples = design.reshape(-1, 7)
            low_indices = np.r_[np.arange(32768), N+np.arange(32768)]
            widths = {}
            for condition in conditions():
                values = evaluate(case, samples, condition['fixed'], condition['distance_halfwidth_mm'])
                summary, low = summarize(values), summarize(values[low_indices])
                row = {'pitch': pitch, 'condition': condition['name'], 'seed': seed,
                       'n_per_matrix': N, 'nominal_kmh': nominal, 'summary': summary,
                       'mean_minus_nominal_kmh': summary['mean_kmh']-nominal,
                       'n32768_summary': low, 'sample_count_difference': numerical_difference(summary, low)}
                rows.append(row)
                if condition['name'].startswith('fixed_'):
                    widths[int(condition['name'][6:])] = summary['width_kmh']
            shapley.append({'pitch': pitch, 'seed': seed, 'width_reduction_kmh': width_shapley(widths)})
    for row in rows:
        baseline = next(r for r in rows if r['pitch'] == row['pitch'] and r['seed'] == row['seed'] and r['condition'] == 'fixed_00')
        row['width_reduction_kmh'] = baseline['summary']['width_kmh']-row['summary']['width_kmh']
        row['width_reduction_fraction'] = row['width_reduction_kmh']/baseline['summary']['width_kmh']
    seed_differences = []
    for row in rows:
        if row['seed'] == SEEDS[0]:
            other = next(r for r in rows if r['pitch'] == row['pitch'] and r['condition'] == row['condition'] and r['seed'] == SEEDS[1])
            seed_differences.append({'pitch': row['pitch'], 'condition': row['condition'],
                                     **numerical_difference(row['summary'], other['summary'])})
    goals = {condition['name']: scenario_goal([r for r in rows if r['condition'] == condition['name']]) for condition in conditions()}
    max_slope = max(abs(s['slope_kmh_per_mm']) for row in sensitivities for s in row['finite_differences'])
    payload = {'protocol_commit': PROTOCOL_COMMIT, 'conditions': conditions(),
               'scope': 'conditional engineering scenarios; no empirical accuracy or improvement claim',
               'n_per_matrix': N, 'seeds': SEEDS, 'summaries': rows, 'sensitivities': sensitivities,
               'local_conservative_distance_mm_for_1kmh_across_five': 1./max_slope,
               'width_shapley': shapley, 'seed_differences': seed_differences, 'scenario_goals': goals,
               'fixed_distance_processing_grid': processing}
    write_json(output/'scenario_results.json', payload)
    with (output/'summaries.csv').open('w', newline='', encoding='utf-8') as handle:
        flat = [{k: row[k] for k in ('pitch','condition','seed','nominal_kmh','mean_minus_nominal_kmh','width_reduction_kmh','width_reduction_fraction')} | row['summary'] for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    frozen = {name: file_hash(output/name) for name in ('scenario_results.json', 'summaries.csv')}
    comparison_rows = [{'pitch': pitch, 'nominal_15cm_kmh': next(r['nominal_kmh'] for r in rows if r['pitch'] == pitch),
                        'primary': {'mean_kmh': next(r['summary']['mean_kmh'] for r in rows if r['pitch'] == pitch and r['condition'] == 'fixed_00' and r['seed'] == SEEDS[0])}} for pitch in PITCHES]
    write_json(output/'recorded_comparison.json', recorded_comparison(data, comparison_rows))
    after = tracked_hashes()
    if before != after or any(file_hash(path) != digest for path, digest in sources.items()):
        raise AssertionError('existing_files_or_sources_modified')
    if any(file_hash(output/name) != digest for name, digest in frozen.items()):
        raise AssertionError('scenario_changed_after_recorded_read')
    write_json(output/'manifest.json', {'protocol_commit': PROTOCOL_COMMIT,
               'protocol_sha256': file_hash(PACKAGE/'PROTOCOL.md'), 'sources_sha256': sources,
               'existing_tracked_sha256': before, 'existing_files_preserved': True,
               'scenario_frozen_before_recorded_read': frozen,
               'code_sha256': {p.name: file_hash(p) for p in PACKAGE.glob('*.py')},
               'head_at_execution': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PACKAGE/'results')
    args = parser.parse_args()
    payload = run(args.data, args.output)
    print(json.dumps({'rows': len(payload['summaries']), 'conditions': len(payload['conditions']),
                      'local_distance_mm_for_1kmh': payload['local_conservative_distance_mm_for_1kmh_across_five']}, indent=2))


if __name__ == '__main__':
    main()
