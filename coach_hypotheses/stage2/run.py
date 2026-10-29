"""Stage2 artifacts; scenario outputs precede the separate recorded comparison."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from .uncertainty import (PITCHES, prepare_case, speeds, grouped_mc, convergence,
                          recorded_comparison, file_hash, rigid_transform_velocity)

PROTOCOL_COMMIT = '45069ca'


def run(data, output):
    output.mkdir(parents=True, exist_ok=True)
    summaries, designs, sources, diagnostics = [], [], {}, []
    for pitch in PITCHES:
        case = prepare_case(data, pitch)
        sources.update(case['sources'])
        modes = {}
        for label, radius, motion in [('primary', 0., True), ('relative_offset_stress', .017, True), ('constant_distance_companion', 0., False)]:
            repeats = [grouped_mc(lambda u: speeds(case, u, radius, motion), n, seed)
                       for n in (8192, 32768) for seed in (20261029, 20261030)]
            designs.append({'pitch': pitch, 'mode': label, 'results': repeats, 'convergence': convergence(repeats)})
            modes[label] = repeats[2]  # Preselected N32768,seed20261029, never outcome-selected.
        nominal = float(speeds(case, np.full((1,7), .5))[0])
        summaries.append({'pitch': pitch, 'nominal_15cm_kmh': nominal, **modes})
        v = np.array([[1., 2., 3.], [-10., 20., 4.]])
        transformed = rigid_transform_velocity(v, case['rotation'][1, 1], np.array([.017, -.017, .017]))
        diagnostics.append({'pitch': pitch, 'legacy_15cm_error_kmh': case['legacy_regression_error_kmh'],
                            'fixed_global_transform_speed_difference_kmh': float(np.max(abs(np.linalg.norm(v, axis=1)-np.linalg.norm(transformed, axis=1)))*3.6),
                            'lengths_m': case['lengths_m'], 'br_s': case['br_s'], 'rate_hz': case['rate_hz']})
    # Freeze all outcome-free results before the only function that reads radar.
    payload = {'scope': 'engineering scenario, conditional kinematics, not empirical measurement intervals',
               'protocol_commit': PROTOCOL_COMMIT, 'summaries': summaries, 'diagnostics': diagnostics,
               'convergence': [{'pitch': item['pitch'], 'mode': item['mode'], 'verdict': item['convergence']['verdict']} for item in designs]}
    (output/'scenario_results.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (output/'mc_designs.json').write_text(json.dumps(designs, indent=2, allow_nan=False), encoding='utf-8')
    comparison = recorded_comparison(data, summaries)
    (output/'recorded_comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding='utf-8')
    package = Path(__file__).parent
    manifest = {'sources': sources,
                'protocol_sha256': file_hash(package/'STAGE2_PROTOCOL.md'),
                'code_sha256': {path.name: file_hash(path) for path in package.glob('*.py')},
                'reused_source_code_sha256': {name: file_hash(package.parents[1]/name) for name in ('ball_release.py', 'pilot.py', 'speed_path.py', 'release_vector_audit.py')},
                'head_at_execution': subprocess.check_output(['git','rev-parse','HEAD'],cwd=package,text=True).strip()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must remain in coach_hypotheses/stage2')
    payload = run(args.data, args.output)
    print(json.dumps({'pitches': len(payload['summaries']), 'convergence': payload['convergence']}, indent=2))


if __name__ == '__main__':
    main()
