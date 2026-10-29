"""Disjoint local derivative support for existing hand/Ball surface-marker pilot."""
import argparse
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from ..research_reset.validation import file_digest

PILOT_SHA256 = '7c7c6b1183c5c9a38ddd16f6dac598e15fec50030d7232a08a7ffe5917747acc'


def load_pilot(path):
    if file_digest(path) != PILOT_SHA256:
        raise ValueError('changed_pilot_code')
    module = ModuleType('frozen_marker_pilot')
    module.__file__ = str(path)
    # Execute the verified source itself, without writing caches into another worktree.
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def guard_indices(time):
    time = np.asarray(time)
    train = np.flatnonzero((time >= -.09-1e-6) & (time <= -.045+1e-6))
    old = np.flatnonzero((time >= -.04-1e-6) & (time <= -.005+1e-6))
    test = np.flatnonzero((time >= -.035-1e-6) & (time <= -.005+1e-6))
    if (len(train), len(old), len(test)) != (10, 8, 7) or test[-1]+1 >= len(time) or test[0] < 1:
        raise ValueError('unexpected_guard_window')
    support = np.unique(np.r_[test-1, test, test+1])
    if np.intersect1d(train, support).size:
        raise ValueError('overlapping_derivative_support')
    return train, test, support, dict(old_train_derivative_shared_samples=int(np.intersect1d(train, np.r_[old-1, old+1]).size),
                                      strict_train_derivative_shared_samples=0,
                                      train_samples=len(train), test_samples=len(test))


def guarded_case(module, source, calibration, side, shift):
    events = [event for event in source['events'] if event['label'] == 'Event']
    if len(events) != 1:
        raise ValueError('ambiguous_event')
    time = source['time']-events[0]['time']-shift/source['rate']
    train, test, support, checks = guard_indices(time)
    wrist, rotations, valid, _ = module.dynamic_hand(source, calibration, side)
    ball = source['markers']['Ball']
    all_required = np.unique(np.r_[train, support])
    if not (valid[all_required].all() and source['valid']['Ball'][all_required].all()):
        raise ValueError('missing_guard_observations')
    truth_velocity = (ball[test+1]-ball[test-1])*source['rate']/2
    rows = []
    for linear, name in [(False, 'rigid_offset'), (True, 'linear_relative_offset')]:
        fitted = module.fit_offset(time, wrist, rotations, ball, train, linear)
        predicted = module.predict_marker(time, wrist, rotations, fitted)
        velocity = (predicted[test+1]-predicted[test-1])*source['rate']/2
        rows.append(dict(model=name, event_shift_frames=shift, **checks,
                         marker_position_rmse_mm=float(np.sqrt(np.mean(np.sum((predicted[test]-ball[test])**2, axis=1)))*1000),
                         marker_speed_mae_kmh=float(np.mean(abs(np.linalg.norm(velocity, axis=1)-np.linalg.norm(truth_velocity, axis=1)))*3.6),
                         marker_vector_rmse_kmh=float(np.sqrt(np.mean(np.sum((velocity-truth_velocity)**2, axis=1)))*3.6)))
    return rows


def guard_verdict(rows):
    differences = []
    for trial in sorted({row['trial'] for row in rows}):
        selected = {r['model']: r for r in rows if r['trial'] == trial and r['event_shift_frames'] == 0}
        if set(selected) != {'rigid_offset', 'linear_relative_offset'}:
            raise ValueError('incomplete_nominal_guard_case')
        a, b = selected['rigid_offset'], selected['linear_relative_offset']
        differences.append(dict(trial=trial,
                                position_difference_mm=b['marker_position_rmse_mm']-a['marker_position_rmse_mm'],
                                speed_difference_kmh=b['marker_speed_mae_kmh']-a['marker_speed_mae_kmh']))
    if not differences:
        raise ValueError('empty_guard_results')
    position = [r['position_difference_mm'] for r in differences]
    speed = [r['speed_difference_kmh'] for r in differences]
    return dict(pilot_joint_improvement=bool(max(position) <= 0 and max(speed) <= 0 and
                                           np.mean(position) < 0 and np.mean(speed) < 0),
                nominal_differences=differences, center_speed_validated=False,
                source_filter_support_verified=False, coaching_recommendation_ready=False)


def run_guard(data, pilot_path, pilot_summary, output):
    module = load_pilot(pilot_path)
    data, output = Path(data).resolve(), Path(output)
    frozen = json.loads(Path(pilot_summary).read_text(encoding='utf-8'))['inputs_sha256']
    hashes = {}
    for dynamic, static, _ in module.CASES:
        for name in (dynamic, static):
            path = (data/name).resolve()
            if not path.is_relative_to(data):
                raise ValueError('guard_source_outside_data')
            actual = file_digest(path)
            if actual != frozen[name]:
                raise ValueError('changed_guard_source:' + name)
            hashes[name] = actual
    rows = []
    for dynamic, static, side in module.CASES:
        source = module.load_c3d(data/dynamic)
        calibration = module.calibrate_wrist(module.load_c3d(data/static), side)
        for shift in (0, -1):
            rows.extend(dict(row, trial=dynamic) for row in guarded_case(module, source, calibration, side, shift))
    result = dict(pilot_code_sha256=PILOT_SHA256, inputs_sha256=hashes,
                  rows=rows, verdict=guard_verdict(rows),
                  purpose='same_trial_surface_marker_reconstruction_not_ball_center_or_intervention')
    output.mkdir(parents=True, exist_ok=True)
    (output/'guard_results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['verdict'], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--pilot-module', required=True, type=Path)
    parser.add_argument('--pilot-summary', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('artifacts')/'guard')
    args = parser.parse_args()
    run_guard(args.data, args.pilot_module, args.pilot_summary, args.output)


if __name__ == '__main__':
    main()
