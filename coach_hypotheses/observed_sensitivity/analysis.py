"""Error margins and event-window stress on frozen observed pitching pairs."""
import argparse
import itertools
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from ..observed_tradeoffs.analysis import file_digest, CUTOFFS, EPSILON

PROTOCOL_COMMIT = 'd893612496449b9b77c6404394487d51e2a28c44'
EXTRACTOR_SHA256 = '21b2921dc381fd6ac6d52d01052c1dcb2d4da8f50a275bd93589e4baafb01cec'
PRIOR_HASHES = {
    'summary.json': '2778ee589658ed8165cd17a0b0e74916a906cef9b2af0ae62345e9b15ce947d2',
    'all_pair_stability.csv': '7384a36460c80fd0463e6211ab5ce8c0f63646d5144c78c2a9e41b3860008685',
    'all_pair_filter_comparisons.csv': 'be5168f39202f85434059d760cb68edb589bece901b93b23bd475a5454a05567',
}
OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=2))
NOMINAL = OFFSETS.index((0, 0))
PROFILES = ((0., 0., 0.), (.25, .5, 5.), (.5, 1., 10.), (1., 2., 20.), (2., 5., 50.))
RATE = 360.


def load_prior(directory):
    """Only the previously frozen 59-pair set may enter this sensitivity test."""
    directory = Path(directory)
    for name, expected in PRIOR_HASHES.items():
        if file_digest(directory / name) != expected:
            raise ValueError('changed_prior:' + name)
    pairs = pd.read_csv(directory / 'all_pair_stability.csv', dtype={'user': str})
    nominal = pd.read_csv(directory / 'all_pair_filter_comparisons.csv', dtype={'user': str})
    selected = pairs[pairs.filter_stable_observed_nonincrease].copy()
    if len(selected) != 59 or selected.recorded_speed_gap_kmh.le(0).any():
        raise ValueError('changed_selected_cohort')
    identities = set(zip(selected.user, selected.pitch_i, selected.pitch_j))
    nominal = nominal[[tuple(row) in identities for row in nominal[['user', 'pitch_i', 'pitch_j']].to_numpy()]]
    if len(nominal) != 177 or not nominal.status.eq('observed_nonincrease').all():
        raise ValueError('inconsistent_prior_pair_rows')
    return selected, nominal


def load_sources(data_dir, extractor_path, pitches):
    """Load original finite support through the exact upstream extraction code."""
    data_dir, extractor_path = Path(data_dir), Path(extractor_path)
    if file_digest(extractor_path) != EXTRACTOR_SHA256:
        raise ValueError('changed_upstream_extractor')
    module = ModuleType('frozen_tradeoff_metrics')
    module.__file__ = str(extractor_path)
    exec(compile(extractor_path.read_bytes(), str(extractor_path), 'exec'), module.__dict__)
    upstream = json.loads((extractor_path.parent / 'results/manifest.json').read_text(encoding='utf-8'))
    frames, hashes = {}, {}
    for name, columns in module.COLUMNS.items():
        relative = 'full_sig/' + name + '.csv'
        source = data_dir / relative
        digest = file_digest(source)
        if digest != upstream['sources'][relative]['sha256']:
            raise ValueError('changed_full_signal:' + name)
        hashes[relative] = digest
        usecols = ['session_pitch', 'time', *module.EVENTS, *columns]
        table = pd.read_csv(source, usecols=usecols, dtype={'session_pitch': str})
        table = table[table.session_pitch.isin(pitches)]
        frames[name] = dict(tuple(table.groupby('session_pitch', sort=True)))
        if set(frames[name]) != set(pitches):
            raise ValueError('missing_full_signal_pitch:' + name)
    prepared = {pitch: module.prepare_pitch({name: groups[pitch].sort_values('time')
                                           for name, groups in frames.items()}) for pitch in sorted(pitches)}
    return module, prepared, hashes


def survives(high, low, gap, profile, minimum_gap=0.):
    """Worst signed extrema errors; last axis is varus, resultant, signed tension."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    profile = np.asarray(profile, float)
    if (high.shape[-1:] != (3,) or low.shape[-1:] != (3,) or profile.shape != (3,) or
            not all(np.isfinite(x).all() for x in (high, low, profile)) or
            not np.isfinite([gap, minimum_gap]).all() or (profile < 0).any() or minimum_gap < 0):
        raise ValueError('invalid_stress_inputs')
    speed_error, torque_error, force_error = profile
    remaining = gap - 2 * speed_error
    speed_ok = remaining > 0 if minimum_gap == 0 else remaining >= minimum_gap
    return (speed_ok & (high[..., 0] + 2 * torque_error <= low[..., 0] + EPSILON) &
            (high[..., 1] + 2 * force_error <= low[..., 1] + EPSILON) &
            (np.maximum(0., high[..., 2] + force_error) <=
             np.maximum(0., low[..., 2] - force_error) + EPSILON))


def error_margins(high, low, gap):
    """Per-reading bounds, not estimates of actual instrument/model error."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    if high.shape != (3,) or low.shape != (3,) or not survives(high, low, gap, PROFILES[0]):
        raise ValueError('nominal_pair_not_nonincrease')
    tension = max(0., -high[2], (low[2] - high[2]) / 2)
    resultant = (low[1] - high[1]) / 2
    return dict(speed_order_error_limit_kmh=float(gap / 2),
                gap_at_least_1_error_limit_kmh=float((gap - 1.) / 2),
                elbow_peak_error_limit_nm=float((low[0] - high[0]) / 2),
                shoulder_resultant_error_limit_n=float(resultant),
                signed_tension_error_limit_n=float(tension),
                joint_shoulder_error_limit_n=float(min(resultant, tension)))


def knee_direction_audit(time, angles, velocity, fp, br):
    """Compare provided extension velocity to minus the flexion-angle derivative."""
    time, angles, velocity = map(lambda x: np.asarray(x, float), (time, angles, velocity))
    if (time.ndim != 1 or len(time) < 5 or angles.shape != time.shape or velocity.shape != time.shape or
            not np.isfinite(np.r_[time, angles, velocity, fp, br]).all() or np.any(np.diff(time) <= 0)):
        raise ValueError('invalid_knee_audit_support')
    mask = (time >= fp) & (time <= br)
    mask[[0, -1]] = False
    if mask.sum() < 3:
        raise ValueError('insufficient_knee_audit_samples')
    derivative = -np.gradient(angles, time)[mask]
    observed = velocity[mask]
    correlation = None if np.std(derivative) == 0 or np.std(observed) == 0 else float(np.corrcoef(derivative, observed)[0, 1])
    return dict(samples=int(mask.sum()), extension_derivative_correlation=correlation,
                same_direction_mae_dps=float(np.mean(np.abs(derivative - observed))),
                opposite_direction_mae_dps=float(np.mean(np.abs(derivative + observed))))


def event_states(module, prepared, cutoff, expected):
    """Keep filtered curves fixed; only PKH/BR selection boundaries change."""
    time, matrix, columns, events, _ = prepared
    signal = module.filtered_signals(matrix, columns, cutoff)
    fp = events['fp_10_time']
    resultant = np.linalg.norm(np.column_stack([signal['shoulder_upper_arm_force_' + axis] for axis in 'xyz']), axis=1)
    states = []
    for pkh_offset, br_offset in OFFSETS:
        pkh = events['pkh_time'] + pkh_offset / RATE
        br = events['BR_time'] + br_offset / RATE
        if not time[0] <= pkh < fp < br <= time[-1]:
            raise ValueError('shifted_event_outside_support_or_order')
        varus = module.window_peak(time, signal['elbow_moment_y'], pkh, br)[1]
        shoulder = module.window_peak(time, resultant, pkh, br)[1]
        tension = module.window_peak(time, -signal['shoulder_upper_arm_force_z'], pkh, br)[1]
        knee = module.at_event(time, signal['lead_knee_angle_x'], fp) - module.at_event(time, signal['lead_knee_angle_x'], br)
        states.append([varus, shoulder, tension, knee])
    states = np.asarray(states)
    if not np.isfinite(states).all():
        raise ValueError('nonfinite_event_states')
    reference = np.array([expected[name] for name in ('elbow_varus_peak_nm', 'shoulder_resultant_peak_n',
                                                    'shoulder_distraction_peak_n', 'lead_knee_extension_deg')], float)
    difference = float(np.max(np.abs(states[NOMINAL] - reference)))
    if not np.allclose(states[NOMINAL], reference, atol=1e-7, rtol=0):
        raise ValueError('nominal_reproduction_failed')
    audit = knee_direction_audit(time, signal['lead_knee_angle_x'], signal['lead_knee_velo_x'], fp, events['BR_time'])
    return states, audit, difference


def inspect_pairs(pairs, nominal, states):
    """Evaluate all 3*81 cases per pair and all fixed stress profiles."""
    records, events, margins = [], [], []
    for pair in sorted(pairs.to_dict('records'), key=lambda r: (int(r['user']), r['pitch_i'], r['pitch_j'])):
        rows = nominal[(nominal.user == pair['user']) & (nominal.pitch_i == pair['pitch_i']) & (nominal.pitch_j == pair['pitch_j'])]
        source = rows[rows.cutoff_hz == 20].iloc[0]
        faster, slower, gap = source.faster_pitch, source.slower_pitch, float(source.recorded_speed_gap_kmh)
        high = np.stack([states[(faster, c)][:, :3] for c in CUTOFFS])
        low = np.stack([states[(slower, c)][:, :3] for c in CUTOFFS])
        high_grid, low_grid = high[:, :, None, :], low[:, None, :, :]
        valid = survives(high_grid, low_grid, gap, PROFILES[0])
        raw = high_grid - low_grid
        corrected = raw.copy()
        corrected[..., 2] = np.maximum(0., high_grid[..., 2]) - np.maximum(0., low_grid[..., 2])
        item = dict(user=pair['user'], pitch_i=pair['pitch_i'], pitch_j=pair['pitch_j'],
                    faster_pitch=faster, slower_pitch=slower, recorded_speed_gap_kmh=gap,
                    nominal_still_valid=bool(valid[:, NOMINAL, NOMINAL].all()),
                    common_event_stable=bool(np.diagonal(valid, axis1=1, axis2=2).all()),
                    independent_event_stable=bool(valid.all()),
                    worst_elbow_delta_nm=float(corrected[..., 0].max()),
                    worst_shoulder_resultant_delta_n=float(corrected[..., 1].max()),
                    worst_tension_magnitude_delta_n=float(corrected[..., 2].max()),
                    raw_signed_independent_event_stable=bool((raw <= EPSILON).all()))
        if not item['nominal_still_valid']:
            raise ValueError('prior_nominal_selection_not_reproduced')
        per_filter = []
        for index, cutoff in enumerate(CUTOFFS):
            row = dict(user=pair['user'], pitch_i=pair['pitch_i'], pitch_j=pair['pitch_j'], cutoff_hz=cutoff,
                       **error_margins(high[index, NOMINAL], low[index, NOMINAL], gap))
            per_filter.append(row)
            margins.append(row)
            for f_index, (fp_offset, fb_offset) in enumerate(OFFSETS):
                for s_index, (sp_offset, sb_offset) in enumerate(OFFSETS):
                    events.append(dict(user=pair['user'], pitch_i=pair['pitch_i'], pitch_j=pair['pitch_j'], cutoff_hz=cutoff,
                                       faster_pkh_shift_frames=fp_offset, faster_br_shift_frames=fb_offset,
                                       slower_pkh_shift_frames=sp_offset, slower_br_shift_frames=sb_offset,
                                       load_nonincrease=bool(valid[index, f_index, s_index]),
                                       elbow_delta_nm=float(corrected[index, f_index, s_index, 0]),
                                       shoulder_resultant_delta_n=float(corrected[index, f_index, s_index, 1]),
                                       tension_magnitude_delta_n=float(corrected[index, f_index, s_index, 2]),
                                       knee_indicator_delta_deg=float(states[(faster, cutoff)][f_index, 3] - states[(slower, cutoff)][s_index, 3])))
        item['minimum_margins_across_filters'] = {key: min(r[key] for r in per_filter) for key in per_filter[0] if key.endswith(('_kmh', '_nm', '_n'))}
        item['profiles'] = []
        for profile in PROFILES:
            stressed = survives(high_grid, low_grid, gap, profile)
            item['profiles'].append(dict(per_reading_bounds=list(profile),
                                         nominal=bool(stressed[:, NOMINAL, NOMINAL].all()),
                                         common_event=bool(np.diagonal(stressed, axis1=1, axis2=2).all()),
                                         independent_event=bool(stressed.all()),
                                         independent_event_gap_at_least_1=bool(survives(high_grid, low_grid, gap, profile, 1.).all())))
        records.append(item)
    return records, pd.DataFrame(events), pd.DataFrame(margins)


def summarize(records, audits, maximum_difference):
    if not records:
        raise ValueError('empty_sensitivity_records')
    profiles = []
    for index, profile in enumerate(PROFILES):
        row = dict(per_reading_bounds=list(profile))
        for key in ('nominal', 'common_event', 'independent_event', 'independent_event_gap_at_least_1'):
            row[key + '_pairs'] = sum(r['profiles'][index][key] for r in records)
        row['original_gap_at_least_1_and_independent_event_pairs'] = sum(
            r['recorded_speed_gap_kmh'] >= 1. and r['profiles'][index]['independent_event'] for r in records)
        profiles.append(row)
    correlations = [a['extension_derivative_correlation'] for a in audits if a['extension_derivative_correlation'] is not None]
    limits = {key: [float(x) for x in np.quantile([r['minimum_margins_across_filters'][key] for r in records], [0, .5, 1])]
              for key in records[0]['minimum_margins_across_filters']}
    return dict(selected_pairs=len(records), selected_players=len({r['user'] for r in records}),
                event_filter_pair_rows=len(records) * 3 * 81,
                nominal_reproduction_max_absolute_difference=maximum_difference,
                common_event_stable_pairs=sum(r['common_event_stable'] for r in records),
                independent_event_stable_pairs=sum(r['independent_event_stable'] for r in records),
                independent_event_stable_gap_at_least_1_pairs=sum(r['independent_event_stable'] and r['recorded_speed_gap_kmh'] >= 1. for r in records),
                raw_signed_independent_event_stable_pairs=sum(r['raw_signed_independent_event_stable'] for r in records),
                per_reading_margin_min_median_max=limits, engineering_profiles=profiles,
                knee_audit=dict(pitch_filter_cases=len(audits), finite_correlations=len(correlations),
                                extension_correlation_min_median_max=None if not correlations else [float(x) for x in np.quantile(correlations, [0, .5, 1])],
                                same_direction_mae_median_dps=float(np.median([a['same_direction_mae_dps'] for a in audits])),
                                opposite_direction_mae_median_dps=float(np.median([a['opposite_direction_mae_dps'] for a in audits]))),
                pair_records=records, measurement_robustness_validated=False,
                same_capacity_validated=False, intervention_effect_validated=False,
                coaching_recommendation_ready=False, speed_prediction_gate_changed=False)


def run(prior_results, metrics_dir, data_dir, output):
    pairs, nominal = load_prior(prior_results)
    pitches = set(pairs.pitch_i) | set(pairs.pitch_j)
    module, prepared, source_hashes = load_sources(data_dir, Path(metrics_dir) / 'metrics.py', pitches)
    metrics_file = Path(metrics_dir) / 'results/pitch_metrics.csv'
    if file_digest(metrics_file) != 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4':
        raise ValueError('changed_reference_metrics')
    reference = pd.read_csv(metrics_file).set_index(['session_pitch', 'cutoff_hz'])
    states, audits, differences = {}, [], []
    for pitch in sorted(pitches):
        for cutoff in CUTOFFS:
            states[(pitch, cutoff)], audit, difference = event_states(module, prepared[pitch], cutoff, reference.loc[(pitch, cutoff)])
            audits.append(dict(pitch=pitch, cutoff_hz=cutoff, **audit))
            differences.append(difference)
    records, event_rows, margins = inspect_pairs(pairs, nominal, states)
    result = summarize(records, audits, max(differences))
    if len(event_rows) != 14337:
        raise ValueError('unexpected_event_case_count')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    event_rows.to_csv(output / 'all_event_comparisons.csv', index=False)
    margins.to_csv(output / 'all_nominal_margins.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'knee_direction_audit.csv', index=False)
    manifest = dict(protocol_commit=PROTOCOL_COMMIT, protocol_sha256=file_digest(Path(__file__).with_name('PROTOCOL.md')),
                    code_sha256=file_digest(__file__), prior_result_hashes=PRIOR_HASHES,
                    extractor_sha256=EXTRACTOR_SHA256, source_hashes=source_hashes,
                    reference_metrics_sha256=file_digest(metrics_file), selected_pitch_count=len(pitches),
                    source_path=str(Path(data_dir).resolve()), prior_results_path=str(Path(prior_results).resolve()),
                    data_license='CC BY-NC-SA 4.0', profiles_are_measured_error_bounds=False,
                    source_url='https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching')
    for name, payload in [('summary.json', result), ('manifest.json', manifest)]:
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('selected_pairs', 'common_event_stable_pairs',
                                           'independent_event_stable_pairs', 'independent_event_stable_gap_at_least_1_pairs')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-results', type=Path, required=True)
    parser.add_argument('--metrics-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must remain in observed_sensitivity package')
    run(args.prior_results, args.metrics_dir, args.data_dir, args.output)


if __name__ == '__main__':
    main()
