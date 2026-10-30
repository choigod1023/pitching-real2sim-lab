"""Describe near-maximal peak sets instead of assuming precise event order."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..observed_sensitivity.analysis import load_sources, OFFSETS, NOMINAL
from ..observed_tradeoffs.analysis import file_digest, CUTOFFS

PROTOCOL_COMMIT = '9c128dbb7b639671acef599ee0e38091d401173d'
TOLERANCES = (.005, .01, .02)
SEGMENTS = ('pelvis', 'torso')
METRICS_HASH = 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4'
SUMMARY_HASH = '32ca2146b29a31ddf212988711705832e05366a0300300999f4535ed5fa0e490'


def order_state(lower, upper):
    """A one-sample deadband is engineering resolution, not measured uncertainty."""
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError('invalid_order_interval')
    if lower > 1:
        return 'torso_after'
    if upper < -1:
        return 'torso_before'
    if lower < -1 and upper > 1:
        return 'both_orders_possible'
    return 'one_sample_or_overlap'


def describe_peak(time, values, begin, end):
    time, values = np.asarray(time, float), np.asarray(values, float)
    if (time.ndim != 1 or time.shape != values.shape or len(time) < 2 or
            not np.isfinite(np.r_[time, values, begin, end]).all() or np.any(np.diff(time) <= 0) or
            not time[0] <= begin < end <= time[-1]):
        raise ValueError('invalid_peak_support')
    indices = np.flatnonzero((time >= begin) & (time <= end))
    if not len(indices):
        raise ValueError('empty_peak_window')
    selected = int(indices[np.argmax(values[indices])])
    height = float(values[selected])
    if height <= 0:
        raise ValueError('nonpositive_peak_for_relative_band')
    bands = {}
    for tolerance in TOLERANCES:
        near = indices[values[indices] >= height * (1 - tolerance)]
        components = np.split(near, np.flatnonzero(np.diff(near) > 1) + 1)
        component = next(group for group in components if selected in group)
        bands[str(tolerance)] = dict(first_index=int(near[0]), last_index=int(near[-1]), samples=len(near),
                                     width_ms=float((time[near[-1]] - time[near[0]]) * 1000),
                                     components=len(components),
                                     selected_component_width_ms=float((time[component[-1]] - time[component[0]]) * 1000))
    return dict(index=selected, time=float(time[selected]), height=height,
                on_window_boundary=bool(selected in (indices[0], indices[-1])), bands=bands)


def audit_pitch(module, prepared, reference):
    time, matrix, columns, events, _ = prepared
    curves = {cutoff: module.filtered_signals(matrix, columns, cutoff) for cutoff in CUTOFFS}
    descriptions, rows, max_error = {}, [], 0.
    for cutoff in CUTOFFS:
        for offset_index, (pk, br) in enumerate(OFFSETS):
            begin, end = events['pkh_time'] + pk / 360., events['BR_time'] + br / 360.
            if not begin < events['fp_10_time'] < end:
                raise ValueError('shifted_event_order')
            for segment in SEGMENTS:
                description = describe_peak(time, curves[cutoff][segment + '_velo_z'], begin, end)
                descriptions[(cutoff, offset_index, segment)] = description
                flat = dict(cutoff_hz=cutoff, pkh_shift_frames=pk, br_shift_frames=br, segment=segment,
                            index=description['index'], time=description['time'], height=description['height'],
                            on_window_boundary=description['on_window_boundary'])
                for tolerance, band in description['bands'].items():
                    flat.update({tolerance + '_' + key: value for key, value in band.items()})
                rows.append(flat)
    nominal_lags, steps = {}, {}
    for cutoff in CUTOFFS:
        p = descriptions[(cutoff, NOMINAL, 'pelvis')]
        t = descriptions[(cutoff, NOMINAL, 'torso')]
        expected = reference.loc[cutoff]
        differences = [abs((t['time'] - p['time']) * 1000 - expected.sequence_lag_ms)]
        for segment, description in (('pelvis', p), ('torso', t)):
            differences.extend([abs(description['height'] - expected[segment + '_peak_dps']),
                                abs((description['time'] - events['fp_10_time']) * 1000 - expected[segment + '_peak_time_fp10_ms'])])
        max_error = max(max_error, max(differences))
        nominal_lags[str(cutoff)] = float((t['time'] - p['time']) * 1000)
        steps[str(cutoff)] = t['index'] - p['index']
    if max_error > 1e-7:
        raise ValueError('nominal_timing_feature_reproduction_failed')
    ranges = {}
    for tolerance in TOLERANCES:
        lo, hi = [], []
        for cutoff in CUTOFFS:
            for index in range(9):
                p = descriptions[(cutoff, index, 'pelvis')]['bands'][str(tolerance)]
                t = descriptions[(cutoff, index, 'torso')]['bands'][str(tolerance)]
                lo.append(t['first_index'] - p['last_index'])
                hi.append(t['last_index'] - p['first_index'])
        ranges[str(tolerance)] = dict(minimum_lag_steps=min(lo), maximum_lag_steps=max(hi),
                                      minimum_lag_ms=min(lo) / 360 * 1000, maximum_lag_ms=max(hi) / 360 * 1000,
                                      classification=order_state(min(lo), max(hi)))
    movement = {}
    for segment in SEGMENTS:
        baseline = descriptions[(20, NOMINAL, segment)]
        alternatives = []
        for cutoff in (10, 30):
            alternative = descriptions[(cutoff, NOMINAL, segment)]
            ratio = float(curves[20][segment + '_velo_z'][alternative['index']] / baseline['height'])
            alternatives.append(dict(cutoff_hz=cutoff, index_shift=alternative['index'] - baseline['index'],
                                     shift_ms=(alternative['time'] - baseline['time']) * 1000,
                                     height_on_fixed20_relative_to_max=ratio,
                                     compatible_with_fixed20_one_percent_band=bool(ratio >= .99)))
        movement[segment] = alternatives
    return dict(nominal_lag_ms=nominal_lags, nominal_lag_steps=steps,
                nominal_order_across_filters=order_state(min(steps.values()), max(steps.values())),
                ranges=ranges, filter_peak_movements=movement,
                nominal_reproduction_error=max_error), rows


def audit_pairs(source, pitches):
    selected = [row for row in source['pair_records'] if row['independent_event_stable']]
    if len(selected) != 35 or len({r['user'] for r in selected}) != 23:
        raise ValueError('changed_fixed_pair_cohort')
    records = []
    for row in selected:
        high, low = pitches[row['faster_pitch']], pitches[row['slower_pitch']]
        if high['user'] != row['user'] or low['user'] != row['user']:
            raise ValueError('cross_player_peak_pair')
        difference = {str(c): high['nominal_lag_ms'][str(c)] - low['nominal_lag_ms'][str(c)] for c in CUTOFFS}
        ranges = {}
        for tolerance in TOLERANCES:
            h, l = high['ranges'][str(tolerance)], low['ranges'][str(tolerance)]
            minimum = h['minimum_lag_steps'] - l['maximum_lag_steps']
            maximum = h['maximum_lag_steps'] - l['minimum_lag_steps']
            classification = {'torso_after': 'delta_positive', 'torso_before': 'delta_negative',
                              'both_orders_possible': 'both_delta_signs_possible',
                              'one_sample_or_overlap': 'one_sample_or_overlap'}[order_state(minimum, maximum)]
            ranges[str(tolerance)] = dict(minimum_delta_steps=minimum, maximum_delta_steps=maximum,
                                         minimum_delta_ms=minimum / 360 * 1000, maximum_delta_ms=maximum / 360 * 1000,
                                         classification=classification)
        records.append(dict(user=row['user'], faster_pitch=row['faster_pitch'], slower_pitch=row['slower_pitch'],
                            nominal_delta_lag_ms=difference, ranges=ranges))
    players = []
    for user in sorted({r['user'] for r in records}, key=int):
        group = [r for r in records if r['user'] == user]
        means = {str(c): float(np.mean([r['nominal_delta_lag_ms'][str(c)] for r in group])) for c in CUTOFFS}
        players.append(dict(user=user, pairs=len(group), means=means,
                            mean_delta20_minus_delta10_ms=means['20'] - means['10']))
    largest = sorted(players, key=lambda r: (-abs(r['mean_delta20_minus_delta10_ms']), int(r['user'])))[:3]
    return dict(pairs=records, players=players,
                player_balanced_delta_ms={str(c): float(np.mean([p['means'][str(c)] for p in players])) for c in CUTOFFS},
                largest_filter_change_players=largest,
                classifications={str(r): {state: sum(p['ranges'][str(r)]['classification'] == state for p in records)
                                         for state in ('delta_positive', 'delta_negative', 'both_delta_signs_possible', 'one_sample_or_overlap')}
                                 for r in TOLERANCES})


def summarize(pitches, segment_rows, pair_result):
    classifications = {str(r): {state: sum(p['ranges'][str(r)]['classification'] == state for p in pitches.values())
                                for state in ('torso_after', 'torso_before', 'both_orders_possible', 'one_sample_or_overlap')}
                       for r in TOLERANCES}
    segments = {}
    for segment in SEGMENTS:
        rows = [r for r in segment_rows if r['segment'] == segment and r['pkh_shift_frames'] == 0 and r['br_shift_frames'] == 0]
        movements = [m for p in pitches.values() for m in p['filter_peak_movements'][segment] if m['index_shift'] != 0]
        widths = [r['0.01_width_ms'] for r in rows]
        segments[segment] = dict(nominal_pitch_filter_cases=len(rows),
                                 nominal_window_boundary_peaks=sum(r['on_window_boundary'] for r in rows),
                                 one_percent_disconnected_candidate_cases=sum(r['0.01_components'] > 1 for r in rows),
                                 one_percent_candidate_width_ms_min_median_p95_max=[float(x) for x in np.quantile(widths, [0, .5, .95, 1])],
                                 moved_alternative_filter_peaks=len(movements),
                                 moved_compatible_with_fixed20_one_percent_band=sum(r['compatible_with_fixed20_one_percent_band'] for r in movements),
                                 moved_outside_fixed20_one_percent_band=sum(not r['compatible_with_fixed20_one_percent_band'] for r in movements))
    return dict(pitches=len(pitches), players=len({p['user'] for p in pitches.values()}), nominal_peak_reproduction_max_error=max(p['nominal_reproduction_error'] for p in pitches.values()),
                segment_event_filter_rows=len(segment_rows), segments=segments,
                interval_order_classifications=classifications, selected_pair_audit=pair_result,
                per_pitch=pitches, calibrated_timing_accuracy_validated=False,
                timing_feature_ready_for_coaching=False, speed_prediction_gate_changed=False,
                new_speed_model_fitted=False, athlete_effect_validated=False,
                spin_or_movement_improvement_validated=False)


def run(data_dir, metrics_dir, source_summary, output):
    metrics_file = Path(metrics_dir) / 'results/pitch_metrics.csv'
    if file_digest(metrics_file) != METRICS_HASH or file_digest(source_summary) != SUMMARY_HASH:
        raise ValueError('changed_frozen_inputs')
    frame = pd.read_csv(metrics_file, dtype={'user': str})
    pitch_ids = set(frame.session_pitch)
    if len(pitch_ids) != 399 or frame.user.nunique() != 98:
        raise ValueError('changed_metric_cohort')
    module, prepared, source_hashes = load_sources(data_dir, Path(metrics_dir) / 'metrics.py', pitch_ids)
    audits, rows = {}, []
    reference = frame.set_index(['session_pitch', 'cutoff_hz'])
    for pitch in sorted(pitch_ids):
        audits[pitch], details = audit_pitch(module, prepared[pitch], reference.loc[pitch])
        audits[pitch]['user'] = str(reference.loc[(pitch, 20), 'user'])
        rows.extend(dict(pitch=pitch, **row) for row in details)
    pair_result = audit_pairs(json.loads(Path(source_summary).read_text()), audits)
    result = summarize(audits, rows, pair_result)
    if len(rows) != 21546:
        raise ValueError('unexpected_peak_case_count')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'all_peak_candidates.csv', index=False)
    manifest = dict(protocol_commit=PROTOCOL_COMMIT, protocol_sha256=file_digest(Path(__file__).with_name('PROTOCOL.md')),
                    code_sha256=file_digest(__file__), source_hashes=source_hashes,
                    metrics_sha256=file_digest(metrics_file), selected_summary_sha256=file_digest(source_summary),
                    source_path=str(Path(data_dir).resolve()), metrics_path=str(Path(metrics_file).resolve()),
                    additional_filters_hz=list(CUTOFFS), source_preprocessed_hz=20,
                    relative_height_bands_are_measured_noise=False, data_license='CC BY-NC-SA 4.0')
    for name, value in [('summary.json', result), ('manifest.json', manifest)]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('pitches', 'segment_event_filter_rows', 'interval_order_classifications')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--metrics-dir', required=True, type=Path)
    parser.add_argument('--source-summary', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must remain inside peak_stability package')
    run(args.data_dir, args.metrics_dir, args.source_summary, args.output)


if __name__ == '__main__':
    main()
