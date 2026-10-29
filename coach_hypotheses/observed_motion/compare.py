"""Frozen-cohort, player-equal observed motion differences; no predictive gate."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

PREREGISTRATION_COMMIT = '4664982b8a5638c76d03927a2e1a94c56a2a3e8f'
SUMMARY_SHA256 = '32ca2146b29a31ddf212988711705832e05366a0300300999f4535ed5fa0e490'
METRICS_SHA256 = 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4'
CUTOFFS = (10, 20, 30)
SEED, BOOTSTRAPS = 20261029, 2000
PRIMARY = ('sequence_lag_ms', 'lead_knee_extension_deg',
           'hip_shoulder_sep_fp10_deg', 'arm_slot_poi_deg')
SECONDARY = ('pelvis_peak_dps', 'torso_peak_dps', 'lead_knee_peak_extension_dps',
             'stride_reported', 'wrist_height_br_m')
METRICS = PRIMARY + SECONDARY
UNITS = dict(zip(METRICS, ('ms', 'deg', 'deg', 'deg', 'deg/s', 'deg/s', 'deg/s',
                         'original reported stride_length (%body height label; no conversion)', 'm')))


def digest(path):
    """Hash files in bounded memory."""
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_inputs(summary_path, metrics_path):
    """Fail closed on changed sources, identities, filters, or finite support."""
    if digest(summary_path) != SUMMARY_SHA256 or digest(metrics_path) != METRICS_SHA256:
        raise ValueError('changed_frozen_source')
    source = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    pairs = pd.DataFrame([r for r in source['pair_records'] if r['independent_event_stable'] is True])
    if len(pairs) != 35 or pairs.user.nunique() != 23 or pairs.recorded_speed_gap_kmh.ge(1).sum() != 14:
        raise ValueError('changed_frozen_cohort')
    if pairs.duplicated(['user', 'pitch_i', 'pitch_j']).any():
        raise ValueError('duplicate_frozen_pair')
    metrics = pd.read_csv(metrics_path, dtype={'session_pitch': str, 'user': str})
    return pairs, metrics


def pair_differences(pairs, metrics):
    """Subtract slower from faster, preserving all three nominal filter rows."""
    required = ['user', 'session_pitch', 'cutoff_hz', 'pitch_speed_kmh', *METRICS]
    if not set(required).issubset(metrics.columns) or metrics.duplicated(['session_pitch', 'cutoff_hz']).any():
        raise ValueError('invalid_metric_schema_or_duplicate')
    lookup = metrics.set_index(['session_pitch', 'cutoff_hz'])
    rows = []
    for pair in pairs.to_dict('records'):
        user, fast, slow = str(pair['user']), pair['faster_pitch'], pair['slower_pitch']
        if fast == slow or {fast, slow} != {pair['pitch_i'], pair['pitch_j']}:
            raise ValueError('invalid_pair_identity')
        gap = float(pair['recorded_speed_gap_kmh'])
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError('invalid_speed_order')
        for cutoff in CUTOFFS:
            if (fast, cutoff) not in lookup.index or (slow, cutoff) not in lookup.index:
                raise ValueError('missing_pitch_filter')
            high, low = lookup.loc[(fast, cutoff)], lookup.loc[(slow, cutoff)]
            if str(high.user) != user or str(low.user) != user:
                raise ValueError('cross_player_pair')
            values = np.asarray([high[list(METRICS)], low[list(METRICS)]], float)
            speed = np.asarray([high.pitch_speed_kmh, low.pitch_speed_kmh], float)
            if not np.isfinite(values).all() or not np.isfinite(speed).all():
                raise ValueError('nonfinite_selected_metric')
            if not np.isclose(speed[0] - speed[1], gap, atol=1e-10, rtol=0):
                raise ValueError('recorded_speed_mismatch')
            rows.append(dict(user=user, pitch_i=pair['pitch_i'], pitch_j=pair['pitch_j'],
                             faster_pitch=fast, slower_pitch=slow, cutoff_hz=cutoff,
                             recorded_speed_gap_kmh=gap,
                             **dict(zip(METRICS, values[0] - values[1]))))
    if not rows:
        raise ValueError('empty_pair_cohort')
    return pd.DataFrame(rows).sort_values(['user', 'faster_pitch', 'slower_pitch', 'cutoff_hz']).reset_index(drop=True)


def player_differences(rows):
    """Average pairs within each player before any between-player statistic."""
    if rows.empty or not set(rows.cutoff_hz) == set(CUTOFFS):
        raise ValueError('empty_or_incomplete_filters')
    counts = rows.groupby(['user', 'cutoff_hz']).size().rename('pairs')
    result = rows.groupby(['user', 'cutoff_hz'])[list(METRICS)].mean().join(counts).reset_index()
    if (not result.groupby('user').size().eq(3).all() or
            not result.groupby('user').pairs.nunique().eq(1).all() or
            not np.isfinite(result[list(METRICS)]).all().all()):
        raise ValueError('invalid_player_filter_support')
    return result.sort_values(['user', 'cutoff_hz']).reset_index(drop=True)


def summarize_filter(values, bootstraps=BOOTSTRAPS, seed=SEED):
    """Percentile 95% interval from resampling players, not pairs or interventions."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or bootstraps < 1:
        raise ValueError('invalid_player_values')
    indices = np.random.default_rng(seed).integers(0, len(values), size=(bootstraps, len(values)))
    interval = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return dict(players=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                positive_players=int((values > 0).sum()), negative_players=int((values < 0).sum()),
                zero_players=int((values == 0).sum()), bootstrap_mean_percentile_95=interval.tolist())


def summarize_metric(players, metric):
    """Retain signs across filters and apply only the preregistered exploratory flag."""
    if metric not in METRICS:
        raise ValueError('unknown_metric')
    wide = players.pivot(index='user', columns='cutoff_hz', values=metric).reindex(columns=CUTOFFS)
    if wide.empty or not np.isfinite(wide).all().all():
        raise ValueError('incomplete_metric_support')
    signs = np.sign(wide.to_numpy())
    filters = {str(c): summarize_filter(wide[c].to_numpy()) for c in CUTOFFS}
    means = np.asarray([filters[str(c)]['mean'] for c in CUTOFFS])
    intervals = np.asarray([filters[str(c)]['bootstrap_mean_percentile_95'] for c in CUTOFFS])
    direction = 1 if (means > 0).all() else -1 if (means < 0).all() else 0
    consistent_ci = bool(direction and ((intervals > 0).all() if direction == 1 else (intervals < 0).all()))
    proportions = [float(np.mean(signs[:, i] == direction)) if direction else 0. for i in range(3)]
    shared_player_fraction = float(np.mean((signs == direction).all(axis=1))) if direction else 0.
    flag = bool(metric in PRIMARY and consistent_ci and min(proportions) >= .75)
    return dict(role='primary' if metric in PRIMARY else 'secondary', unit=UNITS[metric],
                filters=filters, shared_mean_direction=direction,
                all_intervals_same_direction=consistent_ci,
                fraction_players_in_shared_mean_direction_by_filter=dict(zip(map(str, CUTOFFS), proportions)),
                fraction_players_in_shared_mean_direction_all_filters=shared_player_fraction,
                exploratory_followup_flag=flag,
                player_positive_negative_reversal_count=int(((signs > 0).any(axis=1) & (signs < 0).any(axis=1)).sum()),
                player_any_sign_change_including_zero_count=int((np.ptp(signs, axis=1) != 0).sum()),
                player_deltas=[dict(user=str(user), pairs=int(players[players.user == user].pairs.iloc[0]),
                                    differences=dict(zip(map(str, CUTOFFS), map(float, values))),
                                    signs=dict(zip(map(str, CUTOFFS), map(int, np.sign(values)))))
                               for user, values in zip(wide.index, wide.to_numpy())])


def compare(rows):
    """Keep the full cohort and original >=1 km/h subset separate."""
    output = {}
    for name, selected in [('independent_event_stable', rows),
                           ('original_gap_at_least_1_kmh', rows[rows.recorded_speed_gap_kmh >= 1])]:
        players = player_differences(selected)
        output[name] = dict(pairs=len(selected) // len(CUTOFFS), players=players.user.nunique(),
                            metrics={metric: summarize_metric(players, metric) for metric in METRICS})
    return output


def independent_check(summary_path, metrics_path, result):
    """Independent stdlib CSV arithmetic checks every pair, player and filter summary."""
    source = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    with Path(metrics_path).open(encoding='utf-8-sig', newline='') as stream:
        lookup = {(r['session_pitch'], int(r['cutoff_hz'])): r for r in csv.DictReader(stream)}
    comparisons, max_error = 0, 0.
    for name, cohort in result.items():
        pairs = [r for r in source['pair_records'] if r['independent_event_stable'] and
                 (name == 'independent_event_stable' or r['recorded_speed_gap_kmh'] >= 1)]
        if len(pairs) != cohort['pairs']:
            raise AssertionError('independent_pair_count_mismatch')
        for metric in METRICS:
            observed = cohort['metrics'][metric]
            for cutoff in CUTOFFS:
                buckets = {}
                for pair in pairs:
                    high, low = lookup[(pair['faster_pitch'], cutoff)], lookup[(pair['slower_pitch'], cutoff)]
                    buckets.setdefault(pair['user'], []).append(float(high[metric]) - float(low[metric]))
                values = {user: statistics.mean(deltas) for user, deltas in buckets.items()}
                if (len(values) != cohort['players'] or
                        {p['user'] for p in observed['player_deltas']} != set(values) or
                        any(p['pairs'] != len(buckets[p['user']]) for p in observed['player_deltas'])):
                    raise AssertionError('independent_player_support_mismatch')
                expected = dict(mean=statistics.mean(values.values()), median=statistics.median(values.values()),
                                minimum=min(values.values()), maximum=max(values.values()),
                                positive_players=sum(v > 0 for v in values.values()),
                                negative_players=sum(v < 0 for v in values.values()), zero_players=sum(v == 0 for v in values.values()))
                for key, value in expected.items():
                    difference = abs(observed['filters'][str(cutoff)][key] - value)
                    max_error = max(max_error, difference)
                    comparisons += 1
                for player in observed['player_deltas']:
                    difference = abs(player['differences'][str(cutoff)] - values[player['user']])
                    max_error = max(max_error, difference)
                    comparisons += 1
    if max_error > 1e-10:
        raise AssertionError('independent_arithmetic_mismatch')
    return dict(method='stdlib csv + statistics; re-read raw frozen metric rows; independent pair subtraction and player means',
                checked_values=comparisons, maximum_absolute_difference=max_error, passed=True)


def run(summary_path, metrics_path, output):
    pairs, metrics = load_inputs(summary_path, metrics_path)
    rows = pair_differences(pairs, metrics)
    cohorts = compare(rows)
    verification = independent_check(summary_path, metrics_path, cohorts)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(difference='recorded faster minus slower', cohorts=cohorts,
                  selected_pairs=pairs[['user', 'pitch_i', 'pitch_j', 'faster_pitch', 'slower_pitch',
                                        'recorded_speed_gap_kmh']].to_dict('records'),
                  bootstrap=dict(replicates=BOOTSTRAPS, seed=SEED, unit='player',
                                 interval='2.5th/97.5th percentile of player-equal mean',
                                 paired_filter_resamples=True, uncertainty_is_intervention_or_measurement_error=False),
                  upstream_full_sig_preprocessed_hz=20, added_filter_cutoffs_hz=list(CUTOFFS),
                  conditions='selected development cohort by recorded speed, provided arm load and event stability; conditional exploratory association',
                  speed_prediction_gate_changed=False, intervention_effect_validated=False,
                  same_capacity_or_work_validated=False, coaching_recommendation_ready=False,
                  data_source='OpenBiomechanics / Driveline Baseball', data_license='CC BY-NC-SA 4.0',
                  source_url='https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching')
    manifest = dict(protocol_commit=PREREGISTRATION_COMMIT, protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),
                    code_sha256=digest(__file__), test_sha256=digest(Path(__file__).with_name('test_compare.py')),
                    summary_source=dict(path=str(Path(summary_path).resolve()), sha256=digest(summary_path)),
                    metric_source=dict(path=str(Path(metrics_path).resolve()), sha256=digest(metrics_path)),
                    tracked_outputs=['comparison.json', 'comparison_manifest.json', 'comparison_verification.json'],
                    local_ignored_csv=['comparison_pairs.csv', 'comparison_players.csv'])
    for name, value in [('comparison', result), ('comparison_manifest', manifest), ('comparison_verification', verification)]:
        (output / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    rows.to_csv(output / 'comparison_pairs.csv', index=False)
    pd.concat([player_differences(rows).assign(cohort='independent_event_stable'),
               player_differences(rows[rows.recorded_speed_gap_kmh >= 1]).assign(cohort='original_gap_at_least_1_kmh')]).to_csv(output / 'comparison_players.csv', index=False)
    return result, verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    result, verification = run(args.summary, args.metrics, args.output)
    print(json.dumps(dict(cohorts={name: dict(pairs=value['pairs'], players=value['players'],
                                            flagged=[m for m in PRIMARY if value['metrics'][m]['exploratory_followup_flag']])
                                  for name, value in result['cohorts'].items()}, verification=verification), indent=2))
