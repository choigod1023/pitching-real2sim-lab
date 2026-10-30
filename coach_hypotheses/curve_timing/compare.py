"""Compare positive-component centroids and point peaks without target tuning."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..observed_sensitivity.analysis import load_sources, OFFSETS
from ..observed_tradeoffs.analysis import file_digest, CUTOFFS

PROTOCOL_COMMIT = 'b784cac8d0431b1b8756229ad6007ee2cde052b9'
METRICS_HASH = 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4'
FEATURES = ('peak_lag_ms', 'centroid_lag_ms')


def temporal_centroid(time, values, begin, end):
    """Exact area/first moment of the positive part of a linear interpolant."""
    time, values = np.asarray(time, float), np.asarray(values, float)
    if (time.ndim != 1 or len(time) < 2 or time.shape != values.shape or
            not np.isfinite(np.r_[time, values, begin, end]).all() or np.any(np.diff(time) <= 0) or
            not time[0] <= begin < end <= time[-1]):
        raise ValueError('invalid_centroid_support')
    knots = np.r_[begin, time[(time > begin) & (time < end)], end]
    y = np.interp(knots, time, values)
    crossed = y[:-1] * y[1:] < 0
    zeros = knots[:-1][crossed] - y[:-1][crossed] * np.diff(knots)[crossed] / np.diff(y)[crossed]
    knots = np.sort(np.r_[knots, zeros])
    y = np.maximum(0., np.interp(knots, time, values))
    h = np.diff(knots)
    area = float(np.sum(h * (y[:-1] + y[1:]) / 2))
    moment = float(np.sum(h * ((2 * knots[:-1] + knots[1:]) * y[:-1] +
                               (knots[:-1] + 2 * knots[1:]) * y[1:]) / 6))
    if area <= 0 or not np.isfinite([area, moment]).all():
        raise ValueError('empty_or_nonfinite_positive_area')
    return moment / area


def pitch_features(module, prepared, expected):
    time, matrix, columns, events, _ = prepared
    nominal, cases = [], []
    for cutoff in CUTOFFS:
        signals = module.filtered_signals(matrix, columns, cutoff)
        for pk, br in OFFSETS:
            begin, end = events['pkh_time'] + pk / 360., events['BR_time'] + br / 360.
            if not begin < events['fp_10_time'] < end:
                raise ValueError('shifted_event_order')
            centers, peaks = [], []
            for segment in ('pelvis', 'torso'):
                y = signals[segment + '_velo_z']
                centers.append(temporal_centroid(time, y, begin, end))
                peaks.append(module.window_peak(time, y, begin, end)[0])
            row = dict(cutoff_hz=cutoff, pkh_shift_frames=pk, br_shift_frames=br,
                       peak_lag_ms=float((peaks[1] - peaks[0]) * 1000),
                       centroid_lag_ms=float((centers[1] - centers[0]) * 1000))
            cases.append(row)
            if pk == br == 0:
                if abs(row['peak_lag_ms'] - expected.loc[cutoff, 'sequence_lag_ms']) > 1e-7:
                    raise ValueError('peak_reference_not_reproduced')
                nominal.append(row.copy())
    ranges = {feature + '_range_ms': max(r[feature] for r in cases) - min(r[feature] for r in cases) for feature in FEATURES}
    return nominal, cases, ranges


def make_pairs(frame):
    if (frame.empty or frame.session_pitch.duplicated().any() or
            not np.isfinite(frame[['pitch_speed_kmh', *FEATURES]].to_numpy(float)).all()):
        raise ValueError('invalid_pair_frame')
    rows = []
    for user, player in frame.groupby('user', sort=True):
        if len(player) < 2:
            raise ValueError('insufficient_player_repeats')
        for first, second in itertools.combinations(player.sort_values('session_pitch').to_dict('records'), 2):
            rows.append(dict(user=str(user), pitch_i=first['session_pitch'], pitch_j=second['session_pitch'],
                             truth_delta_kmh=second['pitch_speed_kmh'] - first['pitch_speed_kmh'],
                             **{feature: second[feature] - first[feature] for feature in FEATURES}))
    return pd.DataFrame(rows)


def leave_one_player_out(pairs, feature):
    if feature not in FEATURES or pairs.user.nunique() < 2:
        raise ValueError('invalid_feature_or_player_count')
    if not np.isfinite(pairs[[feature, 'truth_delta_kmh']].to_numpy(float)).all():
        raise ValueError('nonfinite_model_input')
    predictions, splits = [], []
    for player in sorted(pairs.user.unique()):
        train, test = pairs[pairs.user != player], pairs[pairs.user == player]
        x, y = train[feature].to_numpy(float), train.truth_delta_kmh.to_numpy(float)
        weights = 1. / train.groupby('user').user.transform('size').to_numpy(float)
        scale = float(np.sqrt(np.average(x * x, weights=weights)))
        if scale == 0:
            scale, beta = 1., 0.
        else:
            z = x / scale
            beta = float(np.sum(weights * z * y) / (np.sum(weights * z * z) + 10.))
        rows = test[['user', 'pitch_i', 'pitch_j', 'truth_delta_kmh']].copy()
        rows['prediction_delta_kmh'] = test[feature].to_numpy(float) / scale * beta
        rows['baseline_delta_kmh'] = 0.
        predictions.append(rows)
        splits.append(dict(heldout_user=str(player), train_players=len(set(train.user)),
                           heldout_absent_from_train=player not in set(train.user), scale=scale, beta=beta))
    return pd.concat(predictions, ignore_index=True), splits


def score(predictions):
    frame = predictions.copy()
    frame['model_error'] = abs(frame.prediction_delta_kmh - frame.truth_delta_kmh)
    frame['baseline_error'] = abs(frame.truth_delta_kmh)
    players = frame.groupby('user')[['model_error', 'baseline_error']].mean()
    difference = (players.model_error - players.baseline_error).to_numpy()
    if not len(difference) or not np.isfinite(difference).all():
        raise ValueError('invalid_score_input')
    rng = np.random.default_rng(20261030)
    means = difference[rng.integers(0, len(difference), size=(2000, len(difference)))].mean(axis=1)
    return dict(model_mae_kmh=float(players.model_error.mean()), zero_delta_mae_kmh=float(players.baseline_error.mean()),
                paired_mae_difference_kmh=float(difference.mean()),
                player_bootstrap_difference_ci95=np.quantile(means, [.025, .975]).tolist(),
                players=len(players), pairs=len(frame))


def development_checks(summaries, players, pitches, integrity):
    primary = summaries['20']
    checks = dict(minimum_players=players >= 50, minimum_pitches=pitches >= 150, integrity=integrity is True,
                  primary_better=primary['model_mae_kmh'] < primary['zero_delta_mae_kmh'],
                  upper_ci_below_zero=primary['player_bootstrap_difference_ci95'][1] < 0,
                  other_filters_not_worse=all(summaries[str(c)]['model_mae_kmh'] <= summaries[str(c)]['zero_delta_mae_kmh'] for c in (10, 30)))
    return dict(checks=checks, development_prediction_criteria_pass=all(checks.values()),
                real2sim_scientific_gate_changed=False, coaching_recommendation_ready=False,
                same_capacity_validated=False, athlete_effect_validated=False)


def run(data_dir, metrics_dir, output):
    metrics_file = Path(metrics_dir) / 'results/pitch_metrics.csv'
    if file_digest(metrics_file) != METRICS_HASH:
        raise ValueError('changed_frozen_metrics')
    reference = pd.read_csv(metrics_file, dtype={'user': str})
    ids = set(reference.session_pitch)
    if len(ids) != 399 or reference.user.nunique() != 98:
        raise ValueError('changed_cohort')
    module, prepared, source_hashes = load_sources(data_dir, Path(metrics_dir) / 'metrics.py', ids)
    lookup = reference.set_index(['session_pitch', 'cutoff_hz'])
    features, event_rows, ranges = [], [], []
    for pitch in sorted(ids):
        nominal, cases, widths = pitch_features(module, prepared[pitch], lookup.loc[pitch])
        user, speed = str(lookup.loc[(pitch, 20), 'user']), float(lookup.loc[(pitch, 20), 'pitch_speed_kmh'])
        features.extend(dict(session_pitch=pitch, user=user, pitch_speed_kmh=speed, **row) for row in nominal)
        event_rows.extend(dict(session_pitch=pitch, **row) for row in cases)
        ranges.append(dict(session_pitch=pitch, **widths))
    features = pd.DataFrame(features)
    widths = pd.DataFrame(ranges)
    stability = {name: dict(median_range_ms=float(widths[name + '_range_ms'].median()),
                            p95_range_ms=float(widths[name + '_range_ms'].quantile(.95)),
                            maximum_range_ms=float(widths[name + '_range_ms'].max())) for name in FEATURES}
    stability['engineering_stability_improved'] = (stability['centroid_lag_ms']['median_range_ms'] < stability['peak_lag_ms']['median_range_ms'] and
                                                  stability['centroid_lag_ms']['p95_range_ms'] <= stability['peak_lag_ms']['p95_range_ms'])
    summaries = {name: {} for name in FEATURES}
    all_predictions, splits = [], {}
    for cutoff in CUTOFFS:
        pairs = make_pairs(features[features.cutoff_hz.eq(cutoff)])
        if len(pairs) != 666:
            raise ValueError('unexpected_pair_count')
        for feature in FEATURES:
            predictions, audit = leave_one_player_out(pairs, feature)
            summaries[feature][str(cutoff)] = score(predictions)
            all_predictions.append(predictions.assign(feature=feature, cutoff_hz=cutoff))
            splits[feature + '_' + str(cutoff)] = audit
    result = dict(players=98, pitches=399, pairs_per_filter=666, stability=stability, prediction=summaries,
                  verdicts={feature: development_checks(values, 98, 399, all(r['heldout_absent_from_train'] for r in splits[feature + '_20'])) for feature, values in summaries.items()},
                  numerical_precision_is_measurement_accuracy=False, existing_failures_preserved=True,
                  intervention_effect_validated=False, spin_or_movement_improvement_validated=False)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / 'features.csv', index=False)
    pd.DataFrame(event_rows).to_csv(output / 'event_features.csv', index=False)
    widths.to_csv(output / 'stability_ranges.csv', index=False)
    pd.concat(all_predictions).to_csv(output / 'predictions.csv', index=False)
    manifest = dict(protocol_commit=PROTOCOL_COMMIT, protocol_sha256=file_digest(Path(__file__).with_name('PROTOCOL.md')),
                    code_sha256=file_digest(__file__), metrics_sha256=file_digest(metrics_file), source_hashes=source_hashes,
                    source_processed_hz=20, additional_filter_hz=list(CUTOFFS), alpha=10., bootstrap_seed=20261030,
                    source_path=str(Path(data_dir).resolve()), data_license='CC BY-NC-SA 4.0')
    for name, value in [('summary.json', result), ('manifest.json', manifest), ('split_audit.json', splits)]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--metrics-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must remain inside curve_timing package')
    run(args.data_dir, args.metrics_dir, args.output)


if __name__ == '__main__':
    main()
