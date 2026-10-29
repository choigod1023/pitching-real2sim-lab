"""Player-disjoint retrospective evaluation; no physical-model calibration."""
import numpy as np
import pandas as pd

from .metrics import FEATURES, LOADS, CUTOFFS

SEED = 20261029
ALPHAS = (.01, .1, 1., 10., 100.)
ASSOCIATION_METRICS = list(dict.fromkeys(FEATURES + [
    'torso_peak_time_br_ms', 'pelvis_peak_time_fp10_ms', 'torso_peak_time_fp10_ms',
    'pelvis_peak_dps', 'torso_peak_dps', 'lead_knee_extension_at_fp10_dps',
    'hip_shoulder_sep_fp100_deg', 'elbow_varus_bw_height_pct',
    'shoulder_resultant_peak_bw']))


def demean(values, groups):
    values = np.asarray(values, float)
    result = values.copy()
    for group in np.unique(groups):
        mask = groups == group
        result[mask] -= values[mask].mean(axis=0)
    return result


def player_weights(groups):
    groups = np.asarray(groups)
    return np.array([1. / np.sum(groups == group) for group in groups])


def fit_ridge(x, y, groups, alpha):
    """Equal player weight; train-only scale, intercept and constant handling."""
    weights = player_weights(groups)
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=weights))
    constant = scale < 1e-10
    scale[constant] = 1.
    z = (x - center) / scale
    z[:, constant] = 0.
    intercept = float(np.average(y, weights=weights))
    coef = np.linalg.solve(z.T @ (weights[:, None] * z) + alpha * np.eye(x.shape[1]),
                           z.T @ (weights * (y - intercept)))
    return center, scale, constant, coef, intercept


def predict_ridge(model, x):
    center, scale, constant, coef, intercept = model
    z = (x - center) / scale
    z[:, constant] = 0.
    return z @ coef + intercept


def choose_alpha(x, y, groups):
    """Inner folds only see players in the outer training set."""
    players = np.unique(groups)
    if len(players) < 2:
        raise ValueError('too_few_training_players')
    folds = np.array_split(players, min(5, len(players)))
    scores = []
    for alpha in ALPHAS:
        errors = []
        for fold in folds:
            test = np.isin(groups, fold)
            model = fit_ridge(x[~test], y[~test], groups[~test], alpha)
            residual = np.abs(predict_ridge(model, x[test]) - y[test])
            errors.extend(float(residual[groups[test] == p].mean()) for p in fold)
        scores.append(float(np.mean(errors)))
    minimum = min(scores)
    return max(a for a, score in zip(ALPHAS, scores) if abs(score - minimum) <= 1e-12)


def nested_predictions(frame, within=True):
    """Held player's outcome mean is used only to score retrospective changes."""
    ordered = frame.sort_values(['user', 'session_pitch']).copy()
    if not within:
        ordered = ordered.groupby('user', as_index=False)[FEATURES + ['pitch_speed_kmh']].mean()
        ordered['session_pitch'] = ordered.user
    groups = ordered.user.to_numpy(str)
    raw_x = ordered[FEATURES].to_numpy(float)
    raw_y = ordered.pitch_speed_kmh.to_numpy(float)
    x = demean(raw_x, groups) if within else raw_x
    y = demean(raw_y, groups) if within else raw_y
    rows, audits = [], []
    for held in np.unique(groups):
        test = groups == held
        alpha = choose_alpha(x[~test], y[~test], groups[~test])
        model = fit_ridge(x[~test], y[~test], groups[~test], alpha)
        predicted = predict_ridge(model, x[test])
        baseline = np.zeros(test.sum()) if within else np.full(test.sum(), raw_y[~test].mean())
        if within:
            predicted -= predicted.mean()
        for key, truth, pred, base, raw in zip(ordered.session_pitch[test], y[test], predicted, baseline, raw_y[test]):
            rows.append({'user': held, 'session_pitch': str(key), 'truth': float(truth),
                         'prediction': float(pred), 'baseline': float(base),
                         'observed_speed_kmh': float(raw), 'alpha': alpha})
        audits.append({'held_player': held, 'training_players': sorted(set(groups[~test])),
                       'overlap': bool(held in groups[~test]), 'alpha': alpha,
                       'train_scale': model[1].tolist()})
    return pd.DataFrame(rows), audits


def bootstrap_mean(values, repetitions=2000):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('invalid_bootstrap_values')
    indices = np.random.default_rng(SEED).integers(0, len(values), (repetitions, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()


def prediction_summary(predictions):
    errors = predictions.assign(model_error=np.abs(predictions.prediction - predictions.truth),
                                baseline_error=np.abs(predictions.baseline - predictions.truth))
    players = errors.groupby('user')[['model_error', 'baseline_error']].mean()
    difference = players.model_error - players.baseline_error
    return {'players': len(players), 'pitches': len(errors),
            'player_balanced_model_mae_kmh': float(players.model_error.mean()),
            'player_balanced_baseline_mae_kmh': float(players.baseline_error.mean()),
            'pooled_model_mae_kmh': float(errors.model_error.mean()),
            'pooled_baseline_mae_kmh': float(errors.baseline_error.mean()),
            'paired_mae_difference_kmh': float(difference.mean()),
            'paired_difference_cluster_ci95': bootstrap_mean(difference),
            'players_model_better': int((difference < 0).sum())}, players.reset_index()


def associations(frame, repetitions=2000):
    """Cluster sufficient moments support unequal pitch counts without pooling bias."""
    records = []
    targets = ['pitch_speed_kmh', *LOADS]
    for metric in ASSOCIATION_METRICS:
        for target in targets:
            if metric == target:
                continue
            subset = frame[['user', metric, target]].dropna()
            subset = subset[subset.groupby('user').user.transform('size') >= 2]
            blocks = list(subset.groupby('user', sort=True))
            if len(blocks) < 3:
                continue
            indices = np.random.default_rng(SEED).integers(0, len(blocks), (repetitions, len(blocks)))
            for level in ('between', 'within'):
                moments = []
                for _, block in blocks:
                    x, y = block[metric].to_numpy(float), block[target].to_numpy(float)
                    if level == 'between':
                        x, y = np.array([x.mean()]), np.array([y.mean()])
                    else:
                        x, y = x - x.mean(), y - y.mean()
                    moments.append([x.mean(), y.mean(), (x*x).mean(), (y*y).mean(), (x*y).mean()])
                moments = np.array(moments)
                estimates = np.vstack([moments.mean(axis=0), moments[indices].mean(axis=1)])
                cov = estimates[:, 4] - estimates[:, 0] * estimates[:, 1]
                variance = np.maximum(estimates[:, 2] - estimates[:, 0] ** 2, 0) * np.maximum(estimates[:, 3] - estimates[:, 1] ** 2, 0)
                r = np.divide(cov, np.sqrt(variance), out=np.full(len(cov), np.nan), where=variance > 1e-20)
                valid = r[1:][np.isfinite(r[1:])]
                ci = np.quantile(valid, [.025, .975]) if len(valid) else [None, None]
                records.append({'metric': metric, 'target': target, 'level': level,
                                'players': len(blocks), 'pitches': len(subset),
                                'r': float(r[0]) if np.isfinite(r[0]) else None,
                                'ci95_low': ci[0], 'ci95_high': ci[1],
                                'valid_bootstraps': len(valid)})
    return pd.DataFrame(records)


def filter_sensitivity(frame):
    results = []
    columns = list(dict.fromkeys(ASSOCIATION_METRICS + ['elbow_varus_peak_nm',
                    'shoulder_compression_peak_n', 'shoulder_distraction_peak_n']))
    for column in columns:
        wide = frame.pivot(index='session_pitch', columns='cutoff_hz', values=column)
        if wide.isna().any().any():
            raise ValueError('noncommon_filter_support:' + column)
        spread = wide.max(axis=1) - wide.min(axis=1)
        results.append({'metric': column, 'median_range': float(spread.median()),
                        'p95_range': float(spread.quantile(.95)), 'max_range': float(spread.max())})
    lag = frame.pivot(index='session_pitch', columns='cutoff_hz', values='sequence_lag_ms')
    sign_changes = int(((lag.min(axis=1) < 0) & (lag.max(axis=1) > 0)).sum())
    return pd.DataFrame(results), sign_changes


def scientific_gate(summaries, integrity):
    primary = summaries['20']['within']
    criteria = {'sample_size': primary['players'] >= 50 and primary['pitches'] >= 150,
                'split_and_preprocessing_integrity': bool(integrity),
                'primary_mae_better': primary['paired_mae_difference_kmh'] < 0,
                'primary_ci_upper_negative': primary['paired_difference_cluster_ci95'][1] < 0,
                'other_filters_noninferior': all(summaries[str(c)]['within']['paired_mae_difference_kmh'] <= 0 for c in (10, 30))}
    return {'verdict': 'PASS' if all(criteria.values()) else 'FAIL', 'criteria': criteria}
