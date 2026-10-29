"""Source verification, representation controls and grouped model comparisons."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..stage1.metrics import COLUMNS, CUTOFFS, FEATURES, prepare_pitch, filtered_signals, at_event
from ..stage1.modeling import demean, fit_ridge, predict_ridge, player_weights, ALPHAS
from ..stage1.run import file_hash

STABLE = ['lead_knee_extension_deg', 'stride_reported', 'stride_reported_sq',
          'hip_shoulder_sep_fp10_deg', 'hip_shoulder_sep_peak_deg',
          'wrist_height_br_m', 'arm_slot_poi_deg']
EXTENDED = FEATURES + ['pelvis_peak_dps', 'torso_peak_dps']
CHANNELS = ['pelvis_velo_z', 'torso_velo_z', 'lead_knee_velo_x', 'torso_pelvis_angle_z']
WAVE = [f'phase_{channel}_{i}' for channel in CHANNELS for i in range(5)]
FAMILIES = {'original12': FEATURES, 'stable7': STABLE, 'extended14': EXTENDED,
            'waveform20': WAVE, 'mae14': EXTENDED}
PRIORITY = {'null': 0, 'stable7': 1, 'original12': 2, 'extended14': 3, 'mae14': 4, 'waveform20': 5}


def verify_sources(data, frozen):
    manifest = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    checks = {name: file_hash(data / name) == spec['sha256'] for name, spec in manifest['sources'].items()}
    if not all(checks.values()):
        raise ValueError('changed_frozen_sources')
    return checks


def phase_features(time, signals, fp, br):
    if not fp < br:
        raise ValueError('invalid_phase_events')
    samples = np.linspace(fp, br, 5)
    return {f'phase_{channel}_{i}': at_event(time, signals[channel], event)
            for channel in CHANNELS for i, event in enumerate(samples)}


def add_waveforms(data, frozen_frame):
    groups = {}
    for name, columns in COLUMNS.items():
        raw = pd.read_csv(data / f'full_sig/{name}.csv', usecols=[
            'session_pitch', 'time', 'pkh_time', 'fp_10_time', 'BR_time', 'fp_100_time', *columns])
        groups[name] = dict(tuple(raw.groupby('session_pitch', sort=False)))
    rows = []
    for key in sorted(frozen_frame.session_pitch.unique()):
        time, matrix, columns, events, _ = prepare_pitch({name: group[key] for name, group in groups.items()})
        for cutoff in CUTOFFS:
            signals = filtered_signals(matrix, columns, cutoff)
            row = phase_features(time, signals, events['fp_10_time'], events['BR_time'])
            rows.append(dict(row, session_pitch=key, cutoff_hz=cutoff))
    result = frozen_frame.merge(pd.DataFrame(rows), on=['session_pitch', 'cutoff_hz'], validate='one_to_one')
    if len(result) != len(frozen_frame) or not np.isfinite(result[WAVE]).all().all():
        raise ValueError('waveform_cohort_mismatch')
    return result


def data_diagnostics(frame):
    primary = frame[frame.cutoff_hz == 20]
    count = primary.groupby('user').size()
    speed = primary.groupby('user').pitch_speed_kmh.agg(['min', 'max', 'std'])
    records = []
    for column in EXTENDED:
        mean = primary.groupby('user')[column].mean()
        within = demean(primary[column].to_numpy(float), primary.user.to_numpy(str))
        weights = player_weights(primary.user.to_numpy(str))
        within_sd = np.sqrt(np.average(within ** 2, weights=weights))
        between_sd = float(mean.std(ddof=0))
        wide = frame.pivot(index='session_pitch', columns='cutoff_hz', values=column)
        spread = float((wide.max(axis=1) - wide.min(axis=1)).median())
        records.append({'feature': column, 'within_sd': float(within_sd),
                        'between_sd': between_sd, 'within_between_ratio': float(within_sd/between_sd) if between_sd else None,
                        'median_filter_range': spread,
                        'filter_range_within_sd_ratio': float(spread/within_sd) if within_sd else None})
    x = demean(primary[FEATURES].to_numpy(float), primary.user.to_numpy(str))
    scale = x.std(axis=0)
    z = x[:, scale > 1e-10]/scale[scale > 1e-10]
    singular = np.linalg.svd(z, compute_uv=False)
    speeds_mph = primary.pitch_speed_kmh.to_numpy(float)/1.609344
    report = {'players': len(count), 'pitches': len(primary), 'counts_per_player': count.value_counts().sort_index().to_dict(),
              'speed_range_median_kmh': float((speed['max']-speed['min']).median()),
              'speed_range_p95_kmh': float((speed['max']-speed['min']).quantile(.95)),
              'players_zero_observed_speed_range': int((speed['max']==speed['min']).sum()),
              'pitches_on_one_tenth_mph_grid': int(np.isclose(speeds_mph*10, np.round(speeds_mph*10), atol=1e-8).sum()),
              'one_tenth_mph_in_kmh': .1609344, 'this_is_not_radar_accuracy': True,
              'centered_feature_rank': int(np.linalg.matrix_rank(z)),
              'centered_condition_number': float(singular[0]/singular[-1]),
              'within_degrees_of_freedom': int(len(primary)-len(count))}
    return report, pd.DataFrame(records)


def fit_model(x, y, groups, alpha, loss='ridge'):
    model = fit_ridge(x, y, groups, alpha)
    if loss == 'ridge':
        return model
    if loss != 'mae':
        raise ValueError('invalid_loss')
    center, scale, constant, coef, intercept = model
    z = (x-center)/scale
    z[:, constant] = 0.
    weights = player_weights(groups)

    def objective(parameters):
        residual = z @ parameters[:-1] + parameters[-1] - y
        smooth = np.sqrt(residual**2 + .05**2)
        value = np.sum(weights*(smooth-.05)) + alpha/2*np.sum(parameters[:-1]**2)
        weighted = weights*residual/smooth
        gradient = np.r_[z.T @ weighted + alpha*parameters[:-1], weighted.sum()]
        return value, gradient

    result = minimize(objective, np.r_[coef, intercept], jac=True, method='L-BFGS-B',
                      bounds=[(0.,0.) if c else (None,None) for c in constant]+[(None,None)],
                      tol=1e-8, options={'maxiter': 200})
    if not result.success:
        raise RuntimeError('mae_optimizer_failed:' + result.message)
    return center, scale, constant, result.x[:-1], float(result.x[-1])


def balanced_mae(prediction, truth, groups):
    return float(np.average(np.abs(prediction-truth), weights=player_weights(groups)))


def inner_choices(matrices, y, groups):
    players = np.unique(groups)
    folds = np.array_split(players, min(5, len(players)))
    if len(players) < 2:
        raise ValueError('too_few_inner_players')
    scores = []
    for family, x in matrices.items():
        alphas = ALPHAS if family == 'original12' else (*ALPHAS, 1000.)
        for alpha in alphas:
            errors = []
            for fold in folds:
                test = np.isin(groups, fold)
                model = fit_model(x[~test], y[~test], groups[~test], alpha, 'mae' if family == 'mae14' else 'ridge')
                prediction = demean(predict_ridge(model, x[test]), groups[test])
                for player in fold:
                    selected = groups[test] == player
                    errors.append(float(np.mean(np.abs(prediction[selected]-y[test][selected]))))
            scores.append({'family': family, 'alpha': alpha, 'inner_mae': float(np.mean(errors))})
    scores.append({'family': 'null', 'alpha': 0., 'inner_mae': balanced_mae(np.zeros(len(y)), y, groups)})
    # Stable deterministic tie handling, without looking at outer outcomes.
    best = min(scores, key=lambda a: (round(a['inner_mae'], 12), PRIORITY[a['family']], -a['alpha']))
    choices = {family: min([r for r in scores if r['family'] == family], key=lambda a: (round(a['inner_mae'],12), -a['alpha'])) for family in [*matrices, 'null']}
    return choices, best, scores


def compare_models(frame):
    ordered = frame.sort_values(['user','session_pitch'])
    groups = ordered.user.to_numpy(str)
    raw_y = ordered.pitch_speed_kmh.to_numpy(float)
    y = demean(raw_y, groups)
    matrices = {name: demean(ordered[columns].to_numpy(float), groups) for name, columns in FAMILIES.items()}
    rows, diagnostics, all_scores = [], [], []
    for held in np.unique(groups):
        test = groups == held
        choices, best, scores = inner_choices({k:v[~test] for k,v in matrices.items()}, y[~test], groups[~test])
        all_scores.extend(dict(s, held_player=held) for s in scores)
        predictions = {}
        for family, choice in choices.items():
            if family == 'null':
                prediction = np.zeros(test.sum())
                train = np.zeros((~test).sum())
                coefficients = []
            else:
                x = matrices[family]
                model = fit_model(x[~test], y[~test], groups[~test], choice['alpha'], 'mae' if family == 'mae14' else 'ridge')
                prediction = demean(predict_ridge(model, x[test]), groups[test])
                train = demean(predict_ridge(model, x[~test]), groups[~test])
                coefficients = model[3].tolist()
            predictions[family] = prediction
            diagnostics.append({'held_player': held, 'family': family, **choice,
                                'train_mae': balanced_mae(train, y[~test], groups[~test]),
                                'test_mae': balanced_mae(prediction, y[test], groups[test]),
                                'coefficients': coefficients, 'held_player_in_training': bool(held in groups[~test])})
        predictions['selector'] = predictions[best['family']]
        for family, prediction in predictions.items():
            for key, truth, pred, raw in zip(ordered.session_pitch[test], y[test], prediction, raw_y[test]):
                rows.append({'user': held, 'session_pitch': key, 'family': family, 'truth': float(truth),
                             'prediction': float(pred), 'baseline': 0., 'observed_speed_kmh': float(raw),
                             'selected_family': best['family'] if family == 'selector' else family})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics), pd.DataFrame(all_scores)


def permutation_controls(frame, repetitions=20):
    """Training-only shuffles; fixed alpha is a negative control, not model selection."""
    ordered = frame.sort_values(['user','session_pitch'])
    groups = ordered.user.to_numpy(str)
    x = demean(ordered[EXTENDED].to_numpy(float), groups)
    y = demean(ordered.pitch_speed_kmh.to_numpy(float), groups)
    records = []
    for repetition in range(-1, repetitions):
        rng = np.random.default_rng(20261029+max(repetition,0))
        errors = []
        for held in np.unique(groups):
            test = groups == held
            shuffled = y[~test].copy()
            if repetition >= 0:
                for player in np.unique(groups[~test]):
                    mask = groups[~test] == player
                    shuffled[mask] = rng.permutation(shuffled[mask])
            model = fit_ridge(x[~test], shuffled, groups[~test], 10.)
            prediction = demean(predict_ridge(model, x[test]), groups[test])
            errors.append(balanced_mae(prediction, y[test], groups[test]))
        records.append({'repetition': repetition, 'training_labels_shuffled': repetition >= 0,
                        'mae_kmh': float(np.mean(errors))})
    return pd.DataFrame(records)
