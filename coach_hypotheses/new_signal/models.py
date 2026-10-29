"""Train-only nonlinear pitch maps contrasted within athlete, nested by athlete."""
import numpy as np
import pandas as pd

from ..research_reset.validation import FEATURES, pair_summary
from .signals import NEW_FEATURES

FAMILIES = ('null', 'stable7', 'new48_linear', 'new48_rff')


def pair_data(frame, mapped):
    frame = frame.reset_index(drop=True)
    if frame.session_pitch.duplicated().any():
        raise ValueError('duplicate_pitch')
    first, second = [], []
    for _, group in frame.groupby('user', sort=True):
        indices = group.sort_values('session_pitch').index.to_numpy()
        if len(indices) < 2:
            raise ValueError('insufficient_repeats')
        i, j = np.triu_indices(len(indices), 1)
        first.extend(indices[i]); second.extend(indices[j])
    a, b = np.array(first, int), np.array(second, int)
    if not len(a):
        raise ValueError('empty_pairs')
    result = pd.DataFrame(dict(user=frame.user.to_numpy()[a],
                               pitch_i=frame.session_pitch.to_numpy()[a],
                               pitch_j=frame.session_pitch.to_numpy()[b],
                               truth_delta_kmh=frame.pitch_speed_kmh.to_numpy()[b]-frame.pitch_speed_kmh.to_numpy()[a]))
    return result, np.asarray(mapped)[b]-np.asarray(mapped)[a]


def pitch_map(train, test, family):
    if family not in FAMILIES:
        raise ValueError('unknown_family')
    names = FEATURES if family in ('stable7', 'null') else NEW_FEATURES
    x, z = train[names].to_numpy(float), test[names].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('nonfinite_map_input')
    if family != 'new48_rff':
        return x, z, {'mapping': 'identity'}
    weights = 1./train.groupby('user').user.transform('size').to_numpy(float)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[(scale == 0) | (np.ptp(x, axis=0) == 0)] = 1.
    rng = np.random.default_rng(20261029)
    matrix = rng.normal(size=(len(names), 64))/np.sqrt(len(names))
    phase = rng.uniform(0, 2*np.pi, 64)
    return (np.sqrt(2/64)*np.cos((x-mean)/scale@matrix+phase),
            np.sqrt(2/64)*np.cos((z-mean)/scale@matrix+phase),
            dict(mapping='train_only_rff', mean=mean, scale=scale))


def fit_predict(train, test, family):
    mapped, held, mapping = pitch_map(train, test, family)
    target, dx = pair_data(train, mapped)
    records, dz = pair_data(test, held)
    if family == 'null':
        records['predicted_pair_delta_kmh'] = 0.
        return records, dict(mapping=mapping)
    weights = 1./target.groupby('user').user.transform('size').to_numpy(float)
    scale = np.sqrt(np.average(dx*dx, axis=0, weights=weights))
    scale[scale == 0] = 1.
    x = dx/scale
    beta = np.linalg.solve(x.T@(weights[:, None]*x)+10*np.eye(x.shape[1]),
                           x.T@(weights*target.truth_delta_kmh.to_numpy()))
    records['predicted_pair_delta_kmh'] = dz/scale@beta
    return records, dict(mapping=mapping, contrast_scale=scale, beta=beta)


def choose_family(train):
    players = sorted(train.user.unique())
    if len(players) < 2:
        raise ValueError('insufficient_players_inner')
    folds = {player: i % 5 for i, player in enumerate(players)}
    errors = {name: [] for name in FAMILIES}
    for fold in sorted(set(folds.values())):
        validation = train[train.user.map(folds) == fold]
        fitting = train[train.user.map(folds) != fold]
        for family in FAMILIES:
            predictions, _ = fit_predict(fitting, validation, family)
            predictions['error'] = abs(predictions.predicted_pair_delta_kmh-predictions.truth_delta_kmh)
            errors[family].extend(predictions.groupby('user').error.mean().tolist())
    scores = {family: float(np.mean(values)) for family, values in errors.items()}
    winner = min(FAMILIES, key=lambda name: (round(scores[name], 12), FAMILIES.index(name)))
    return winner, scores


def nested_loao(frame):
    if frame.user.nunique() < 4:
        raise ValueError('insufficient_players_outer')
    collected = {family: [] for family in FAMILIES + ('selector',)}
    splits = []
    for player in sorted(frame.user.unique()):
        train, test = frame[frame.user != player], frame[frame.user == player]
        winner, scores = choose_family(train)
        fitted = {}
        for family in FAMILIES:
            predictions, _ = fit_predict(train, test, family)
            predictions['zero_delta_baseline_kmh'] = 0.
            fitted[family] = predictions
            collected[family].append(predictions)
        collected['selector'].append(fitted[winner])
        splits.append(dict(heldout_user=player, chosen=winner, inner_mae=scores,
                           players_disjoint=player not in set(train.user)))
    predictions = {name: pd.concat(rows, ignore_index=True) for name, rows in collected.items()}
    summaries = {name: pair_summary(rows) for name, rows in predictions.items()}
    return predictions, summaries, splits
