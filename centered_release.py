"""Exploratory train-only within-athlete release calibration, not external validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from release_calibration import predict_speed


def center_by_athlete(values, users):
    values = np.asarray(values, dtype=float)
    users = np.asarray(users)
    centered = values.copy()
    for user in np.unique(users):
        mask = users == user
        centered[mask] -= values[mask].mean()
    return centered


def fit_centered(v, derivative, observed, users, bound=.10):
    """Training arrays only; deterministic seven-start constrained fit."""
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError('bound must be finite and positive')
    _, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(counts[inverse])
    target = center_by_athlete(observed, users)
    zero = predict_speed(v, derivative, np.zeros(3))
    predictor = center_by_athlete(zero, users)
    denominator = np.sum(weights**2 * predictor**2)
    slope = float(np.sum(weights**2 * predictor * target) / denominator) if denominator > 1e-12 else 0.

    def residual(offset):
        return weights * (center_by_athlete(predict_speed(v, derivative, offset), users) - target)

    starts = [np.zeros(3)] + [sign * axis * bound * .25 for axis in np.eye(3) for sign in [-1, 1]]
    fits = [least_squares(residual, start, bounds=(-bound, bound), max_nfev=500,
                          ftol=1e-10, xtol=1e-10, gtol=1e-10) for start in starts]
    fits = [fit for fit in fits if fit.success and np.isfinite(fit.cost)]
    if not fits:
        raise RuntimeError('No centered fit converged')
    best = min(fits, key=lambda fit: fit.cost)
    return best.x, slope


def evaluate_fold(v, derivative, observed, users, heldout, bound=.10):
    train = users != heldout
    test = ~train
    if not train.any() or not test.any():
        raise ValueError('Need disjoint nonempty training and evaluation athletes')
    offset, slope = fit_centered(v[train], derivative[train], observed[train], users[train], bound)
    zero = predict_speed(v[test], derivative[test], np.zeros(3))
    return dict(offset=offset, slope=slope,
                centered_offset=predict_speed(v[test], derivative[test], offset),
                centered_scalar=slope * zero)


def correlation(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(features_path, out, bound=.10):
    features_path, out = Path(features_path), Path(out)
    source = features_path.read_bytes()
    data = pd.read_csv(features_path)
    if data.empty or data.duplicated(['mode', 'session_pitch']).any():
        raise ValueError('Empty or duplicated feature rows')
    rows, folds, diagnostics = [], [], {}
    for mode, group in data.groupby('mode'):
        group = group.sort_values(['user', 'session_pitch']).reset_index(drop=True)
        v = group[[f'v{i}' for i in range(3)]].to_numpy()
        derivative = group[[f'd{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
        observed, users = group.observed_kmh.to_numpy(), group.user.to_numpy()
        if not all(np.isfinite(array).all() for array in [v, derivative, observed]):
            raise ValueError('Nonfinite input')
        zero = predict_speed(v, derivative, np.zeros(3))
        centered_y = center_by_athlete(observed, users)
        centered_x = center_by_athlete(zero, users)
        means = pd.DataFrame(dict(user=users, observed=observed, zero=zero)).groupby('user').mean()
        diagnostics[mode] = dict(within_correlation=correlation(centered_x, centered_y),
                                 between_correlation=correlation(means.zero, means.observed),
                                 centered_observed_sd_kmh=float(np.std(centered_y)),
                                 centered_marker_sd_kmh=float(np.std(centered_x)))
        for heldout in np.unique(users):
            result = evaluate_fold(v, derivative, observed, users, heldout, bound)
            offset = result['offset']
            folds.append(dict(mode=mode, heldout_user=int(heldout),
                              **{f'offset_{axis}_m': float(value) for axis, value in zip('xyz', offset)},
                              scalar_slope=result['slope'],
                              bound_hit=bool((np.abs(offset) >= bound - 1e-5).any())))
            selected = group[users == heldout]
            if len(selected) < 2:
                continue
            actual = selected.observed_kmh.to_numpy()
            pitches = selected.session_pitch.to_numpy()
            for method in ['centered_offset', 'centered_scalar', 'no_change']:
                predicted = np.zeros(len(actual)) if method == 'no_change' else result[method]
                for index in range(1, len(actual)):
                    delta = float(actual[index] - actual[0])
                    pred_delta = float(predicted[index] - predicted[0])
                    rows.append(dict(mode=mode, user=int(heldout), method=method,
                                     reference_pitch=pitches[0], evaluated_pitch=pitches[index],
                                     observed_delta_kmh=delta, predicted_delta_kmh=pred_delta,
                                     absolute_delta_error_kmh=abs(delta - pred_delta)))
    changes = pd.DataFrame(rows)
    if changes.empty:
        raise ValueError('Need repeated pitches for change evaluation')
    summary = [dict(mode=mode, method=method, athletes=int(group.user.nunique()),
                    evaluation_pitches=len(group),
                    athlete_balanced_delta_mae_kmh=float(group.groupby('user').absolute_delta_error_kmh.mean().mean()))
               for (mode, method), group in changes.groupby(['mode', 'method'])]
    report = dict(input_path=str(features_path.resolve()), input_sha256=hashlib.sha256(source).hexdigest(),
                  scope='Exploratory development using an already-inspected cohort; not external validation or intervention evidence.',
                  fitting='Leave one athlete out; centered outcome/prediction residuals and scalar slope fitted only on training athletes; equal athlete squared weight.',
                  reference='First lexicographic session_pitch, not asserted chronological.',
                  optimization='Seven deterministic starts; no random seed or stochastic optimization.',
                  numerical_bound_m=bound, summary=summary, diagnostics=diagnostics,
                  geometry_identified=False, intervention_validated=False, coaching_ready=False,
                  limitations=['Centered optimization discards between-athlete absolute speed information.',
                               'Offset is a numerical compensator, not identified grip geometry.',
                               'Raw and 20Hz reported separately; no heldout-selected processing.',
                               'Observed same-session differences do not establish effects of a prescribed change.'])
    out.mkdir(parents=True, exist_ok=True)
    changes.to_csv(out/'heldout_deltas.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'offsets.csv', index=False)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=root/'artifacts/release-calibration/features.csv')
    parser.add_argument('--out', type=Path, default=root/'artifacts/centered-release')
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.out), indent=2))
