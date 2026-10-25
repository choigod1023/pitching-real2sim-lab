"""Observed-pitch baseline; predictive association is not an intervention effect."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pilot import regular_time, vector, speed_at

FEATURES = ['hand_release_kmh', 'wrist_release_kmh', 'shoulder_release_kmh',
            'session_height_m', 'session_mass_kg']


def extract(metadata, landmarks):
    if metadata.duplicated(['user', 'session_pitch']).any() or metadata.session_pitch.duplicated().any():
        raise ValueError('Pitch identity must be unique')
    groups = {key: g.sort_values('time') for key, g in landmarks.groupby('session_pitch')}
    rows = []
    for m in metadata.itertuples(index=False):
        g = groups[m.session_pitch]
        t = regular_time(g)
        release = float(g.BR_time.iloc[0])
        if not np.isfinite(release) or not (t[1] < release < t[-2]):
            raise ValueError(f'Invalid release window: {m.session_pitch}')
        row = {'user': m.user, 'session': m.session, 'session_pitch': m.session_pitch,
               'observed_ball_kmh': m.pitch_speed_mph * 1.609344,
               'session_height_m': m.session_height_m, 'session_mass_kg': m.session_mass_kg}
        for part in ['hand', 'wrist', 'shoulder']:
            p = vector(g, part + '_jc')
            if not np.isfinite(p).all():
                raise ValueError(f'Nonfinite trajectory: {m.session_pitch}')
            row[part + '_release_kmh'] = speed_at(t, p, release) * 3.6
        row['hand_release_minus_frame_kmh'] = speed_at(t, vector(g, 'hand_jc'), release-1/360) * 3.6
        row['hand_release_plus_frame_kmh'] = speed_at(t, vector(g, 'hand_jc'), release+1/360) * 3.6
        rows.append(row)
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all() or (result.observed_ball_kmh <= 0).any():
        raise ValueError('Invalid pitch-level measurements')
    return result


def ridge_predict(train_x, train_y, test_x, penalty=1.):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-12, 1., scale)
    x = (train_x-mean)/scale
    centered_y = train_y-train_y.mean()
    beta = np.linalg.solve(x.T@x + penalty*np.eye(x.shape[1]), x.T@centered_y)
    return train_y.mean() + (test_x-mean)/scale@beta


def evaluate(features):
    # Athlete-disjoint leave-one-athlete-out evaluation; no tuned hyperparameters.
    if features.user.nunique() < 3:
        raise ValueError('At least three athletes required')
    outputs = []
    for uid in sorted(features.user.unique()):
        train = features[features.user != uid]
        test = features[features.user == uid]
        predictions = {
            'training_mean': np.full(len(test), train.observed_ball_kmh.mean()),
            'hand_equals_ball': test.hand_release_kmh.to_numpy(),
            'hand_ridge': ridge_predict(train[['hand_release_kmh']].to_numpy(), train.observed_ball_kmh.to_numpy(), test[['hand_release_kmh']].to_numpy()),
            'kinematic_ridge': ridge_predict(train[FEATURES].to_numpy(), train.observed_ball_kmh.to_numpy(), test[FEATURES].to_numpy()),
        }
        for name, prediction in predictions.items():
            out = test[['user', 'session_pitch', 'observed_ball_kmh']].copy()
            out['method'] = name
            out['predicted_ball_kmh'] = prediction
            outputs.append(out)
    result = pd.concat(outputs, ignore_index=True)
    summary = {}
    for method, group in result.groupby('method'):
        error = group.predicted_ball_kmh-group.observed_ball_kmh
        per = group.assign(abs_error=error.abs(), squared_error=error**2, error=error).groupby('user')[['abs_error','squared_error','error']].mean()
        summary[method] = {'athlete_balanced_mae_kmh': float(per.abs_error.mean()),
                           'athlete_balanced_rmse_kmh': float(np.sqrt(per.squared_error.mean())),
                           'athlete_balanced_bias_kmh': float(per.error.mean())}
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--output', type=Path, default=Path('artifacts/release-baseline'))
    args = parser.parse_args()
    features = extract(pd.read_csv(args.data_root/'metadata.csv'), pd.read_csv(args.data_root/'full_sig/landmarks.csv'))
    predictions, summary = evaluate(features)
    report = {'stage': 'observational_release_baseline', 'pitches': len(features),
              'athletes': int(features.user.nunique()), 'split': 'leave_one_athlete_out',
              'ridge_penalty': 1., 'summary': summary,
              'counterfactual_validated': False, 'supports_optimization': False,
              'next_gate': 'Validate force-driven arm motion and explicit ball release; never optimize this observational regressor as causal physics.'}
    args.output.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output/'features.csv', index=False)
    predictions.to_csv(args.output/'predictions.csv', index=False)
    (args.output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
