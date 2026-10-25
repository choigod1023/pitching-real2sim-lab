"""Train-only rigid-offset calibration diagnostic, not identified ball geometry.

Leave-one-athlete-out fits a shared offset from speed magnitude alone. The fixed
component box +/-0.10 m is a numerical constraint, NOT an anatomical bound.
Neither fitted offsets nor counterfactual gains are physically validated here.
"""
import argparse
import json
import time
from pathlib import Path
import ezc3d
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt
from ball_release import hand_frame, rigid_offset_velocity


def predict_speed(v, derivative, offset):
    return np.linalg.norm(v+np.einsum('nij,j->ni', derivative, offset), axis=1)*3.6


def fit_offset(v, derivative, observed, users, bound=.10):
    """Fit ONLY supplied training rows with equal athlete total squared weight."""
    if bound <= 0 or not np.isfinite(bound):
        raise ValueError('Invalid numerical bound')
    _, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    weights = 1/np.sqrt(counts[inverse])
    def residual(offset):
        return (predict_speed(v, derivative, offset)-observed)*weights
    starts = [np.zeros(3)]+[sign*axis*bound*.25 for axis in np.eye(3) for sign in [-1, 1]]
    fits = [least_squares(residual, start, bounds=(-bound, bound), max_nfev=500,
                         ftol=1e-10, xtol=1e-10, gtol=1e-10) for start in starts]
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.cost)]
    if not successful:
        raise RuntimeError('No multistart fit converged')
    best = min(successful, key=lambda fit: fit.cost)
    return best.x


def evaluate_heldout(v, derivative, observed, users, heldout, bound=.10):
    train = users != heldout
    test = ~train
    if not train.any() or not test.any():
        raise ValueError('Need nonempty disjoint train/test athletes')
    offset = fit_offset(v[train], derivative[train], observed[train], users[train], bound)
    means = [observed[train & (users == user)].mean() for user in np.unique(users[train])]
    zero = predict_speed(v, derivative, np.zeros(3))
    design = np.column_stack([np.ones(train.sum()), zero[train]])
    _, inverse, counts = np.unique(users[train], return_inverse=True, return_counts=True)
    weights = 1/np.sqrt(counts[inverse])
    linear_coefficients = np.linalg.lstsq(design*weights[:, None], observed[train]*weights, rcond=None)[0]
    scalar_linear = linear_coefficients[0]+linear_coefficients[1]*zero[test]
    return dict(offset=offset, predicted=predict_speed(v[test], derivative[test], offset),
                zero=zero[test], scalar_linear=scalar_linear, linear_coefficients=linear_coefficients,
                train_mean=np.full(test.sum(), np.mean(means)))


def extract_features(data):
    metadata = pd.read_csv(data/'metadata.csv')
    poi = pd.read_csv(data/'poi/poi_metrics.csv', usecols=['session_pitch', 'p_throws'])
    metadata = metadata.merge(poi, on='session_pitch', validate='one_to_one')
    eligible = metadata[metadata.p_throws.eq('R')].sort_values('session_pitch')
    event_data = pd.read_csv(data/'full_sig/landmarks.csv', usecols=['session_pitch', 'time', 'BR_time'])
    events_by_pitch = {key: group.sort_values('time') for key, group in event_data.groupby('session_pitch')}
    paths = {}
    for path in (data/'c3d').rglob('*.c3d'):
        paths.setdefault(path.name, []).append(path)
    features, excluded = [], []
    for row in eligible.itertuples():
        try:
            matches = paths.get(row.filename_new, [])
            if len(matches) != 1:
                raise ValueError('Expected exactly one C3D')
            c = ezc3d.c3d(str(matches[0]))
            units = c['parameters']['POINT']['UNITS']['value'][0].strip().lower()
            scale = {'m': 1., 'cm': .01, 'mm': .001}[units]
            rate = float(c['parameters']['POINT']['RATE']['value'][0])
            if not np.isclose(rate, c['header']['points']['frame_rate']):
                raise ValueError('Header/rate mismatch')
            labels = [label.strip() for label in c['parameters']['POINT']['LABELS']['value']]
            markers = []
            for label in ['RFIN', 'RWRA', 'RWRB']:
                points = c['data']['points'][:, labels.index(label), :].T
                if not np.isfinite(points[:, :3]).all() or (points[:, 3] < 0).any():
                    raise ValueError('Marker gap/nonfinite sample')
                markers.append(points[:, :3]*scale)
            events = events_by_pitch[row.session_pitch]
            expected = np.arange(len(markers[0]))/rate
            if len(events) != len(expected) or not np.allclose(events.time, expected, atol=.000051, rtol=0):
                raise ValueError('C3D/full-signal time mismatch')
            if events.BR_time.nunique() != 1 or not np.isfinite(events.BR_time.iloc[0]):
                raise ValueError('Ambiguous release time')
            frame = int(np.rint(events.BR_time.iloc[0]*rate))
            if frame < 1 or frame >= len(expected)-1:
                raise ValueError('Release outside interior samples')
            if not np.isfinite(row.pitch_speed_mph):
                raise ValueError('Missing observed speed')
            sos = butter(4, 20, fs=rate, output='sos')
            pitch_features = []
            for mode, inputs in [('raw', markers), ('lowpass20', [sosfiltfilt(sos, x, axis=0) for x in markers])]:
                origin, rotation = hand_frame(*inputs)
                _, velocity = rigid_offset_velocity(origin, rotation, np.zeros(3), rate)
                derivative = np.gradient(rotation, 1/rate, axis=0, edge_order=2)[frame]
                feature = dict(user=int(row.user), session_pitch=row.session_pitch, mode=mode,
                               observed_kmh=float(row.pitch_speed_mph)*1.609344)
                feature.update({f'v{i}': float(x) for i, x in enumerate(velocity[frame])})
                feature.update({f'd{i}{j}': float(derivative[i,j]) for i in range(3) for j in range(3)})
                pitch_features.append(feature)
            features.extend(pitch_features)
        except (ValueError, KeyError, RuntimeError, OSError) as error:
            excluded.append(dict(session_pitch=row.session_pitch, user=int(row.user), reason=str(error)))
    return pd.DataFrame(features), excluded, len(eligible)


def run(data, out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    features, excluded, eligible = extract_features(data)
    features.to_csv(out/'features.csv', index=False)
    if features.empty:
        raise ValueError('No eligible features')
    rows, folds, summaries = [], [], {}
    for mode, group in features.groupby('mode'):
        group = group.reset_index(drop=True)
        v = group[[f'v{i}' for i in range(3)]].to_numpy()
        derivative = group[[f'd{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
        observed = group.observed_kmh.to_numpy()
        users = group.user.to_numpy()
        mode_rows, mode_folds = [], []
        for heldout in np.unique(users):
            result = evaluate_heldout(v, derivative, observed, users, heldout)
            offset = result['offset']
            fold = dict(mode=mode, heldout_user=int(heldout), offset_x_m=offset[0], offset_y_m=offset[1], offset_z_m=offset[2],
                        bound_hit=bool((np.abs(offset) >= .10-1e-5).any()),
                        linear_intercept_kmh=float(result['linear_coefficients'][0]),
                        linear_slope=float(result['linear_coefficients'][1]))
            mode_folds.append(fold)
            for index, source in enumerate(group[users == heldout].itertuples()):
                record = dict(mode=mode, user=int(heldout), session_pitch=source.session_pitch, observed_kmh=source.observed_kmh)
                for method in ['predicted', 'zero', 'train_mean', 'scalar_linear']:
                    record[method+'_kmh'] = float(result[method][index])
                    record[method+'_absolute_error_kmh'] = abs(record[method+'_kmh']-record['observed_kmh'])
                mode_rows.append(record)
        table = pd.DataFrame(mode_rows)
        offset_values = np.array([[f[f'offset_{axis}_m'] for axis in 'xyz'] for f in mode_folds])
        summaries[mode] = dict(athletes=len(mode_folds), pitches=len(table),
            athlete_balanced_mae_kmh={method: float(table.groupby('user')[method+'_absolute_error_kmh'].mean().mean()) for method in ['predicted','zero','train_mean','scalar_linear']},
            bound_hit_fraction=float(np.mean([f['bound_hit'] for f in mode_folds])),
            offset_mean_m=offset_values.mean(axis=0).tolist(), offset_std_m=offset_values.std(axis=0).tolist(),
            offset_min_m=offset_values.min(axis=0).tolist(), offset_max_m=offset_values.max(axis=0).tolist())
        rows.extend(mode_rows); folds.extend(mode_folds)
    pd.DataFrame(rows).to_csv(out/'heldout.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'folds.csv', index=False)
    report = dict(eligible_right_handed_pitches=eligible, included_pitches=int(features.session_pitch.nunique()),
        exclusions=excluded, elapsed_seconds=time.perf_counter()-start, results=summaries,
        validation='Leave one athlete out; offset and athlete-balanced mean fitted only on other athletes; no release-time fitting',
        objective='Athlete-balanced squared speed error; seven deterministic starts', numerical_box_m=[-.10,.10],
        scalar_linear_baseline='Train-only affine fit of observed speed against zero-offset marker speed, athlete-balanced weighted least squares, no clipping',
        geometry_identified=False, predicted_gain_kmh=None, coaching_ready=False,
        limitations=['Offset box is numerical, not anatomical plausibility bounds',
                     'Speed magnitude alone does not identify true grip geometry',
                     'Fitted offsets may compensate marker, timing, filtering and model errors',
                     'Raw and20Hz are separate prespecified diagnostics, not selected by heldout performance',
                     'No ball contact, finger mechanics, spin or force-driven intervention prediction'])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root.parent/'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--out', type=Path, default=root/'artifacts/release-calibration')
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.out), indent=2))
