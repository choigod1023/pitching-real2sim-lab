"""Outcome-free conditional kinematics and grouped Monte Carlo propagation."""
from pathlib import Path
import hashlib

import ezc3d
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, RotationSpline

from ball_release import hand_frame
from pilot import regular_time
from speed_path import retarget_velocity, rotation_velocity

PITCHES = ('3034_1', '3034_2', '3034_3', '3034_4', '3034_5')
CUTOFFS = (10, 20, 30)
GROUPS = ((0, 1), (2,), (3,), (4, 5, 6))
FACTOR_NAMES = ('distance_and_relative_motion', 'release_frame', 'hand_orientation_filter', 'relative_local_offset')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_case(data, pitch):
    """Read identifiers, centers and raw markers; radar is not a model input."""
    data = Path(data)
    metadata = pd.read_csv(data/'metadata.csv', usecols=['session_pitch', 'filename_new'])
    rows = metadata[metadata.session_pitch.eq(pitch)]
    if len(rows) != 1:
        raise ValueError('nonunique_pitch_identity')
    paths = list((data/'c3d').rglob(rows.iloc[0].filename_new))
    if len(paths) != 1:
        raise ValueError('nonunique_c3d')
    c3d = ezc3d.c3d(str(paths[0]))
    labels = [label.strip() for label in c3d['parameters']['POINT']['LABELS']['value']]
    rate = float(c3d['parameters']['POINT']['RATE']['value'][0])
    units = c3d['parameters']['POINT']['UNITS']['value'][0].strip().lower()
    if rate != 360. or units not in ('m', 'mm', 'cm'):
        raise ValueError('unsupported_rate_or_units')
    scale = {'m': 1., 'cm': .01, 'mm': .001}[units]
    raw = []
    for marker in ('RFIN', 'RWRA', 'RWRB'):
        values = c3d['data']['points'][:, labels.index(marker), :].T
        if not np.isfinite(values).all() or np.any(values[:, 3] < 0):
            raise ValueError('invalid_marker_no_gap_imputation')
        raw.append(values[:, :3]*scale)
    columns = [name+'_jc_'+axis for name in ('shoulder', 'elbow', 'wrist') for axis in 'xyz']
    landmarks = pd.read_csv(data/'full_sig/landmarks.csv', usecols=['session_pitch', 'time', 'fp_10_time', 'BR_time', *columns])
    frame = landmarks[landmarks.session_pitch.eq(pitch)].sort_values('time')
    time = regular_time(frame)
    if len(time) != len(raw[0]) or len(time) < 40 or np.any(np.diff(time) <= 0):
        raise ValueError('source_time_mismatch')
    if not np.allclose(time, np.arange(len(time))/rate, atol=.000051, rtol=0):
        raise ValueError('raw_processed_time_alignment')
    centers = frame[columns].to_numpy(float).reshape(len(frame), 3, 3)
    if not np.isfinite(centers).all():
        raise ValueError('invalid_joint_centers')
    for event in ('fp_10_time', 'BR_time'):
        if not np.isfinite(frame[event]).all() or frame[event].nunique() != 1:
            raise ValueError('invalid_event')
    fp, br = float(frame.fp_10_time.iloc[0]), float(frame.BR_time.iloc[0])
    if not time[1] < fp < br-1/rate < br+1/rate < time[-2]:
        raise ValueError('event_outside_support')
    curve = CubicSpline(time, centers)
    lengths = np.median(np.linalg.norm(np.diff(curve(np.linspace(fp, br, 100)), axis=1), axis=2), axis=0)
    if np.any(lengths < .01):
        raise ValueError('degenerate_arm')
    evaluation_times = br + np.array([-1, 0, 1])/rate
    wrist_velocity = np.array([retarget_velocity(curve(t), curve(t, 1), lengths) for t in evaluation_times])
    rotations, derivatives, legacy = [], [], []
    for cutoff in CUTOFFS:
        filtered = [sosfiltfilt(butter(4, cutoff, fs=rate, output='sos'), values, axis=0) for values in raw]
        _, matrices = hand_frame(*filtered)
        rotation = RotationSpline(time, Rotation.from_matrix(matrices))
        rotations.append(rotation(evaluation_times).as_matrix())
        epsilon = 1e-5
        derivatives.append((rotation(evaluation_times+epsilon).as_matrix()-rotation(evaluation_times-epsilon).as_matrix())/(2*epsilon))
        legacy.append(float(np.linalg.norm(wrist_velocity[1] + rotation_velocity(rotation, br, np.array([.15, 0, 0])))*3.6))
    case = {'pitch': pitch, 'rate_hz': rate, 'wrist_velocity': wrist_velocity,
            'rotation': np.array(rotations), 'rotation_derivative': np.array(derivatives),
            'sources': {str(paths[0].resolve()): file_hash(paths[0]),
                        str((data/'full_sig/landmarks.csv').resolve()): file_hash(data/'full_sig/landmarks.csv'),
                        str((data/'metadata.csv').resolve()): file_hash(data/'metadata.csv')},
            'legacy_nominal_kmh': legacy, 'lengths_m': lengths.tolist(), 'br_s': br}
    nominal = np.full((1, 7), .5)
    error = abs(float(speeds(case, nominal)[0])-legacy[1])
    if error > 1e-7:
        raise AssertionError('legacy_15cm_regression')
    case['legacy_regression_error_kmh'] = error
    return case


def velocities(case, unit_samples, offset_radius_m=0., relative_motion=True):
    u = np.asarray(unit_samples, float)
    if u.ndim != 2 or u.shape[1] != 7 or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('unit_samples_must_be_N_by_7')
    if not np.isfinite(offset_radius_m) or offset_radius_m < 0:
        raise ValueError('invalid_offset_radius')
    time_index = np.minimum((u[:, 2]*3).astype(int), 2)
    filter_index = np.minimum((u[:, 3]*3).astype(int), 2)
    distance = .1 + .1*u[:, 0]
    distance_rate = (u[:, 1]-.5) if relative_motion else np.zeros(len(u))
    distance += distance_rate*(time_index-1)/case['rate_hz']
    if np.any((distance <= 0) | (distance < .0986) | (distance > .2014)):
        raise ValueError('distance_outside_local_protocol')
    cos_polar = 2*u[:, 5]-1
    azimuth = 2*np.pi*u[:, 6]
    sin_polar = np.sqrt(np.maximum(1-cos_polar**2, 0))
    offset = offset_radius_m*u[:, 4, None]*np.column_stack([sin_polar*np.cos(azimuth), sin_polar*np.sin(azimuth), cos_polar])
    offset[:, 0] += distance
    rotation = case['rotation'][filter_index, time_index]
    derivative = case['rotation_derivative'][filter_index, time_index]
    return (case['wrist_velocity'][time_index] + np.einsum('nij,nj->ni', derivative, offset)
            + rotation[:, :, 0]*distance_rate[:, None])


def speeds(case, unit_samples, offset_radius_m=0., relative_motion=True):
    return np.linalg.norm(velocities(case, unit_samples, offset_radius_m, relative_motion), axis=1)*3.6


def rigid_transform_velocity(velocity, rotation, translation):
    """A common fixed translation has zero derivative; rotation preserves speed."""
    velocity, rotation, translation = map(np.asarray, (velocity, rotation, translation))
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(rotation), 1.):
        raise ValueError('need_proper_fixed_rigid_transform')
    return velocity @ rotation.T


def grouped_mc(model, n, seed, groups=GROUPS):
    """Saltelli first/Jansen total indices; no clipping of sampling variation."""
    if n < 16 or sorted(i for group in groups for i in group) != list(range(7)):
        raise ValueError('invalid_design')
    generator = np.random.default_rng(seed)
    a, b = generator.random((2, n, 7))
    ya, yb = model(a), model(b)
    y = np.r_[ya, yb]
    if not np.isfinite(y).all():
        raise ValueError('nonfinite_model_output')
    variance = float(np.var(y, ddof=1))
    if variance < 1e-15:
        raise ValueError('degenerate_output_variance')
    first, total = [], []
    for group in groups:
        c = a.copy()
        c[:, group] = b[:, group]
        yc = model(c)
        first.append(float(np.mean((yb-y.mean())*(yc-ya))/variance))
        total.append(float(np.mean((ya-yc)**2)/(2*variance)))
    return {'n_per_matrix': n, 'seed': seed, 'mean_kmh': float(y.mean()),
            'sd_kmh': float(y.std(ddof=1)), 'variance_kmh2': variance,
            'scenario_quantiles_025_50_975_kmh': np.quantile(y, [.025, .5, .975]).tolist(),
            'first_order': dict(zip(FACTOR_NAMES, first)), 'total_order': dict(zip(FACTOR_NAMES, total)),
            'interaction_residual': float(1-sum(first))}


def convergence(results):
    """Compare both seeds and both sample counts against fixed thresholds."""
    pairs = []
    for i, a in enumerate(results):
        for b in results[i+1:]:
            means = abs(a['mean_kmh']-b['mean_kmh'])
            quantiles = float(np.max(np.abs(np.array(a['scenario_quantiles_025_50_975_kmh'])[[0,2]]-np.array(b['scenario_quantiles_025_50_975_kmh'])[[0,2]])))
            indices = max(abs(a[k][factor]-b[k][factor]) for k in ('first_order', 'total_order') for factor in FACTOR_NAMES)
            pairs.append({'a': [a['n_per_matrix'], a['seed']], 'b': [b['n_per_matrix'], b['seed']],
                          'mean_difference_kmh': means, 'max_quantile_difference_kmh': quantiles,
                          'max_index_difference': indices,
                          'pass': bool(means < .1 and quantiles < .25 and indices < .03)})
    return {'verdict': 'PASS' if all(p['pass'] for p in pairs) else 'FAIL', 'comparisons': pairs}


def recorded_comparison(data, summaries):
    """Outcome read happens only after all scenario outputs have been frozen."""
    source = pd.read_csv(Path(data)/'metadata.csv', usecols=['session_pitch', 'pitch_speed_mph']).set_index('session_pitch')
    return [{'pitch': row['pitch'], 'recorded_speed_kmh': float(source.loc[row['pitch'], 'pitch_speed_mph'])*1.609344,
             'nominal_difference_kmh': row['nominal_15cm_kmh']-float(source.loc[row['pitch'], 'pitch_speed_mph'])*1.609344,
             'scenario_mean_difference_kmh': row['primary']['mean_kmh']-float(source.loc[row['pitch'], 'pitch_speed_mph'])*1.609344}
            for row in summaries]
