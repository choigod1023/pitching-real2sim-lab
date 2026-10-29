"""Explicit event/axis definitions for already-filtered OBP time series."""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

RATE = 360.
G = 9.80665
CUTOFFS = (10, 20, 30)
EVENTS = ('pkh_time', 'fp_10_time', 'BR_time')
COLUMNS = {
    'joint_velos': ['pelvis_velo_z', 'torso_velo_z', 'lead_knee_velo_x'],
    'joint_angles': ['lead_knee_angle_x', 'torso_pelvis_angle_z'],
    'forces_moments': ['elbow_moment_y', 'shoulder_upper_arm_force_x',
                       'shoulder_upper_arm_force_y', 'shoulder_upper_arm_force_z',
                       'shoulder_thorax_force_z'],
    'landmarks': ['wrist_jc_z'],
}
FEATURES = ['sequence_lag_ms', 'pelvis_peak_time_br_ms', 'lead_knee_extension_deg',
            'lead_knee_peak_extension_dps', 'lead_knee_extension_at_br_dps',
            'stride_reported', 'stride_reported_sq', 'hip_shoulder_sep_fp10_deg',
            'hip_shoulder_sep_at_torso_peak_deg', 'hip_shoulder_sep_peak_deg',
            'wrist_height_br_m', 'arm_slot_poi_deg']
LOADS = ['elbow_varus_bw_height_pct', 'shoulder_resultant_peak_bw']


def checked_event(frame, column):
    values = frame[column].to_numpy(float)
    if not np.isfinite(values).all() or len(np.unique(values)) != 1:
        raise ValueError('missing_or_nonconstant_event:' + column)
    return float(values[0])


def prepare_pitch(tables):
    """Keep finite contiguous support, without interpolating missing observations."""
    reference = tables['joint_velos']
    time = reference.time.to_numpy(float)
    if len(time) < 65 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('invalid_or_duplicate_time')
    if np.max(np.abs(np.diff(time) - 1 / RATE)) > .00011:
        raise ValueError('sampling_gap_or_rate')
    if np.max(np.abs(time - time[0] - np.arange(len(time)) / RATE)) > .00011:
        raise ValueError('sampling_grid_drift')
    events = {c: checked_event(reference, c) for c in EVENTS}
    pkh, fp, br = (events[c] for c in EVENTS)
    if not 0 <= pkh < fp < br:
        raise ValueError('invalid_event_order')
    if pkh < time[0] or br > time[-1]:
        raise ValueError('event_outside_time_support')
    columns, values = [], []
    for name, needed in COLUMNS.items():
        frame = tables[name]
        if not np.array_equal(frame.time.to_numpy(float), time):
            raise ValueError('cross_table_time_mismatch:' + name)
        if any(checked_event(frame, c) != events[c] for c in EVENTS):
            raise ValueError('cross_table_event_mismatch:' + name)
        columns += needed
        values.append(frame[needed].to_numpy(float))
    matrix = np.column_stack(values)
    finite = np.isfinite(matrix).all(axis=1)
    start = int(np.searchsorted(time, pkh, side='right') - 1)
    end = int(np.searchsorted(time, br, side='left'))
    if not finite[start:end + 1].all():
        raise ValueError('nonfinite_analysis_window')
    left, right = start, end
    while left > 0 and finite[left - 1]:
        left -= 1
    while right + 1 < len(time) and finite[right + 1]:
        right += 1
    if start - left < 30 or right - end < 30:
        raise ValueError('insufficient_filter_padding')
    fp100 = None
    if 'fp_100_time' in reference and np.isfinite(reference.fp_100_time).all():
        fp100 = checked_event(reference, 'fp_100_time')
    return time[left:right + 1], matrix[left:right + 1], columns, events, fp100


def filtered_signals(matrix, columns, cutoff):
    if cutoff not in CUTOFFS or not np.isfinite(matrix).all():
        raise ValueError('invalid_filter_or_signal')
    values = np.asarray(matrix, float).copy()
    for column in COLUMNS['joint_angles']:
        i = columns.index(column)
        values[:, i] = np.rad2deg(np.unwrap(np.deg2rad(values[:, i])))
    filtered = sosfiltfilt(butter(4, cutoff, fs=RATE, output='sos'), values, axis=0)
    return {c: filtered[:, i] for i, c in enumerate(columns)}


def at_event(time, values, event):
    if not time[0] <= event <= time[-1]:
        raise ValueError('event_extrapolation')
    return float(np.interp(event, time, values))


def window_peak(time, values, begin, end):
    indices = np.flatnonzero((time >= begin) & (time <= end))
    if not len(indices):
        raise ValueError('empty_peak_window')
    i = int(indices[np.argmax(values[indices])])
    return float(time[i]), float(values[i]), bool(i in (indices[0], indices[-1]))


def pitch_metrics(prepared, meta, poi, cutoff):
    time, matrix, columns, events, fp100 = prepared
    signal = filtered_signals(matrix, columns, cutoff)
    pkh, fp, br = (events[c] for c in EVENTS)
    mass, height = float(meta['session_mass_kg']), float(meta['session_height_m'])
    if not mass > 0 or not height > 0 or not np.isfinite([mass, height]).all():
        raise ValueError('invalid_anthropometrics')
    stride, slot = float(poi['stride_length']), float(poi['arm_slot'])
    if not np.isfinite([stride, slot]).all():
        raise ValueError('nonfinite_poi_predictor')
    pt, pv, pb = window_peak(time, signal['pelvis_velo_z'], pkh, br)
    tt, tv, tb = window_peak(time, signal['torso_velo_z'], pkh, br)
    # Actual CSV uses extension positive; see AMENDMENT_01_KNEE_SIGN.md.
    extension = signal['lead_knee_velo_x']
    _, peak_knee, kb = window_peak(time, extension, fp, br)
    _, separation, sb = window_peak(time, signal['torso_pelvis_angle_z'], pkh, br)
    _, varus, _ = window_peak(time, signal['elbow_moment_y'], pkh, br)
    z = signal['shoulder_upper_arm_force_z']
    _, compression, _ = window_peak(time, z, pkh, br)
    _, distraction, _ = window_peak(time, -z, pkh, br)
    resultant = np.linalg.norm(np.column_stack([signal['shoulder_upper_arm_force_' + a] for a in 'xyz']), axis=1)
    _, peak_force, _ = window_peak(time, resultant, pkh, br)
    _, superior, _ = window_peak(time, signal['shoulder_thorax_force_z'], pkh, br)
    _, inferior, _ = window_peak(time, -signal['shoulder_thorax_force_z'], pkh, br)
    row = {'session_pitch': str(meta['session_pitch']), 'user': str(meta['user']),
           'session': str(meta['session']), 'cutoff_hz': cutoff,
           'pitch_speed_kmh': float(meta['pitch_speed_mph']) * 1.609344,
           'sequence_lag_ms': (tt - pt) * 1000,
           'pelvis_peak_time_fp10_ms': (pt - fp) * 1000,
           'torso_peak_time_fp10_ms': (tt - fp) * 1000,
           'pelvis_peak_time_br_ms': (pt - br) * 1000,
           'torso_peak_time_br_ms': (tt - br) * 1000,
           'pelvis_peak_dps': pv, 'torso_peak_dps': tv,
           'lead_knee_extension_deg': at_event(time, signal['lead_knee_angle_x'], fp) - at_event(time, signal['lead_knee_angle_x'], br),
           'lead_knee_peak_extension_dps': peak_knee,
           'lead_knee_extension_at_fp10_dps': at_event(time, extension, fp),
           'lead_knee_extension_at_br_dps': at_event(time, extension, br),
           'stride_reported': stride, 'stride_reported_sq': stride ** 2,
           'hip_shoulder_sep_fp10_deg': at_event(time, signal['torso_pelvis_angle_z'], fp),
           'hip_shoulder_sep_at_torso_peak_deg': at_event(time, signal['torso_pelvis_angle_z'], tt),
           'hip_shoulder_sep_peak_deg': separation,
           'hip_shoulder_sep_fp100_deg': at_event(time, signal['torso_pelvis_angle_z'], fp100) if fp100 is not None and time[0] <= fp100 <= time[-1] else None,
           'elbow_varus_peak_nm': varus, 'elbow_varus_bw_height_pct': 100 * varus / (mass * G * height),
           'shoulder_compression_peak_n': compression, 'shoulder_distraction_peak_n': distraction,
           'shoulder_resultant_peak_n': peak_force, 'shoulder_resultant_peak_bw': peak_force / (mass * G),
           'shoulder_thorax_superior_peak_n': superior, 'shoulder_thorax_inferior_peak_n': inferior,
           'wrist_height_br_m': at_event(time, signal['wrist_jc_z'], br),
           'actual_ball_release_height_m': None, 'arm_slot_poi_deg': slot,
           'poi_elbow_varus_nm': float(poi['elbow_varus_moment']),
           'poi_separation_fp_deg': float(poi['rotation_hip_shoulder_separation_fp']),
           'poi_separation_max_deg': float(poi['max_rotation_hip_shoulder_separation']),
           'peak_on_window_boundary': {'pelvis': pb, 'torso': tb, 'knee': kb, 'separation': sb}}
    if not np.isfinite([row[c] for c in FEATURES + LOADS + ['pitch_speed_kmh']]).all():
        raise ValueError('nonfinite_extracted_metric')
    return row


def read_cohort(data):
    meta = pd.read_csv(data / 'metadata.csv')
    poi = pd.read_csv(data / 'poi/poi_metrics.csv')
    if meta.session_pitch.duplicated().any() or poi.session_pitch.duplicated().any():
        raise ValueError('duplicate_pitch_key')
    if meta[['user', 'session', 'session_pitch']].isna().any().any() or meta.groupby('session').user.nunique().max() != 1 or meta.groupby('user').session.nunique().max() != 1:
        raise ValueError('ambiguous_player_session_identity')
    check = meta.merge(poi[['session_pitch', 'pitch_speed_mph']], on='session_pitch', validate='one_to_one', suffixes=('', '_poi'))
    if len(check) != len(meta) or len(poi) != len(meta) or not np.allclose(check.pitch_speed_mph, check.pitch_speed_mph_poi, atol=1e-8, rtol=0):
        raise ValueError('metadata_poi_identity_or_speed_conflict')
    groups = {}
    for name, cols in COLUMNS.items():
        frame = pd.read_csv(data / f'full_sig/{name}.csv', usecols=['session_pitch', 'time', *EVENTS, 'fp_100_time', *cols])
        if set(frame.session_pitch) != set(meta.session_pitch):
            raise ValueError('full_signal_pitch_key_mismatch:' + name)
        groups[name] = dict(tuple(frame.groupby('session_pitch', sort=False)))
    poi = poi.set_index('session_pitch')
    rows, exclusions = [], []
    for m in meta.sort_values(['user', 'session_pitch']).to_dict('records'):
        key = m['session_pitch']
        try:
            prepared = prepare_pitch({name: table[key] for name, table in groups.items()})
            trial_rows = [pitch_metrics(prepared, m, poi.loc[key].to_dict(), cutoff) for cutoff in CUTOFFS]
            rows.extend(trial_rows)
        except (ValueError, KeyError) as error:
            exclusions.append({'session_pitch': key, 'user': str(m['user']), 'reason': str(error)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('no_valid_pitches')
    count = frame[frame.cutoff_hz == 20].groupby('user').size()
    singles = count[count < 2].index
    for key, g in frame[(frame.cutoff_hz == 20) & frame.user.isin(singles)].groupby('session_pitch'):
        exclusions.append({'session_pitch': key, 'user': str(g.iloc[0].user), 'reason': 'fewer_than_two_valid_pitches'})
    return frame[~frame.user.isin(singles)].reset_index(drop=True), exclusions, len(meta)
