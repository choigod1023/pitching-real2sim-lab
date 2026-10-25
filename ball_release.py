"""Rigid-offset release sensitivity; unknown grip geometry, not predicted gains.

OBP hand tracking uses RFIN/RWRA/RWRB. This module's frame is an explicitly
defined marker frame, NOT the Visual3D anatomical frame. Its origin is RFIN.
No offset or release time is selected by matching the evaluated ball speed.
"""
import argparse
import json
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def hand_frame(fin, wra, wrb):
    """Return RFIN origin and right-handed matrices with axes in columns.

    x points from wrist midpoint toward RFIN; y is WRA-WRB projected normal
    to x; z=x cross y. Inputs are Nx3 arrays in meters. Reject degenerate data.
    """
    fin, wra, wrb = (np.asarray(a, dtype=float) for a in (fin, wra, wrb))
    if fin.ndim != 2 or fin.shape[1] != 3 or wra.shape != fin.shape or wrb.shape != fin.shape:
        raise ValueError('Markers must have matching Nx3 shapes')
    if not all(np.isfinite(a).all() for a in (fin, wra, wrb)):
        raise ValueError('Nonfinite marker data')
    x = fin - (wra + wrb) / 2
    length = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(length < 1e-8):
        raise ValueError('Degenerate hand longitudinal axis')
    x = x / length
    y = wra - wrb
    y = y - np.sum(y*x, axis=1, keepdims=True)*x
    width = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(width < 1e-8):
        raise ValueError('Degenerate hand transverse axis')
    y = y / width
    return fin.copy(), np.stack((x, y, np.cross(x, y)), axis=-1)


def rigid_offset_velocity(origin, frames, offset_m, rate):
    """Return assumed ball centers and velocities, including d(R*r)/dt.

    This is a kinematic diagnostic, not force-driven simulation of the ball.
    """
    origin, frames, offset = map(np.asarray, (origin, frames, offset_m))
    if origin.ndim != 2 or origin.shape[1:] != (3,) or len(origin) < 3:
        raise ValueError('Need at least three Nx3 origins')
    if frames.shape != (len(origin), 3, 3) or offset.shape != (3,):
        raise ValueError('Invalid frame or offset shape')
    if not np.isfinite(rate) or rate <= 0 or not all(np.isfinite(a).all() for a in (origin, frames, offset)):
        raise ValueError('Nonfinite inputs or invalid sample rate')
    world_offset = np.einsum('nij,j->ni', frames, offset)
    velocity = np.gradient(origin, 1/rate, axis=0, edge_order=2)
    velocity += np.gradient(world_offset, 1/rate, axis=0, edge_order=2)
    return origin + world_offset, velocity


def run(data, out, offset_step_m=.02, cutoff_hz=20.):
    """Select first lexicographic right-handed session_pitch, without outcomes."""
    if not np.isfinite(offset_step_m) or offset_step_m < 0:
        raise ValueError('Offset step must be finite and nonnegative')
    metadata = pd.read_csv(data/'metadata.csv')
    poi = pd.read_csv(data/'poi/poi_metrics.csv', usecols=['session_pitch', 'p_throws'])
    eligible = metadata.merge(poi, on='session_pitch', validate='one_to_one')
    row = eligible[eligible.p_throws.eq('R')].sort_values('session_pitch').iloc[0]
    paths = list((data/'c3d').rglob(row.filename_new))
    if len(paths) != 1:
        raise ValueError('Expected exactly one matching C3D')
    c = ezc3d.c3d(str(paths[0]))
    units = c['parameters']['POINT']['UNITS']['value'][0].strip().lower()
    scale = {'m': 1., 'cm': .01, 'mm': .001}[units]
    rate = float(c['parameters']['POINT']['RATE']['value'][0])
    labels = [x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
    markers = []
    for label in ['RFIN', 'RWRA', 'RWRB']:
        values = c['data']['points'][:, labels.index(label), :].T
        if not np.isfinite(values[:, :3]).all() or np.any(values[:, 3] < 0):
            raise ValueError('Marker gaps: no interpolation permitted in release diagnostic')
        markers.append(values[:, :3]*scale)
    if not 0 < cutoff_hz < rate/2:
        raise ValueError('Cutoff must be between zero and Nyquist')
    events = pd.read_csv(data/'full_sig/landmarks.csv', usecols=['session_pitch', 'time', 'BR_time'])
    events = events[events.session_pitch.eq(row.session_pitch)].sort_values('time')
    # Full signal is zero-based trial-relative time, not absolute C3D frame number.
    expected = np.arange(len(markers[0]))/rate
    if len(events) != len(expected) or not np.allclose(events.time, expected, atol=.000051, rtol=0):
        raise ValueError('C3D/full-signal time alignment not established')
    if events.BR_time.nunique() != 1:
        raise ValueError('Ambiguous release event')
    release = float(events.BR_time.iloc[0])
    center = int(np.rint(release*rate))
    if center < 1 or center >= len(expected)-1:
        raise ValueError('Release neighborhood outside recorded samples')
    offsets = [np.zeros(3)]
    for axis in np.eye(3):
        offsets.extend([axis*offset_step_m, -axis*offset_step_m])
    records = []
    sos = butter(4, cutoff_hz, fs=rate, output='sos')
    for processing, points in [('raw', markers), ('lowpass', [sosfiltfilt(sos, m, axis=0) for m in markers])]:
        origin, frames = hand_frame(*points)
        for offset in offsets:
            _, velocity = rigid_offset_velocity(origin, frames, offset, rate)
            for shift in [-1, 0, 1]:
                index = center + shift
                records.append(dict(processing=processing, offset_x_m=offset[0], offset_y_m=offset[1], offset_z_m=offset[2],
                    release_shift_frames=shift, sample_time_s=expected[index], hypothetical_speed_kmh=np.linalg.norm(velocity[index])*3.6,
                    hypothetical_homeward_velocity_kmh=velocity[index, 0]*3.6))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out/'sensitivity.csv', index=False)
    report = dict(session_pitch=str(row.session_pitch), c3d=str(paths[0]), observed_ball_speed_kmh=float(row.pitch_speed_mph)*1.609344,
        rate_hz=rate, recorded_release_time_s=release, nearest_release_frame=center,
        release_rounding_error_s=float(expected[center]-release), first_c3d_frame=int(c['header']['points']['first_frame']),
        frame_origin='RFIN marker, not ball center or hand_jc',
        offset_step_m=offset_step_m, offset_interpretation='User-configured numerical sensitivity; not anatomical or plausible bounds',
        lowpass=dict(order=4, cutoff_hz=cutoff_hz, method='zero-phase sosfiltfilt; not claimed identical to OBP processing'),
        selection='First lexicographic right-handed session_pitch; no outcome-based selection or calibration',
        time_alignment='Equal sample count and full-signal zero-based times verified; BR event detection provenance unavailable',
        predicted_gain_kmh=None, coaching_ready=False, cases=len(records),
        limitations=['Unknown grip geometry and finger slip', 'No ball dynamics, contact or spin',
                     'Hypothetical point velocities are not validated ball predictions',
                     'Timing sweep concerns recorded event uncertainty, not optimized release'],
        sources=['https://github.com/drivelineresearch/openbiomechanics/blob/main/baseball_pitching/code/v3d/model/v6_model_hybrid_lm.mdh',
                 'https://github.com/drivelineresearch/openbiomechanics/blob/main/baseball_pitching/README.md'])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root.parent/'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--out', type=Path, default=root/'artifacts/ball-release')
    parser.add_argument('--offset-step-m', type=float, default=.02)
    parser.add_argument('--cutoff-hz', type=float, default=20.)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.out, args.offset_step_m, args.cutoff_hz), indent=2))
