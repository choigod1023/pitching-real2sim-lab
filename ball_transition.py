"""Prescribed hand-point to free-ball transition audit, not coupled pitching."""
import argparse
import hashlib
import json
from pathlib import Path
import ezc3d
import mujoco
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from ball_release import hand_frame, rigid_offset_velocity

OUNCE_KG = .028349523125
MASS_RANGE_KG = np.array([5., 5.25])*OUNCE_KG
RADIUS_RANGE_M = np.array([9., 9.25])*.0254/(2*np.pi)
RULE_SOURCE = 'https://content.mlb.com/documents/2/2/4/305750224/2019_Official_Baseball_Rules_FINAL_.pdf'


def attachment_load(centers, rate, mass):
    """Force required by prescribed ball centers; opposite force acts on hand.

    This module reports that reaction but does not integrate a coupled hand.
    """
    if mass <= 0 or not np.isfinite([mass, rate]).all() or rate <= 0:
        raise ValueError('Positive finite mass and rate required')
    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 5 or not np.isfinite(centers).all():
        raise ValueError('Need finite Nx3 centers, at least five samples')
    velocity = np.gradient(centers, 1/rate, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, 1/rate, axis=0, edge_order=2)
    return mass*(acceleration-np.array([0., 0., -9.81]))


def free_flight(position, velocity, mass, radius, duration=.05, dt=.0001):
    """No-impulse release into gravity-only flight; no aerodynamic/spin claim."""
    position, velocity = np.asarray(position), np.asarray(velocity)
    if position.shape != (3,) or velocity.shape != (3,) or not np.isfinite([*position, *velocity, mass, radius, duration, dt]).all():
        raise ValueError('Invalid release state')
    if min(mass, radius, duration, dt) <= 0:
        raise ValueError('Positive mass, radius, duration and timestep required')
    steps = max(1, int(np.ceil(duration/dt)))
    step = duration/steps
    xml = f'''<mujoco><option timestep="{step}" gravity="0 0 -9.81" integrator="RK4"/>
      <worldbody><body name="ball"><freejoint/>
      <geom type="sphere" size="{radius}" mass="{mass}" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:3] = position
    data.qvel[:3] = velocity
    initial_velocity = data.qvel[:3].copy()
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if data.warning.number.any() or not np.isfinite(data.qpos).all():
            raise RuntimeError('Unstable ball flight')
    analytical = position+velocity*duration+np.array([0., 0., -.5*9.81*duration**2])
    return {'initial_speed_kmh': float(np.linalg.norm(initial_velocity)*3.6),
            'initial_kinetic_energy_j': float(.5*mass*np.dot(velocity, velocity)),
            'release_velocity_jump_mps': float(np.linalg.norm(initial_velocity-velocity)),
            'analytical_position_error_m': float(np.linalg.norm(data.qpos[:3]-analytical)),
            'final_position_m': data.qpos[:3].tolist(), 'duration_s': float(data.time)}


def run(data, out):
    data, out = Path(data), Path(out)
    meta = pd.read_csv(data/'metadata.csv').merge(
        pd.read_csv(data/'poi/poi_metrics.csv', usecols=['session_pitch', 'p_throws']),
        on='session_pitch', validate='one_to_one')
    row = meta[meta.p_throws.eq('R')].sort_values('session_pitch').iloc[0]
    paths = list((data/'c3d').rglob(row.filename_new))
    if len(paths) != 1:
        raise ValueError('Ambiguous C3D')
    c = ezc3d.c3d(str(paths[0]))
    labels = [v.strip() for v in c['parameters']['POINT']['LABELS']['value']]
    scale = {'m': 1., 'cm': .01, 'mm': .001}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
    rate = float(c['parameters']['POINT']['RATE']['value'][0])
    markers = []
    for name in ['RFIN', 'RWRA', 'RWRB']:
        values = c['data']['points'][:, labels.index(name), :].T
        if not np.isfinite(values).all() or np.any(values[:, 3] < 0):
            raise ValueError('Missing hand markers')
        markers.append(values[:, :3]*scale)
    events = pd.read_csv(data/'full_sig/landmarks.csv', usecols=['session_pitch', 'time', 'BR_time'])
    events = events[events.session_pitch.eq(row.session_pitch)].sort_values('time')
    if len(events) != len(markers[0]) or not np.allclose(events.time, np.arange(len(events))/rate, atol=.000051, rtol=0):
        raise ValueError('Unverified C3D alignment')
    if events.BR_time.nunique() != 1:
        raise ValueError('Ambiguous release event')
    index = int(np.rint(float(events.BR_time.iloc[0])*rate))
    if not 3 <= index < len(events)-3:
        raise ValueError('Insufficient derivative support')
    offsets = [np.zeros(3)]+[axis*sign*.02 for axis in np.eye(3) for sign in [-1, 1]]
    sos = butter(4, 20, fs=rate, output='sos')
    records = []
    for mode, points in [('raw', markers), ('lowpass20', [sosfiltfilt(sos, p, axis=0) for p in markers])]:
        origin, frames = hand_frame(*points)
        for offset in offsets:
            centers, velocity = rigid_offset_velocity(origin, frames, offset, rate)
            for mass in [*MASS_RANGE_KG, MASS_RANGE_KG.mean()]:
                load = attachment_load(centers, rate, mass)[index]
                result = free_flight(centers[index], velocity[index], mass, RADIUS_RANGE_M.mean())
                records.append({'processing': mode, 'offset_m': offset.tolist(), 'mass_kg': float(mass),
                                'required_attachment_force_n': load.tolist(),
                                'required_attachment_force_norm_n': float(np.linalg.norm(load)), **result})
    report = {'pitch': str(row.session_pitch), 'observed_ball_speed_kmh': float(row.pitch_speed_mph)*1.609344,
              'release_rounding_s': index/rate-float(events.BR_time.iloc[0]),
              'source_c3d': str(paths[0].resolve()),
              'source_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(), 'cases': records,
              'ball_mass_range_kg': MASS_RANGE_KG.tolist(), 'ball_radius_range_m': RADIUS_RANGE_M.tolist(),
              'rule_source': RULE_SOURCE, 'rule_edition': 2019,
              'offsets': 'Zero and axis +/-2cm are numerical probes, not identified or anatomical grip positions.',
              'pre_release': 'Ball center follows a prescribed hand marker frame. Required translational attachment force reported, reaction not fed back to arm.',
              'derivative_scope': 'Offline symmetric derivatives and zero-phase filtering use samples after release; attachment load is a prescribed-path diagnostic, not measured finger force.',
              'release': 'No impulsive force at detachment; continuous position and translational velocity; event from dataset.',
              'post_release': 'MuJoCo sphere under gravity for 50ms. No contact, drag, Magnus force, spin prediction or finger mechanics.',
              'identifiable': ['Regulation ball size/mass interval for cited rule edition', 'Recorded hand markers and event, subject to processing'],
              'not_identified': ['Ball center offset', 'Finger slip and impulse', 'Actual hand/ball inertial coupling', 'True release velocity from direct ball tracking'],
              'coupled_hand_arm_dynamics': False, 'validated_ball_prediction': False,
              'predicted_gain_kmh': None, 'coaching_ready': False}
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--out', type=Path, default=Path('artifacts/ball-transition'))
    args = parser.parse_args()
    report = run(args.data, args.out)
    print(json.dumps({'pitch': report['pitch'], 'cases': len(report['cases']),
                      'max_flight_error_m': max(r['analytical_position_error_m'] for r in report['cases'])}, indent=2))
