"""Marker-frame arm/hand with a mechanically coupled rigid ball payload.

Marker rotations are not anatomically calibrated joint angles. Generic hand
inertia and unknown grip remain explicit assumptions. No muscle capacity model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import ezc3d
import mujoco
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, RotationSpline
from ball_release import hand_frame
from pilot import regular_time, vector
from torque_arm import arm_poses, derivatives, unit


def wxyz(rotation):
    q = rotation.as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(q)):
        if q[i-1]@q[i] < 0:
            q[i] *= -1
    return q


def fit_pose(points, forearm_marker, hand_rotation):
    s, e, w = points[:, 0], points[:, 1], points[:, 2]
    base = arm_poses(s, e, w)
    upper = Rotation.from_quat(base[:, [4, 5, 6, 3]])
    hinge = Rotation.from_rotvec(np.column_stack([np.zeros(len(base)), base[:, 7], np.zeros(len(base))]))
    nominal = (upper*hinge).as_matrix()
    z = -unit(w-e)
    transverse = forearm_marker-e
    transverse -= np.sum(transverse*z, axis=1)[:, None]*z
    support = np.linalg.norm(transverse, axis=1)
    x = unit(transverse)
    actual = np.stack([x, np.cross(z, x), z], axis=-1)
    relative = np.einsum('nji,njk->nik', nominal, actual)
    roll = np.unwrap(np.arctan2(relative[:, 1, 0], relative[:, 0, 0]))
    wrist = Rotation.from_matrix(np.einsum('nji,njk->nik', actual, hand_rotation))
    return np.column_stack([base, roll, wxyz(wrist)]), support


def model_for(lengths, dt, ball_mass=.145, grip=(.15, 0., 0.), free_ball=False, hand_mass_scale=1.):
    if not np.isfinite([*lengths, dt, ball_mass, *grip, hand_mass_scale]).all() or min(*lengths, dt, hand_mass_scale) <= 0 or ball_mass < 0:
        raise ValueError('Invalid model parameters')
    def cylinder(m, l, r):
        return f'{m*(3*r*r+l*l)/12} {m*(3*r*r+l*l)/12} {m*r*r/2}'
    l1, l2 = lengths
    grip_text = ' '.join(map(str, grip))
    payload = '' if ball_mass == 0 else f'<geom type="sphere" size=".0364" mass="{ball_mass}" pos="{grip_text}"/>'
    free_body = ''
    if free_ball:
        if ball_mass <= 0:
            raise ValueError('Detached ball needs positive mass')
        payload = ''
        free_body = f'<body name="free_ball"><freejoint name="ball_free"/><geom type="sphere" size=".0364" mass="{ball_mass}"/></body>'
    # Hand cuboid: 0.16 x 0.08 x 0.03m, 0.482044kg, generic assumption.
    mh = .482044*hand_mass_scale
    hand_i = ' '.join(str(mh*(a*a+b*b)/12) for a, b in [(.08,.03),(.16,.03),(.16,.08)])
    xml = f'''<mujoco><compiler angle="radian"/><option timestep="{dt}" integrator="Euler"/>
    <default><geom contype="0" conaffinity="0"/></default><worldbody><body name="upper">
    <freejoint/><inertial pos="0 0 {-l1/2}" mass="2.14154" diaginertia="{cylinder(2.14154,l1,.04)}"/>
    <body name="forearm" pos="0 0 {-l1}"><joint name="elbow" axis="0 1 0"/>
    <joint name="marker_roll" axis="0 0 1"/>
    <inertial pos="0 0 {-l2/2}" mass="1.280182" diaginertia="{cylinder(1.280182,l2,.03)}"/>
    <body name="hand" pos="0 0 {-l2}"><joint name="wrist_marker_rotation" type="ball"/>
    <inertial pos=".08 0 0" mass="{mh}" diaginertia="{hand_i}"/>
    <site name="wrist"/><body name="ball_payload">
    {payload}<site name="ball_center" pos="{grip_text}"/></body>
    </body></body></body>{free_body}</worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def inverse_and_rollout(model, q, dt, return_state=False, applied_loads=None):
    vel, acc = derivatives(model, q, dt)
    d = mujoco.MjData(model)
    loads, reference = [], []
    for pos, v, a in zip(q, vel, acc):
        d.qpos[:] = pos; d.qvel[:] = v; d.qacc[:] = a
        mujoco.mj_inverse(model, d)
        loads.append(d.qfrc_inverse.copy())
        reference.append(d.site_xpos[model.site('ball_center').id].copy())
    loads, reference = np.array(loads), np.array(reference)
    controls = loads if applied_loads is None else np.asarray(applied_loads)
    if controls.shape != loads.shape or not np.isfinite(controls).all():
        raise ValueError('Applied loads must match reference dimensions and be finite')
    d = mujoco.MjData(model); d.qpos[:] = q[0]; d.qvel[:] = vel[0]
    errors = [0.]
    applied_shoulder_work = 0.
    for force, target in zip(controls[:-1], reference[1:]):
        applied_shoulder_work += float(force[:6]@d.qvel[:6])*dt
        d.qfrc_applied[:] = force
        mujoco.mj_step(model, d); mujoco.mj_forward(model, d)
        if d.warning.number.any() or not np.isfinite(d.qpos).all():
            raise RuntimeError('Unstable coupled rollout')
        errors.append(np.linalg.norm(d.site_xpos[model.site('ball_center').id]-target))
    result = {'ball_point_rmse_mm': float(np.sqrt(np.mean(np.square(errors)))*1000),
              'ball_point_final_error_mm': float(errors[-1]*1000),
              'external_shoulder_net_work_j': float(np.trapezoid(np.sum(loads[:, :6]*vel[:, :6], axis=1), dx=dt)),
              'wrist_peak_generalized_torque_nm': float(np.linalg.norm(loads[:, 8:11], axis=1).max()),
              'state_resets_after_initialization': 0,
              'load_policy': 'recomputed_inverse' if applied_loads is None else 'fixed_supplied',
              'integrator': int(model.opt.integrator),
              'actual_shoulder_work_left_riemann_j': applied_shoulder_work,
              'external_shoulder_net_work_scope': 'Inverse demand evaluated on reference, not actual applied work.'}
    jp, jr = np.zeros((3,model.nv)), np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,d,jp,jr,model.site('ball_center').id)
    result['conditional_release_speed_kmh'] = float(np.linalg.norm(jp@d.qvel)*3.6)
    if return_state:
        return result, loads, {'qpos': d.qpos.copy(), 'qvel': d.qvel.copy()}
    return result, loads


def run(data, pitch, out, cutoff_hz=20., steps=(1/2880,1/5760)):
    data, out = Path(data), Path(out)
    meta = pd.read_csv(data/'metadata.csv').merge(
        pd.read_csv(data/'poi/poi_metrics.csv', usecols=['session_pitch','p_throws']),
        on='session_pitch', validate='one_to_one')
    selected = meta[meta.session_pitch.eq(pitch) & meta.p_throws.eq('R')]
    if len(selected) != 1:
        raise ValueError('Expected one right-handed pitch')
    row = selected.iloc[0]
    paths = list((data/'c3d').rglob(row.filename_new))
    if len(paths) != 1:
        raise ValueError('Ambiguous C3D')
    c = ezc3d.c3d(str(paths[0]))
    labels = [v.strip() for v in c['parameters']['POINT']['LABELS']['value']]
    rate = float(c['parameters']['POINT']['RATE']['value'][0])
    scale = {'m':1., 'cm':.01, 'mm':.001}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
    if not np.isfinite(cutoff_hz) or not 0 <= cutoff_hz < rate/2:
        raise ValueError('Cutoff must be zero (raw) or below Nyquist')
    sos = None if cutoff_hz == 0 else butter(4, cutoff_hz, fs=rate, output='sos')
    markers = {}
    for name in ['RFIN', 'RWRA', 'RWRB', 'RFRM']:
        raw = c['data']['points'][:, labels.index(name), :].T
        if not np.isfinite(raw).all() or np.any(raw[:, 3] < 0):
            raise ValueError('Missing marker; no gap filling')
        markers[name] = raw[:, :3]*scale if sos is None else sosfiltfilt(sos, raw[:, :3]*scale, axis=0)
    all_data = pd.read_csv(data/'full_sig/landmarks.csv')
    g = all_data[all_data.session_pitch.eq(pitch)].sort_values('time')
    t = regular_time(g)
    if len(t) != len(markers['RFIN']) or not np.allclose(t, np.arange(len(t))/rate, atol=.000051, rtol=0):
        raise ValueError('Unverified time alignment')
    _, frames = hand_frame(markers['RFIN'], markers['RWRA'], markers['RWRB'])
    hand_curve = RotationSpline(t, Rotation.from_matrix(frames))
    points_curve = CubicSpline(t, np.stack([vector(g,n+'_jc') for n in ['shoulder','elbow','wrist']], axis=1))
    fore_curve = CubicSpline(t, markers['RFRM'])
    a, b = float(g.fp_10_time.iloc[0]), float(g.BR_time.iloc[0])
    if not t[1] < a < b < t[-2]:
        raise ValueError('Insufficient interpolation support')
    lengths = np.median(np.linalg.norm(np.diff(points_curve(np.linspace(a,b,100)), axis=1), axis=2), axis=0)
    runs = []
    for requested_dt in steps:
        times = np.linspace(a,b,int(np.ceil((b-a)/requested_dt))+1); dt = float(times[1]-times[0])
        q, support = fit_pose(points_curve(times), fore_curve(times), hand_curve(times).as_matrix())
        empty, empty_loads = inverse_and_rollout(model_for(lengths, dt, ball_mass=0.), q, dt)
        loaded, loaded_loads, state = inverse_and_rollout(model_for(lengths, dt), q, dt, return_state=True)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out/f'release-state-{len(runs)}.npz', lengths=lengths, dt=dt,
                 qpos=state['qpos'], qvel=state['qvel'])
        np.savez(out/f'reference-{len(runs)}.npz', lengths=lengths, dt=dt,
                 qpos=q, required_loads=loaded_loads)
        delta = loaded_loads-empty_loads
        runs.append({'dt_s': dt, 'min_forearm_marker_transverse_support_mm': float(support.min()*1000),
                     'without_ball_mass': empty, 'with_ball_mass': loaded,
                     'payload_shoulder_force_delta_rms_n': float(np.sqrt(np.mean(np.sum(delta[:, :3]**2, axis=1)))),
                     'payload_wrist_torque_delta_rms_nm': float(np.sqrt(np.mean(np.sum(delta[:, 8:11]**2, axis=1))))})
    result = {'pitch': pitch, 'runs': runs, 'ball_mass_kg': .145, 'grip_from_wrist_m': [.15,0.,0.],
              'source_c3d_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
              'hand_mass_source': 'https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal_OpenSense/Rajagopal2015_opensense.osim',
              'ball_parameter_source': 'https://uwaterloo.ca/motion-research-group/sites/default/files/2026-01/mbsd_pub_baseball_fd_cattias_final.pdf',
              'grip_assumption': '15cm along marker hand x from retargeted wrist; numerical assumption, not measured grip.',
              'marker_roll_identifiable': 'RFRM supplies a transverse direction; this is a skin-marker frame, not calibrated anatomical pronation.',
              'wrist_frame': 'Hand frame relative to forearm marker frame; three unrestricted rotational DOFs, not a validated wrist joint.',
              'inertia': 'Generic arm cylinders, 0.482044kg hand cuboid 0.16x0.08x0.03m, solid spherical ball. Not subject-specific.',
              'coupling': 'Ball inertia is part of articulated dynamics; ball-present and ball-absent loads are recomputed separately, not equal-ability candidates.',
              'cutoff_hz': cutoff_hz,
              'processing': 'Raw if cutoff=0, otherwise zero-phase hand/forearm filtering; processed joint centers unchanged and may use different processing. Offline RotationSpline interpolation.',
              'limitations': ['No finger mechanics or true grip identification', 'No muscle capacities or anatomical ROM',
                             'Prescribed observed motion and external shoulder wrench', 'No detachment integrated with arm',
                             'No ball-speed improvement prediction or coaching recommendation'], 'optimization_ready': False}
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    p.add_argument('--pitch', default='3034_1'); p.add_argument('--out', type=Path, default=Path('artifacts/coupled-payload'))
    a = p.parse_args(); print(json.dumps(run(a.data,a.pitch,a.out), indent=2))
