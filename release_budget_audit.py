"""Diagnose the original 3034 release deficit without correcting its speed.

The scalar record is used only for a conditional terminal-state discrepancy.
It is not a same-instant direct ball observation, a contact-force measurement,
an additional energy allowance, or an optimizer target.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from coupled_payload import model_for


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_discrepancy(velocity, recorded_speed_kmh, mass=.145):
    v = np.asarray(velocity, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError('Need a finite 3D model velocity')
    if not np.isfinite([recorded_speed_kmh, mass]).all() or recorded_speed_kmh < 0 or mass <= 0:
        raise ValueError('Invalid record or ball mass')
    current = float(np.linalg.norm(v))
    target = recorded_speed_kmh / 3.6
    difference = abs(target-current)
    return {
        'model_speed_kmh': current*3.6,
        'recorded_scalar_kmh': float(recorded_speed_kmh),
        'signed_record_minus_model_kmh': (target-current)*3.6,
        'minimum_velocity_vector_difference_mps': difference,
        'minimum_terminal_momentum_difference_ns': mass*difference,
        'terminal_ball_translational_energy_difference_j': .5*mass*(target**2-current**2),
        'equivalent_average_force_n': {
            str(ms): mass*difference/(ms/1000.) for ms in (7, 8, 9)},
        'actual_contact_force_n': None,
        'actual_release_velocity_error_kmh': None,
        'corrected_speed_kmh': None,
        'scope': 'Conditional terminal comparison; recorded direction/time/position unverified. Not an applied force, work allowance, or correction.'}


def trapezoid_window(time, values, start, stop):
    time, values = np.asarray(time), np.asarray(values)
    if (time.ndim != 1 or len(time) < 2 or values.shape[0] != len(time)
            or not np.isfinite(time).all() or not np.isfinite(values).all()
            or np.any(np.diff(time) <= 0) or not time[0] <= start < stop <= time[-1]):
        raise ValueError('Invalid sampled integration window')
    t = np.r_[start, time[(time > start) & (time < stop)], stop]
    flat = values.reshape(len(time), -1)
    y = np.column_stack([np.interp(t, time, column) for column in flat.T])
    return np.trapezoid(y, t, axis=0).reshape(values.shape[1:])


def energy_work_audit(model, qpos, qvel, loads, time):
    """Evaluate saved states; never replay, reset, or modify model input forces."""
    qpos, qvel, loads, time = map(np.asarray, (qpos, qvel, loads, time))
    n = len(time)
    if qpos.shape != (n, model.nq) or qvel.shape != (n, model.nv) or loads.shape != qvel.shape:
        raise ValueError('State/input shapes differ')
    if n < 3 or not all(np.isfinite(a).all() for a in (qpos, qvel, loads, time)) or np.any(np.diff(time) <= 0):
        raise ValueError('Nonfinite or nonmonotonic saved trajectory')
    d = mujoco.MjData(model)
    velocities, energy, arm_ke, ball_ke = [], [], [], []
    ball = model.body('ball_payload').id
    payload_mass = float(model.body_mass[ball])
    if not np.isclose(payload_mass, .145):
        raise ValueError('Original ball mass changed')
    for q, v in zip(qpos, qvel):
        d.qpos[:] = q; d.qvel[:] = v
        mujoco.mj_forward(model, d)
        mujoco.mj_energyPos(model, d); mujoco.mj_energyVel(model, d)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d, jp, jr, model.site('ball_center').id)
        bv = jp @ v
        # Total energy includes ball rotation and gravity; scalar label constrains translation only.
        bke = .5*payload_mass*float(bv@bv)
        velocities.append(bv); energy.append(d.energy.copy())
        ball_ke.append(bke); arm_ke.append(float(d.energy[1])-bke)
    energy, velocities = np.array(energy), np.array(velocities)
    power = loads*qvel
    windows = {}
    for name, start in [('whole_saved_throw', time[0]), ('last_8ms_diagnostic', time[-1]-.008)]:
        w = trapezoid_window(time, power, start, time[-1])
        total = energy.sum(axis=1)
        delta = float(total[-1]-np.interp(start, time, total))
        work = float(w.sum())
        windows[name] = {
            'start_s': float(start), 'stop_s': float(time[-1]),
            'generalized_net_work_j': work,
            'mechanical_energy_change_j': delta,
            'net_work_minus_energy_change_j': work-delta,
            'shoulder_translation_work_j': float(w[:3].sum()),
            'shoulder_rotation_work_j': float(w[3:6].sum()),
            'elbow_and_forearm_work_j': float(w[6:8].sum()),
            'wrist_marker_rotation_work_j': float(w[8:11].sum()),
            'ball_translational_ke_change_j': float(ball_ke[-1]-np.interp(start, time, ball_ke)),
            'other_kinetic_energy_change_j': float(arm_ke[-1]-np.interp(start, time, arm_ke)),
            'work_quadrature': 'Trapezoid of saved endpoint force*velocity; original simulation uses held interval force. Residual includes quadrature/time-discretization.'}
    return {
        'release_velocity_mps': velocities[-1].tolist(),
        'ball_translational_ke_at_release_j': float(ball_ke[-1]),
        'other_kinetic_energy_at_release_j': float(arm_ke[-1]),
        'other_kinetic_energy_scope': 'Includes rigid ball rotational kinetic energy; not available/recoverable energy capacity.',
        'input_force_max_per_dof': np.max(np.abs(loads), axis=0).tolist(),
        'input_wrist_torque_norm_peak_nm': float(np.linalg.norm(loads[:, 8:11], axis=1).max()),
        'state_changes': 0, 'input_changes': 0,
        'windows': windows}


def run(root, output):
    root, output = Path(root), Path(output)
    baseline = root/'speed-journey/artifacts/speed-video'
    reference = root/'grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz'
    manifest_path, rollout_path = baseline/'video_manifest.json', baseline/'rollout.npz'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['pitch'] != '3034_1' or sha(reference) != manifest['source_sha256']:
        raise ValueError('Original video/source identity changed')
    with np.load(reference) as z:
        lengths = z['lengths'].copy()
    with np.load(rollout_path) as z:
        dt = float(z['dt'])
        model = model_for(lengths, dt)
        saved_speed = float(z['speed_kmh'][-1])
        audit = energy_work_audit(model, z['qpos'], z['qvel'], z['applied_generalized_loads'], z['times'])
    speed = np.linalg.norm(audit['release_velocity_mps'])*3.6
    if not np.isclose(speed, saved_speed, atol=1e-9, rtol=0) or not np.isclose(speed, manifest['model_release_speed_kmh'], atol=1e-9, rtol=0):
        raise ValueError('Saved original video speed no longer reproduced')
    original = terminal_discrepancy(audit['release_velocity_mps'], manifest['observed_speed_kmh'])
    ledger_path = Path(__file__).with_name('same_pitch_results.json')
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    cohort = []
    for record in ledger['records']:
        if record['athlete'] != 2 or record['session'] != 3034:
            raise ValueError('Mixed athlete or session')
        vector = record['ladder']['vectors_mps']['continuous_assumed_ball']
        item = terminal_discrepancy(vector, record['recorded_speed_kmh'])
        item['pitch'] = record['pitch']; item['model_series'] = 'continuous_fixed_grip_kinematics'
        cohort.append(item)
    if [r['pitch'] for r in cohort] != [f'3034_{i}' for i in range(1, 6)]:
        raise ValueError('Development cohort changed')
    output.mkdir(parents=True, exist_ok=True)
    sources = [manifest_path, rollout_path, reference, ledger_path]
    result = {
        'protocol': 'RELEASE_BUDGET_PROTOCOL.md',
        'original_video_identity': {'dataset': 'OBP/baseball_pitching', 'pitch': '3034_1', 'athlete': 2, 'session': 3034,
                                    'model_series': 'original_speed_journey_arm_hand_rigid_ball_RK4'},
        'sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sources],
        'original_video_terminal_discrepancy': original,
        'original_saved_trajectory_energy_audit': audit,
        'five_pitch_kinematic_diagnostics': cohort,
        'new_physical_correction_adopted': False,
        'speed_gap_reduction_kmh': 0.,
        'actual_contact_relative_motion_observed': False,
        'scientific_stage_passed': False,
        'interpretation': 'Missing terminal ball energy must be accounted for by redistribution, reduced absorption, or changed input work. This diagnostic does not show whether same-budget redistribution is feasible. No added finger force or corrected speed is simulated.'}
    path = output/'release_budget_results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(path), 'original': original, 'energy_windows': audit['windows']}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--output', type=Path, default=Path('artifacts/release-budget'))
    args = ap.parse_args()
    run(args.root, args.output)
