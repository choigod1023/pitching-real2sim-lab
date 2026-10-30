"""Substep work-budget shield; all generalized inputs may be attenuated."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from capacity_budget_gate import gate, rollout, split_linear_power
from coupled_payload import model_for
from joint_intervention import inverse_controls
from torque_arm import derivatives
from .profiles import PROFILES, bounded_wrist_controls, modify_wrist, movement_metrics

JOINTS = {'shoulder_rotation': slice(3, 6), 'elbow': slice(6, 7),
          'marker_roll': slice(7, 8), 'wrist_marker_rotation': slice(8, 11)}


def _model(lengths, timestep):
    model = model_for(lengths, timestep)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    return model


def _initial(model, position, velocity):
    data = mujoco.MjData(model)
    data.qpos[:] = position
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    return data


def replay_controls(lengths, position, velocity, applied, timestep):
    """Replay saved substep forces from the saved exact initial state."""
    model = _model(lengths, timestep)
    data = _initial(model, position, velocity)
    positions, velocities = [data.qpos.copy()], [data.qvel.copy()]
    for force in np.asarray(applied):
        data.qfrc_applied[:] = force
        mujoco.mj_step(model, data)
        if data.warning.number.any() or not np.isfinite(data.qvel).all():
            raise RuntimeError('Unstable saved-force replay')
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
    return np.asarray(positions), np.asarray(velocities)


def actual_accounting(lengths, positions, velocities, applied, timestep):
    """Independently recompute work/peak/energy from only accepted states/forces."""
    model = _model(lengths, timestep)
    applied = np.asarray(applied)
    if applied.shape != (len(positions)-1, model.nv) or velocities.shape != (len(positions), model.nv):
        raise ValueError('One applied force per saved state interval required')
    if not all(np.isfinite(a).all() for a in (positions, velocities, applied)):
        raise ValueError('Finite actual states and forces required')
    positive, negative, net = (np.zeros(model.nv) for _ in range(3))
    for force, v0, v1 in zip(applied, velocities[:-1], velocities[1:]):
        a, b, c = split_linear_power(force*v0, force*v1, timestep)
        positive += a
        negative += b
        net += c
    energies = []
    for index in (0, -1):
        data = _initial(model, positions[index], velocities[index])
        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)
        energies.append(float(data.energy.sum()))
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, model.site('ball_center').id)
    return {'speed_kmh': float(np.linalg.norm(jp@velocities[-1])*3.6),
            'positive_work_per_dof_j': positive.tolist(),
            'negative_work_magnitude_per_dof_j': negative.tolist(),
            'net_work_per_dof_j': net.tolist(),
            'total_positive_component_work_j': float(positive.sum()),
            'total_negative_component_work_magnitude_j': float(negative.sum()),
            'total_net_work_j': float(net.sum()),
            'shoulder_translation_net_work_j': float(net[:3].sum()),
            'shoulder_rotation_net_work_j': float(net[3:6].sum()),
            'shoulder_translation_positive_component_work_j': float(positive[:3].sum()),
            'shoulder_rotation_positive_component_work_j': float(positive[3:6].sum()),
            'peak_abs_generalized_force_per_dof': np.max(np.abs(applied), axis=0).tolist(),
            'peak_joint_torque_norm_nm': {name: float(np.linalg.norm(applied[:, part], axis=1).max())
                                          for name, part in JOINTS.items()},
            'mechanical_energy_change_j': energies[1]-energies[0],
            'energy_change_minus_applied_work_j': energies[1]-energies[0]-float(net.sum()),
            'initial_energy_j': energies[0], 'dt_s': timestep}


def shield_rollout(lengths, q, controls, dt, substeps=1, budgets=None):
    """Try 1, 1/2, ..., 2^-40, then zero at each actual RK4 substep.

    Budgets protect per-DOF positive/negative work only. Gravity remains enabled.
    Virtual trials restore MuJoCo's complete integration state and warning state.
    """
    if substeps not in (1, 2) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive dt and one/two substeps required')
    model = _model(lengths, dt/substeps)
    q, controls = np.asarray(q), np.asarray(controls, float)
    if len(q) < 3 or controls.shape != (len(q), model.nv) or not np.isfinite(controls).all():
        raise ValueError('Finite matching reference and controls required')
    if budgets is None:
        caps = None
    else:
        caps = tuple(np.asarray(budgets[k], float) for k in
                     ('positive_work_per_dof_j', 'negative_work_magnitude_per_dof_j'))
        if any(a.shape != (model.nv,) or not np.isfinite(a).all() or np.any(a < 0) for a in caps):
            raise ValueError('Finite nonnegative per-DOF work budgets required')
    velocity, _ = derivatives(model, q, dt)
    data = _initial(model, q[0], velocity[0])
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    snapshot = np.empty(mujoco.mj_stateSize(model, state_spec))
    positions, velocities = [data.qpos.copy()], [data.qvel.copy()]
    positive, negative = np.zeros(model.nv), np.zeros(model.nv)
    accepted, scales, retries, attempts = [], [], [], []
    for requested in controls[:-1]:
        for _ in range(substeps):
            mujoco.mj_getState(model, data, snapshot, state_spec)
            warning_number, warning_info = data.warning.number.copy(), data.warning.lastinfo.copy()
            v0 = data.qvel.copy()
            for trial in range(42):
                if trial:
                    mujoco.mj_setState(model, data, snapshot, state_spec)
                    data.warning.number[:] = warning_number
                    data.warning.lastinfo[:] = warning_info
                scale = 0. if trial == 41 else 2.**(-trial)
                force = requested*scale
                data.qfrc_applied[:] = force
                mujoco.mj_step(model, data)
                stable = not data.warning.number.any() and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                if stable:
                    a, b, _ = split_linear_power(force*v0, force*data.qvel, model.opt.timestep)
                    allowed = caps is None or (np.all(positive+a <= caps[0]) and np.all(negative+b <= caps[1]))
                    if allowed:
                        break
                if trial == 41:
                    raise RuntimeError('Zero-force fallback unstable or budget violated')
            positive += a
            negative += b
            accepted.append(force.copy())
            scales.append(scale)
            retries.append(trial)
            attempts.append(trial+1)
            positions.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
    positions, velocities, accepted = map(np.asarray, (positions, velocities, accepted))
    result = actual_accounting(lengths, positions, velocities, accepted, dt/substeps)
    if not np.array_equal(positive, result['positive_work_per_dof_j']) or not np.array_equal(negative, result['negative_work_magnitude_per_dof_j']):
        raise RuntimeError('Accepted-state work ledger mismatch')
    scales = np.asarray(scales)
    result.update(substeps=substeps, virtual_trial_count=int(sum(attempts)),
                  virtual_retry_count=int(sum(retries)), suppressed_substeps=int(np.sum(scales < 1)),
                  zero_force_substeps=int(np.sum(scales == 0)),
                  max_positive_budget_excess_j=0. if caps is None else float(np.max(positive-caps[0])),
                  max_negative_budget_excess_j=0. if caps is None else float(np.max(negative-caps[1])))
    trace = {'qpos': positions, 'qvel': velocities, 'applied_loads': accepted,
             'scale_per_substep': scales, 'scale_by_source_interval': scales.reshape(-1, substeps),
             'virtual_retries_per_substep': np.asarray(retries),
             'suppressed_time_s': np.flatnonzero(scales < 1)*(dt/substeps),
             'dt': dt/substeps, 'source_dt': dt, 'lengths': np.asarray(lengths)}
    return result, positions, velocities, trace


def run_experiment(source, reference_report, out='artifacts/gain-shield'):
    source, reference_report, out = Path(source), Path(reference_report), Path(out)
    reference = json.loads(reference_report.read_text(encoding='utf-8'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != reference['source_sha256']:
        raise ValueError('Reference input hash mismatch')
    state = np.load(source)
    q, dt, lengths = state['qpos'], float(state['dt']), state['lengths']
    model = _model(lengths, dt)
    controls = inverse_controls(model, q, dt)
    out.mkdir(parents=True, exist_ok=True)
    baselines, baseline_checks = {}, []
    for substeps in (1, 2):
        original, op, ov = rollout(lengths, q, controls, dt, substeps)
        previous = reference['baseline'][str(substeps)]
        for key in ('speed_kmh', 'positive_work_per_dof_j', 'negative_work_magnitude_per_dof_j'):
            if not np.allclose(original[key], previous[key], atol=1e-9, rtol=0):
                raise RuntimeError('Profile baseline not reproduced')
        b, bp, bv, trace = shield_rollout(lengths, q, controls, dt, substeps, original)
        exact = np.array_equal(bp[::substeps], op) and np.array_equal(bv[::substeps], ov)
        if not exact or b['virtual_retry_count'] or not gate(original, b)['strict_model_budget_pass']:
            raise RuntimeError('Baseline changed by work shield')
        baselines[substeps] = (b, bp, bv)
        baseline_checks.append({'substeps': substeps, 'exact_original_states': exact,
                                'virtual_retry_count': b['virtual_retry_count'], 'budget_gate': gate(original, b)})
        np.savez_compressed(out/f'baseline-sub{substeps}.npz', **trace)
    rows = []
    for profile in PROFILES:
        changed, _ = modify_wrist(model, q, dt, .030, profile)
        bounded = bounded_wrist_controls(controls, inverse_controls(model, changed, dt), 1.)
        for substeps in (1, 2):
            b, bp, bv = baselines[substeps]
            result, pos, vel, trace = shield_rollout(lengths, q, bounded, dt, substeps, b)
            result.update(movement_metrics(model, bp, bv, pos, vel))
            result.update(case=profile, window_ms=30., alpha=1.,
                          paired_gain_kmh=result['speed_kmh']-b['speed_kmh'],
                          positive_component_work_change_j=result['total_positive_component_work_j']-b['total_positive_component_work_j'],
                          budget_gate=gate(b, result), initial_state_equal=bool(np.array_equal(pos[0], bp[0]) and np.array_equal(vel[0], bv[0])))
            np.savez_compressed(out/f'{profile}-sub{substeps}.npz', **trace, proposed_source_controls=bounded)
            rows.append(result)
            print(profile, substeps, result['paired_gain_kmh'], result['budget_gate']['strict_model_budget_pass'], flush=True)
    decisions = []
    for profile in PROFILES:
        a, b = [r for r in rows if r['case'] == profile]
        bound = abs(baselines[1][0]['speed_kmh']-baselines[2][0]['speed_kmh'])+abs(a['speed_kmh']-b['speed_kmh'])
        discrepancy = abs(a['paired_gain_kmh']-b['paired_gain_kmh'])
        passed = all(r['initial_state_equal'] and r['budget_gate']['strict_model_budget_pass'] and r['pose_screen_pass'] and r['paired_gain_kmh'] > bound for r in (a, b)) and discrepancy <= .001
        decisions.append({'case': profile, 'conditional_model_candidate': bool(passed), 'fine_gain_kmh': b['paired_gain_kmh'],
                          'gain_resolution_difference_kmh': discrepancy, 'absolute_resolution_sum_kmh': bound,
                          'gain_at_least_1kmh': bool(b['paired_gain_kmh'] >= 1)})
    report = {'pitch': '3034_1', 'source': str(source.resolve()), 'source_sha256': digest,
              'reference_report': str(reference_report.resolve()), 'reference_report_sha256': hashlib.sha256(reference_report.read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256(Path(__file__).with_name('SHIELD_PROTOCOL.md').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'All generalized applied inputs may decrease; gravity retained; per-DOF work protection only. Distinct from fixed shoulder/elbow/marker-roll inputs.',
              'baseline': {k: v[0] for k, v in baselines.items()}, 'baseline_checks': baseline_checks,
              'runs': rows, 'decisions': decisions, 'conditional_candidates': [d for d in decisions if d['conditional_model_candidate']],
              'real_athlete_effect_verified': False, 'empirical_effect_95ci': None,
              'spin_or_command_effect_verified': False, 'physiological_capacity_validated': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--reference-report', required=True)
    parser.add_argument('--out', default='artifacts/gain-shield')
    args = parser.parse_args()
    run_experiment(args.source, args.reference_report, args.out)
