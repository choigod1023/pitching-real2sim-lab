"""Forward whole-body response to input timing, not prescribed-pose search."""
import json
from itertools import product
from pathlib import Path
import numpy as np
import mujoco
from coupled_body_search import build, inverse, groups, positions
from trace_error_audit import ObservedPitch, integrate_inputs
from capacity_budget_gate import split_linear_power

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/control-retiming'
SHIFTS = {'lead_knee': -1/360, 'torso_rotation': -1/360, 'rear_knee': 2/360}
ARM = ('pitch_shoulder', 'elbow', 'marker_roll', 'wrist_marker_rotation')


def retime(m, source_t, source_loads, times, factors):
    """Convex interpolation never scales amplitudes; reject extrapolation."""
    source_t, source_loads, times = map(np.asarray, (source_t, source_loads, times))
    if len(factors) != 3 or source_loads.shape != (len(source_t), m.nv):
        raise ValueError('Expected three timing factors and a full source control matrix')
    if len(source_t) < 2 or not np.all(np.diff(source_t) > 0):
        raise ValueError('Source times must strictly increase')
    if not all(np.isfinite(x).all() for x in (source_t, source_loads, times, np.asarray(factors))):
        raise ValueError('Non-finite timing input')
    out = np.column_stack([np.interp(times, source_t, x) for x in source_loads.T])
    ramp = np.clip((times-.015)/.05, 0, 1)
    ramp = ramp**3 * (10-15*ramp+6*ramp*ramp)
    for (name, shift), factor in zip(SHIFTS.items(), factors):
        query = times - shift*factor*ramp
        if query.min() < source_t[0] or query.max() > source_t[-1]:
            raise ValueError('Retiming would extrapolate source forces')
        for col in groups(m)[name]:
            out[:, col] = np.interp(query, source_t, source_loads[:, col])
    return out


def mechanics(m, q, v, loads, dt):
    gs = groups(m)
    power = np.column_stack([np.sum(loads[:, ix]*v[:, ix], axis=1) for ix in gs.values()])
    pos, neg, _ = split_linear_power(power[:-1], power[1:], dt)
    pos, neg = pos.sum(axis=0), neg.sum(axis=0)
    costs = {name: {'positive_j': float(pos[i]), 'negative_j': float(neg[i]),
                    'peak': float(np.linalg.norm(loads[:, ix], axis=1).max())}
             for i, (name, ix) in enumerate(gs.items())}
    d = mujoco.MjData(m)
    energies = []
    for j in (0, -1):
        d.qpos[:] = q[j]; d.qvel[:] = v[j]
        mujoco.mj_forward(m, d); mujoco.mj_energyPos(m, d); mujoco.mj_energyVel(m, d)
        energies.append(float(d.energy.sum()))
    jp = np.zeros((3, m.nv)); jr = jp.copy()
    mujoco.mj_jacSite(m, d, jp, jr, m.site('ball_center').id)
    velocity = jp@v[-1]
    return {'speed_kmh': float(np.linalg.norm(velocity)*3.6), 'velocity_mps': velocity.tolist(),
            'costs': costs, 'positive_j': float(pos.sum()), 'negative_j': float(neg.sum()),
            'energy_residual_j': float(energies[1]-energies[0]-(pos-neg).sum())}


def exceeds(value, base):
    return value-base > 1e-8*max(1, abs(base))


def budget(base, candidate, redistribution=False):
    violations = []
    for name, cost in candidate['costs'].items():
        metrics = ('peak',) if redistribution and name not in (*ARM, 'external_force', 'external_torque') else ('peak', 'positive_j', 'negative_j')
        for metric in metrics:
            if exceeds(cost[metric], base['costs'][name][metric]):
                violations.append({'group': name, 'metric': metric,
                                   'excess': cost[metric]-base['costs'][name][metric]})
    if redistribution:
        for metric in ('positive_j', 'negative_j'):
            if exceeds(candidate[metric], base[metric]):
                violations.append({'group': 'total', 'metric': metric, 'excess': candidate[metric]-base[metric]})
    return violations


def pose(m, bq, cq, bv, cv):
    names, bp = positions(m, bq); _, cp = positions(m, cq)
    delta = np.zeros((len(bq), m.nv))
    for i, (b, c) in enumerate(zip(bq, cq)):
        mujoco.mj_differentiatePos(m, delta[i], 1., b, c)
    angles = {n: float(np.rad2deg(np.linalg.norm(delta[:, ix], axis=1)).max())
              for n, ix in groups(m).items() if n != 'external_force'}
    site_changes = {n: float(np.linalg.norm(cp[:, i]-bp[:, i], axis=1).max()*1000)
                    for i, n in enumerate(names)}
    feet = max(site_changes[n] for n in ('rear_ankle_jc', 'lead_ankle_jc'))
    root = float(np.linalg.norm(cq[:, :3]-bq[:, :3], axis=1).max()*1000)
    direction = float(np.rad2deg(np.arccos(np.clip(np.dot(bv, cv)/np.linalg.norm(bv)/np.linalg.norm(cv), -1, 1))))
    maximum = max(angles.values())
    return {'joint_change_deg': angles, 'site_change_mm': site_changes,
            'max_joint_change_deg': maximum, 'max_foot_change_mm': feet,
            'max_root_change_mm': root, 'release_direction_change_deg': direction,
            'pose_pass': bool(maximum <= 3 and feet <= 20 and root <= 10 and direction <= 1)}


def run(output=OUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    obs = ObservedPitch('3034_3')
    pad = 50
    m, qsource, dt, ts, observed = build(1/5760, '3034_3', obs, pad, output/'source')
    source_loads, source_v = inverse(m, qsource, dt)
    # Differentiate padded reference, use same initial conditions and XML at both resolutions.
    reference = qsource[pad:-pad]; v0 = source_v[pad].copy(); q0 = reference[0].copy()
    source_t = ts-ts[pad]
    n = len(reference)-1
    if n % 2: raise ValueError('Predeclared coarse grid requires even fine intervals')
    reports = {}; saved = {}
    for stride in (2, 1):
        step = dt*stride; times = np.arange(n//stride+1)*step
        bm = mujoco.MjModel.from_xml_path(str(output/'source/model.xml'))
        base_loads = retime(bm, source_t, source_loads, times, (0, 0, 0))
        bq, bv, info = integrate_inputs(bm, q0, v0, base_loads, step, policy='foh')
        base = mechanics(bm, bq, bv, base_loads, step)
        names, baseline_pos = positions(bm, bq)
        _, reference_pos = positions(bm, reference[::stride])
        base['reference_site_rmse_mm'] = dict(zip(names, (np.sqrt(np.mean(np.sum((baseline_pos-reference_pos)**2, axis=2), axis=0))*1000).tolist()))
        measured = {name: value[pad:-pad:stride] for name, value in observed.items()}
        arm_points = obs.points(ts[pad:-pad:stride])
        measured.update(shoulder_jc=arm_points[:,0], elbow_jc=arm_points[:,1], wrist=arm_points[:,2])
        base['observed_joint_rmse_mm'] = {name: float(np.sqrt(np.mean(np.sum((baseline_pos[:,names.index(name)]-value)**2,axis=1)))*1000)
                                          for name,value in measured.items()}
        print(json.dumps({'stride': stride, 'baseline_speed_kmh': base['speed_kmh'], 'energy_residual_j': base['energy_residual_j']}), flush=True)
        runs = []
        for factors in product((-1, 0, 1), repeat=3):
            key = ','.join(map(str, factors))
            loads = retime(bm, source_t, source_loads, times, factors)
            try:
                cq, cv, ci = integrate_inputs(bm, q0, v0, loads, step, policy='foh')
                candidate = mechanics(bm, cq, cv, loads, step)
                p = pose(bm, bq, cq, base['velocity_mps'], candidate['velocity_mps'])
                strict = budget(base, candidate); redis = budget(base, candidate, True)
                runs.append({'id': key, 'factors': list(factors), 'gain_kmh': candidate['speed_kmh']-base['speed_kmh'],
                             **candidate, **p, 'strict_violations': strict, 'redistribution_violations': redis,
                             'energy_pass': bool(abs(candidate['energy_residual_j']) <= .1 and abs(base['energy_residual_j']) <= .1),
                             'state_resets': ci['state_resets']})
                if stride == 1: saved[key] = (cq, cv, loads)
            except RuntimeError as exc:
                runs.append({'id': key, 'factors': list(factors), 'failure': str(exc)})
        reports[stride] = {'dt_s': step, 'baseline': base, 'runs': runs, 'integrator': info}
        if stride == 1: saved['baseline'] = (bq, bv, base_loads)
    coarse = {r['id']: r for r in reports[2]['runs']}
    base_diff = abs(reports[1]['baseline']['speed_kmh']-reports[2]['baseline']['speed_kmh'])
    for r in reports[1]['runs']:
        if 'failure' in r: continue
        c = coarse[r['id']]
        r['numerical_difference_kmh'] = abs(r['speed_kmh']-c['speed_kmh']) if 'failure' not in c else None
        r['gain_above_discretization'] = bool(r['numerical_difference_kmh'] is not None and r['gain_kmh'] > base_diff+r['numerical_difference_kmh'])
        for gate in ('strict', 'redistribution'):
            r[gate+'_resolved_candidate'] = bool('failure' not in c and r['pose_pass'] and c['pose_pass'] and r['energy_pass'] and c['energy_pass'] and not r[gate+'_violations'] and not c[gate+'_violations'] and r['gain_above_discretization'])
    fine = [r for r in reports[1]['runs'] if 'failure' not in r]
    eligible = [r for r in fine if r['redistribution_resolved_candidate']]
    pool = eligible or [r for r in fine if r['pose_pass'] and r['gain_kmh'] > 0] or fine
    chosen = max(pool, key=lambda r: r['gain_kmh'])
    bq, bv, bl = saved['baseline']; cq, cv, cl = saved[chosen['id']]
    np.savez_compressed(output/'selected.npz', time_s=np.arange(n+1)*dt, baseline_q=bq, baseline_v=bv,
                        candidate_q=cq, candidate_v=cv, baseline_loads=bl, candidate_loads=cl,
                        reference_q=reference, dt_s=dt,
                        observed_positions=np.stack([measured.get(name,np.full((n+1,3),np.nan)) for name in names],axis=1))
    result = {'pitch': '3034_3', 'recorded_speed_kmh': obs.ledger['recorded_speed_kmh'],
              'source_ledger': obs.ledger, 'conditions': 27, 'resolutions': reports,
              'baseline_discretization_kmh': base_diff, 'selected_id': chosen['id'],
              'selected': chosen, 'strict_candidates': sum(r['strict_resolved_candidate'] for r in fine),
              'redistribution_candidates': len(eligible), 'actual_athlete_improvement_demonstrated': False,
              'muscle_capacity_validated': False, 'ground_contact_validated': False,
              'hand_ball_release_validated': False, 'spin_or_command_improvement': None,
              'scope': 'Control-retimed rigid-body mechanism test; no per-candidate inverse fitting; not a validated performance prediction.'}
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('selected_id', 'strict_candidates', 'redistribution_candidates', 'baseline_discretization_kmh')}, ensure_ascii=False), flush=True)
    print(json.dumps({k: chosen[k] for k in ('id', 'gain_kmh', 'speed_kmh', 'positive_j',
                     'max_joint_change_deg', 'max_foot_change_mm', 'pose_pass',
                     'strict_resolved_candidate', 'redistribution_resolved_candidate')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    run()
