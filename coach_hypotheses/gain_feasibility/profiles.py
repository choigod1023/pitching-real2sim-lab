"""Fixed endpoint-angle waveform checks under the existing mechanical gate."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from capacity_budget_gate import gate, rollout
from coupled_payload import model_for
from joint_intervention import initial_energy, inverse_controls, rotation_differences
from torque_arm import derivatives

WINDOWS = (.020, .030, .040, .060)
PROFILES = ('cubic_in', 'smoothstep', 'smootherstep')
ALPHAS = (.25, .5, 1.)


def profile_value(phase, name):
    """Evaluate a declared unit ramp; endpoint derivatives differ by design."""
    s = np.asarray(phase, dtype=float)
    if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Finite phase in [0, 1] required')
    if name == 'cubic_in':
        return s**3
    if name == 'smoothstep':
        return 3*s**2 - 2*s**3
    if name == 'smootherstep':
        return 10*s**3 - 15*s**4 + 6*s**5
    raise ValueError('Undeclared profile')


def modify_wrist(model, q, dt, window, profile):
    """Prescribe +2 degrees on marker DOF10 without changing the initial state."""
    if not np.isfinite([dt, window]).all() or min(dt, window) <= 0:
        raise ValueError('Positive finite times required')
    if len(q) < 3 or window >= (len(q)-2)*dt:
        raise ValueError('Window must leave two initial samples unchanged')
    time = np.arange(len(q))*dt
    phase = np.clip((time-(time[-1]-window))/window, 0, 1)
    angles = np.deg2rad(2.)*profile_value(phase, profile)
    changed = np.array(q, copy=True)
    tangent = np.zeros(model.nv)
    for pos, angle in zip(changed, angles):
        tangent[10] = angle
        mujoco.mj_integratePos(model, pos, tangent, 1.)
    if not np.array_equal(changed[:2], q[:2]):
        raise RuntimeError('Initial state altered')
    return changed, angles


def bounded_wrist_controls(baseline, requested, alpha):
    """Blend then box/norm clip only DOF8:11 using actually applied intervals."""
    b, r = np.asarray(baseline, float), np.asarray(requested, float)
    if b.ndim != 2 or b.shape != r.shape or b.shape[1] != 11 or len(b) < 2:
        raise ValueError('Matching N by 11 controls required')
    if not np.isfinite(b).all() or not np.isfinite(r).all() or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Finite controls and alpha in [0,1] required')
    out = b.copy()
    if alpha == 0:
        return out
    bounds = np.max(np.abs(b[:-1, 8:11]), axis=0)
    norm_bound = np.linalg.norm(b[:-1, 8:11], axis=1).max()
    wrist = np.clip(b[:-1, 8:11]+alpha*(r[:-1, 8:11]-b[:-1, 8:11]), -bounds, bounds)
    norms = np.linalg.norm(wrist, axis=1)
    scales = np.minimum(1., np.divide(norm_bound, norms, out=np.ones_like(norms), where=norms > 0))
    out[:-1, 8:11] = wrist*scales[:, None]
    return out


def ball_states(model, positions, velocities):
    """Calculate actual ball-center positions and Jacobian velocities."""
    if len(positions) != len(velocities) or not len(positions):
        raise ValueError('Matching nonempty states required')
    data = mujoco.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    points, speeds = [], []
    for q, v in zip(positions, velocities):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jp, jr, model.site('ball_center').id)
        points.append(data.site_xpos[model.site('ball_center').id].copy())
        speeds.append(jp@v)
    return np.asarray(points), np.asarray(speeds)


def movement_metrics(model, baseline_pos, baseline_vel, positions, velocities):
    """Retain the prior conservative rotation, position and direction screens."""
    bp, bv = ball_states(model, baseline_pos, baseline_vel)
    cp, cv = ball_states(model, positions, velocities)
    if len(bp) != len(cp):
        raise ValueError('Matching trajectory samples required')
    rotations = rotation_differences(model, baseline_pos, positions)
    norm = np.linalg.norm(bv[-1])*np.linalg.norm(cv[-1])
    if norm == 0:
        raise ValueError('Nonzero release velocity required for direction')
    direction = float(np.rad2deg(np.arccos(np.clip(bv[-1]@cv[-1]/norm, -1, 1))))
    distance = float(np.linalg.norm(cp-bp, axis=1).max()*1000)
    max_rotation = max(rotations.values())
    diff = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, diff, 1., baseline_pos[-1], positions[-1])
    return {'max_rotation_change_deg': max_rotation,
            'rotation_change_by_joint_deg': rotations,
            'max_ball_position_change_mm': distance,
            'release_direction_change_deg': direction,
            'max_shoulder_translation_change_mm': float(np.linalg.norm(positions[:, :3]-baseline_pos[:, :3], axis=1).max()*1000),
            'endpoint_wrist_marker_rotation_change_deg': np.rad2deg(diff[8:11]).tolist(),
            'endpoint_wrist_marker_velocity_change_deg_s': np.rad2deg(velocities[-1, 8:11]-baseline_vel[-1, 8:11]).tolist(),
            'pose_screen_pass': bool(max_rotation <= 3 and distance <= 20 and direction <= 1)}


def run_experiment(source, out, reference_report):
    """Execute every declared profile and bounded input; preserve all failures."""
    source, out, reference_report = Path(source), Path(out), Path(reference_report)
    old = json.loads(reference_report.read_text(encoding='utf-8'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != old['source_sha256']:
        raise ValueError('Reference input hash mismatch')
    state = np.load(source)
    q, dt, lengths = state['qpos'], float(state['dt']), state['lengths']
    model = model_for(lengths, dt)
    controls = inverse_controls(model, q, dt)
    v0, _ = derivatives(model, q, dt)
    energy = initial_energy(model, q[0], v0[0])
    out.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for substeps in (1, 2):
        result, pos, vel = rollout(lengths, q, controls, dt, substeps)
        old_baseline = next(r for r in old['runs'] if r['case'] == 'baseline' and r['substeps'] == substeps)
        if abs(result['speed_kmh']-old_baseline['speed_kmh']) > .00001:
            raise RuntimeError('Original baseline not reproduced')
        baseline[substeps] = (result, pos, vel)
        np.savez_compressed(out/f'baseline-sub{substeps}.npz', qpos=pos, qvel=vel, dt=dt, lengths=lengths)
    rows, states = [], {}
    for window in WINDOWS:
        for profile in PROFILES:
            case = f'{profile}-{round(window*1000)}ms'
            changed, angles = modify_wrist(model, q, dt, window, profile)
            velocity, _ = derivatives(model, changed, dt)
            if not np.array_equal(velocity[0], v0[0]) or initial_energy(model, changed[0], velocity[0]) != energy:
                raise RuntimeError('Unequal initial velocity or energy')
            demand = inverse_controls(model, changed, dt)
            pathways = [('demand', None, demand)]
            pathways += [('bounded', alpha, bounded_wrist_controls(controls, demand, alpha)) for alpha in ALPHAS]
            for pathway, alpha, applied in pathways:
                if pathway == 'bounded' and not np.array_equal(applied[:, :8], controls[:, :8]):
                    raise RuntimeError('Non-wrist forcing changed')
                tag = f'{case}-{pathway}-{alpha}'
                for substeps in (1, 2):
                    result, pos, vel = rollout(lengths, q, applied, dt, substeps)
                    b, bp, bv = baseline[substeps]
                    metrics = movement_metrics(model, bp, bv, pos, vel)
                    result.update(metrics)
                    result.update(case=tag, profile=profile, window_ms=window*1000,
                                  pathway=pathway, alpha=alpha, requested_endpoint_deg=2.,
                                  prescribed_extra_endpoint_rate_deg_s=6/window if profile == 'cubic_in' else 0.,
                                  discrete_requested_endpoint_rate_deg_s=float(np.rad2deg((angles[-1]-angles[-2])/dt)),
                                  paired_gain_kmh=result['speed_kmh']-b['speed_kmh'],
                                  positive_component_work_change_j=result['total_positive_component_work_j']-b['total_positive_component_work_j'],
                                  initial_energy_j=energy, initial_state_equal=True,
                                  budget_gate=gate(b, result),
                                  source_interval_count=len(q)-1)
                    rows.append(result)
                    states[tag, substeps] = (pos, vel, applied)
                    if profile == 'cubic_in' and window == .030 and pathway == 'demand':
                        original = next(r for r in old['runs'] if r['case'] == 'wrist_z_p2' and r['substeps'] == substeps)
                        if abs(result['speed_kmh']-original['speed_kmh']) > .00001:
                            raise RuntimeError('Original +1.665 candidate not reproduced')
                print(tag, round(rows[-1]['paired_gain_kmh'], 6), rows[-1]['budget_gate']['strict_model_budget_pass'], flush=True)
    decisions = []
    for tag in dict.fromkeys(r['case'] for r in rows):
        a, b = [r for r in rows if r['case'] == tag]
        bound = abs(baseline[1][0]['speed_kmh']-baseline[2][0]['speed_kmh']) + abs(a['speed_kmh']-b['speed_kmh'])
        resolution = abs(a['paired_gain_kmh']-b['paired_gain_kmh'])
        passes = all(r['budget_gate']['strict_model_budget_pass'] and r['pose_screen_pass'] and r['paired_gain_kmh'] > bound for r in (a,b)) and resolution <= .001
        decisions.append({'case': tag, 'conditional_model_candidate': bool(passes),
                          'gain_resolution_difference_kmh': resolution, 'absolute_resolution_sum_kmh': bound,
                          'fine_gain_kmh': b['paired_gain_kmh'], 'gain_at_least_1kmh': b['paired_gain_kmh'] >= 1})
    chosen = [d for d in decisions if d['conditional_model_candidate']]
    # Preserve diagnostic states without presenting the best failed case as success.
    diagnostics = {max(decisions, key=lambda x: x['fine_gain_kmh'])['case'], 'cubic_in-30ms-demand-None'}
    diagnostics.update(d['case'] for d in chosen)
    for tag in diagnostics:
        for substeps in (1,2):
            pos, vel, applied = states[tag,substeps]
            np.savez_compressed(out/f'{tag}-sub{substeps}.npz', qpos=pos, qvel=vel, controls=applied, dt=dt, lengths=lengths)
    report = {'pitch': '3034_1', 'source': str(source.resolve()), 'source_sha256': digest,
              'original_report': str(reference_report.resolve()), 'original_report_sha256': hashlib.sha256(reference_report.read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
              'scope': 'Fixed-gain waveform demand and actual bounded wrist-input rollouts; not real intervention validation.',
              'initial_energy_j': energy, 'baseline': {n: v[0] for n,v in baseline.items()},
              'runs': rows, 'decisions': decisions, 'conditional_candidates': chosen,
              'real_athlete_effect_verified': False, 'empirical_effect_95ci': None,
              'spin_or_command_effect_verified': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference-report', required=True)
    args = parser.parse_args()
    result = run_experiment(args.source, args.out, args.reference_report)
    print(json.dumps({'runs': len(result['runs']), 'conditional_candidates': result['conditional_candidates']}))
