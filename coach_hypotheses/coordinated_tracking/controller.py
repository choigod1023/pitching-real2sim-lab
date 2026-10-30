"""Reference-preserving computed-torque feedback with peak-bounded inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from capacity_budget_gate import gate, rollout
from coupled_payload import model_for
from joint_intervention import rotation_differences
from torque_arm import derivatives
from coach_hypotheses.gain_feasibility.profiles import profile_value, movement_metrics
from coach_hypotheses.gain_feasibility.shield import actual_accounting

GROUPS = {'shoulder_rotation': slice(3,6), 'elbow': slice(6,7),
          'marker_roll': slice(7,8), 'wrist_marker_rotation': slice(8,11)}
AMPLITUDES = (.5, 1., 2.)
PROFILES = ('cubic_in', 'smootherstep')
FREQUENCIES = (0., 5., 10., 20.)


def target_states(model, positions, velocities, dt, amplitude, profile):
    """Perturb actual baseline; zero amplitude preserves exact states/velocities."""
    q, v = np.asarray(positions), np.asarray(velocities)
    if q.ndim != 2 or q.shape[1] != model.nq or v.shape != (len(q), model.nv) or len(q) < 3:
        raise ValueError('Matching trajectories required')
    if not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite([dt, amplitude]).all() or dt <= 0:
        raise ValueError('Finite states, amplitude and positive dt required')
    if (len(q)-2)*dt <= .030:
        raise ValueError('30ms window must preserve initial samples')
    time = np.arange(len(q))*dt
    phase = np.clip((time-time[-1]+.030)/.030, 0, 1)
    angles = np.deg2rad(amplitude)*profile_value(phase, profile)
    if amplitude == 0:
        return q.copy(), v.copy(), np.zeros_like(v)
    target = q.copy()
    tangent = np.zeros(model.nv)
    for p, angle in zip(target, angles):
        tangent[10] = angle
        mujoco.mj_integratePos(model, p, tangent, 1.)
    original_rate, _ = derivatives(model, q, dt)
    target_rate, _ = derivatives(model, target, dt)
    target_velocity = v+(target_rate-original_rate)
    base_acceleration = np.gradient(v, dt, axis=0, edge_order=2)
    target_acceleration = np.gradient(target_velocity, dt, axis=0, edge_order=2)
    delta = np.empty_like(v)
    data = mujoco.MjData(model)
    for i in range(len(q)):
        data.qpos[:] = q[i]; data.qvel[:] = v[i]; data.qacc[:] = base_acceleration[i]
        mujoco.mj_inverse(model, data)
        base_force = data.qfrc_inverse.copy()
        data.qpos[:] = target[i]; data.qvel[:] = target_velocity[i]; data.qacc[:] = target_acceleration[i]
        mujoco.mj_inverse(model, data)
        delta[i] = data.qfrc_inverse-base_force
    if not np.array_equal(target[:2], q[:2]) or not np.array_equal(target_velocity[0], v[0]):
        raise RuntimeError('Initial target state changed')
    return target, target_velocity, delta


def force_limits(controls):
    """Use only applied intervals, excluding the unused terminal force."""
    c = np.asarray(controls, float)
    if c.ndim != 2 or c.shape[1] != 11 or len(c) < 2 or not np.isfinite(c).all():
        raise ValueError('Finite N by 11 controls required')
    return {'components': np.max(np.abs(c[:-1]),axis=0),
            'groups': {name: float(np.linalg.norm(c[:-1,part],axis=1).max()) for name,part in GROUPS.items()}}


def bound_force(force, limits):
    """Preserve per-component and existing joint-norm peak envelopes."""
    f = np.asarray(force, float)
    if f.shape != (11,) or not np.isfinite(f).all():
        raise ValueError('Finite 11-vector required')
    out = np.clip(f, -limits['components'], limits['components'])
    for name, part in GROUPS.items():
        norm = np.linalg.norm(out[part])
        if norm > limits['groups'][name]:
            out[part] *= limits['groups'][name]/norm
    return out


def controller_force(model, data, feedforward, delta, target_q, target_v, frequency, limits):
    """M(q)(omega^2 e + 2 omega edot) feedback across all eleven DOFs."""
    if not np.isfinite(frequency) or frequency < 0:
        raise ValueError('Nonnegative finite engineering feedback frequency required')
    if np.asarray(target_q).shape != (model.nq,) or np.asarray(target_v).shape != (model.nv,):
        raise ValueError('Model-sized targets required')
    mujoco.mj_forward(model, data)
    error = np.zeros(model.nv)
    if not np.array_equal(data.qpos, target_q):
        mujoco.mj_differentiatePos(model, error, 1., data.qpos, target_q)
    omega = 2*np.pi*frequency
    correction = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, correction, omega**2*error+2*omega*(target_v-data.qvel))
    requested = np.asarray(feedforward)+np.asarray(delta)+correction
    return bound_force(requested, limits), error, correction


def coordinated_rollout(lengths, baseline_pos, baseline_vel, controls, target_q, target_v, delta, dt, substeps, frequency):
    """Update once per source interval; hold actual force across RK4 substeps."""
    if substeps not in (1,2) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive dt and declared substeps required')
    model = model_for(lengths, dt/substeps)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    arrays = [baseline_pos,baseline_vel,controls,target_q,target_v,delta]
    if len({len(a) for a in arrays}) != 1 or len(controls) < 3 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Matching finite trajectories required')
    limits = force_limits(controls)
    data = mujoco.MjData(model)
    data.qpos[:] = baseline_pos[0]; data.qvel[:] = baseline_vel[0]
    mujoco.mj_forward(model,data)
    positions, velocities = [data.qpos.copy()], [data.qvel.copy()]
    applied, source_controls, corrections = [], [], []
    for i, ff in enumerate(controls[:-1]):
        force, _, correction = controller_force(model,data,ff,delta[i],target_q[i],target_v[i],frequency,limits)
        source_controls.append(force.copy()); corrections.append(correction)
        for _ in range(substeps):
            data.qfrc_applied[:] = force
            mujoco.mj_step(model,data)
            if data.warning.number.any() or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Unstable coordinated rollout')
            positions.append(data.qpos.copy());velocities.append(data.qvel.copy());applied.append(force.copy())
    positions, velocities, applied = map(np.asarray,(positions,velocities,applied))
    metrics = actual_accounting(lengths,positions,velocities,applied,dt/substeps)
    at_source = positions[::substeps]
    pose_errors = np.empty((len(at_source),model.nv))
    for i,(actual,requested) in enumerate(zip(at_source,target_q)):
        mujoco.mj_differentiatePos(model,pose_errors[i],1.,actual,requested)
    metrics.update(substeps=substeps,
        target_rotation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(pose_errors[:,3:]**2)))),
        target_wrist_rotation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(pose_errors[:,8:11]**2,axis=1))))),
        target_position_rmse_mm=float(np.sqrt(np.mean(np.sum(pose_errors[:,:3]**2,axis=1)))*1000),
        feedback_peak_absolute_force=np.max(np.abs(corrections),axis=0).tolist(),
        state_resets=0, source_control_dt_s=dt)
    trace = {'qpos':positions,'qvel':velocities,'applied_loads':applied,
             'source_controls':np.asarray(source_controls),'target_q':target_q,'target_v':target_v,
             'lengths':np.asarray(lengths),'dt':dt/substeps,'source_dt':dt}
    return metrics,positions,velocities,trace


def run_experiment(source, reference_report, out):
    """Prespecified grid with zero-target identity controls and strict acceptance."""
    source,reference_report,out = Path(source),Path(reference_report),Path(out)
    reference = json.loads(reference_report.read_text(encoding='utf-8'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != reference['source_sha256']:
        raise ValueError('Input hash mismatch')
    from joint_intervention import inverse_controls
    z = np.load(source)
    q,dt,lengths = z['qpos'],float(z['dt']),z['lengths']
    model = model_for(lengths,dt)
    controls = inverse_controls(model,q,dt)
    baseline = {}
    identity = []
    out.mkdir(parents=True,exist_ok=True)
    for sub in (1,2):
        b,bp,bv = rollout(lengths,q,controls,dt,sub)
        previous = reference['baseline'][str(sub)]
        np.testing.assert_allclose(b['speed_kmh'],previous['speed_kmh'],atol=1e-10,rtol=0)
        baseline[sub] = b,bp,bv
        zero_q,zero_v,zero_delta = target_states(model,bp,bv,dt,0.,'cubic_in')
        for frequency in FREQUENCIES:
            result,pos,vel,_ = coordinated_rollout(lengths,bp,bv,controls,zero_q,zero_v,zero_delta,dt,sub,frequency)
            if not np.array_equal(pos[::sub],bp) or not np.array_equal(vel[::sub],bv):
                raise RuntimeError('Zero-target controller changed baseline')
            identity.append({'frequency_hz':frequency,'substeps':sub,'exact_baseline_states':True,
                             'budget_pass':gate(b,result)['strict_model_budget_pass']})
            if not identity[-1]['budget_pass']:
                raise RuntimeError('Zero-target budget mismatch')
    rows = []
    for profile in PROFILES:
        for amplitude in AMPLITUDES:
            targets = {s:target_states(model,baseline[s][1],baseline[s][2],dt,amplitude,profile) for s in (1,2)}
            for frequency in FREQUENCIES:
                tag = f'{profile}-{amplitude}deg-{frequency}Hz'
                for sub in (1,2):
                    b,bp,bv = baseline[sub]
                    tq,tv,delta = targets[sub]
                    result,pos,vel,trace = coordinated_rollout(lengths,bp,bv,controls,tq,tv,delta,dt,sub,frequency)
                    result.update(movement_metrics(model,bp,bv,pos[::sub],vel[::sub]))
                    result.update(case=tag,profile=profile,requested_angle_deg=amplitude,frequency_hz=frequency,
                        paired_gain_kmh=result['speed_kmh']-b['speed_kmh'],budget_gate=gate(b,result),
                        positive_component_work_change_j=result['total_positive_component_work_j']-b['total_positive_component_work_j'],
                        initial_state_equal=bool(np.array_equal(pos[0],bp[0]) and np.array_equal(vel[0],bv[0])))
                    rows.append(result)
                    np.savez_compressed(out/f'{tag}-sub{sub}.npz',**trace)
                print(tag,round(rows[-1]['paired_gain_kmh'],6),round(rows[-1]['max_rotation_change_deg'],3),rows[-1]['budget_gate']['strict_model_budget_pass'],flush=True)
    decisions=[]
    for tag in dict.fromkeys(r['case'] for r in rows):
        a,b=[r for r in rows if r['case']==tag]
        discrepancy=abs(a['paired_gain_kmh']-b['paired_gain_kmh'])
        bound=abs(baseline[1][0]['speed_kmh']-baseline[2][0]['speed_kmh'])+abs(a['speed_kmh']-b['speed_kmh'])
        passed=all(r['initial_state_equal'] and r['budget_gate']['strict_model_budget_pass'] and r['pose_screen_pass'] and r['paired_gain_kmh']>bound for r in (a,b)) and discrepancy<=.001
        decisions.append({'case':tag,'conditional_model_candidate':bool(passed),'fine_gain_kmh':b['paired_gain_kmh'],
                          'gain_resolution_difference_kmh':discrepancy,'absolute_resolution_sum_kmh':bound})
    report={'pitch':'3034_1','source':str(source.resolve()),'source_sha256':digest,
            'reference_report':str(reference_report.resolve()),'reference_report_sha256':hashlib.sha256(reference_report.read_bytes()).hexdigest(),
            'protocol_sha256':hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'baseline':{s:v[0] for s,v in baseline.items()},'zero_target_checks':identity,
            'runs':rows,'decisions':decisions,'conditional_candidates':[d for d in decisions if d['conditional_model_candidate']],
            'real_athlete_effect_verified':False,'empirical_effect_95ci':None,'spin_or_command_effect_verified':False,
            'scope':'Mass-coupled engineering feedback across free-root/arm/hand coordinates, not validated muscle or anatomical control.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--reference-report',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();run_experiment(args.source,args.reference_report,args.out)
