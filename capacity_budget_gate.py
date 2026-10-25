"""Actual-trajectory mechanical budget diagnostics; not physiological capacity."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from coupled_payload import model_for
from torque_arm import derivatives


def shift_controls(loads, columns, shift):
    """Shift selected components with zero padding, never wrap or rescale."""
    if not isinstance(shift, (int, np.integer)):
        raise ValueError('Integer sample shift required')
    out = np.array(loads, dtype=float, copy=True)
    if shift == 0:
        return out
    out[:, columns] = 0
    if abs(shift) < len(out):
        if shift > 0:
            out[shift:, columns] = loads[:-shift, columns]
        else:
            out[:shift, columns] = loads[-shift:, columns]
    return out


def split_linear_power(p0, p1, dt):
    """Integrate positive/negative parts of linearly interpolated power exactly."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    net = (p0+p1)*dt/2
    positive = (np.maximum(p0, 0)+np.maximum(p1, 0))*dt/2
    cross = p0*p1 < 0
    if np.any(cross):
        positive[cross] = dt*np.maximum(p0[cross], p1[cross])**2/(2*np.abs(p1[cross]-p0[cross]))
    return positive, positive-net, net


def shift_applied_intervals(loads, columns, shift):
    """Keep unused terminal reference force out of all applied intervals."""
    controls = np.array(loads, dtype=float, copy=True)
    controls[:-1] = shift_controls(controls[:-1], columns, shift)
    return controls


def rollout(lengths, q, controls, dt, substeps=1):
    if substeps not in (1, 2):
        raise ValueError('Use predeclared one or two substeps')
    m = model_for(lengths, dt/substeps)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    if controls.shape != (len(q), m.nv) or not np.isfinite(controls).all():
        raise ValueError('Invalid controls')
    wrist = m.joint('wrist_marker_rotation')
    if int(wrist.dofadr[0]) != 8 or int(wrist.type[0]) != int(mujoco.mjtJoint.mjJNT_BALL):
        raise ValueError('Unexpected wrist model')
    vel, _ = derivatives(m, q, dt)
    d = mujoco.MjData(m)
    d.qpos[:] = q[0]; d.qvel[:] = vel[0]
    mujoco.mj_forward(m,d); mujoco.mj_energyPos(m,d); mujoco.mj_energyVel(m,d)
    initial_energy = float(d.energy.sum())
    positive, negative, net = np.zeros(m.nv), np.zeros(m.nv), np.zeros(m.nv)
    positions, velocities = [d.qpos.copy()], [d.qvel.copy()]
    for force in controls[:-1]:
        d.qfrc_applied[:] = force
        for _ in range(substeps):
            p0 = force*d.qvel.copy()
            mujoco.mj_step(m, d)
            if d.warning.number.any() or not np.isfinite(d.qvel).all():
                raise RuntimeError('Unstable rollout')
            a, b, c = split_linear_power(p0, force*d.qvel, m.opt.timestep)
            positive += a; negative += b; net += c
        positions.append(d.qpos.copy()); velocities.append(d.qvel.copy())
    mujoco.mj_forward(m, d)
    mujoco.mj_energyPos(m,d); mujoco.mj_energyVel(m,d)
    energy_change = float(d.energy.sum())-initial_energy
    jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
    mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
    result = {'speed_kmh': float(np.linalg.norm(jp@d.qvel)*3.6),
              'positive_work_per_dof_j': positive.tolist(),
              'negative_work_magnitude_per_dof_j': negative.tolist(),
              'net_work_per_dof_j': net.tolist(),
              'total_positive_component_work_j': float(positive.sum()),
              'total_negative_component_work_magnitude_j': float(negative.sum()),
              'total_net_work_j': float(net.sum()),
              'mechanical_energy_change_j': energy_change,
              'energy_change_minus_applied_work_j': energy_change-float(net.sum()),
              'shoulder_translation_net_work_j': float(net[:3].sum()),
              'shoulder_rotation_net_work_j': float(net[3:6].sum()),
              'shoulder_translation_positive_component_work_j': float(positive[:3].sum()),
              'shoulder_rotation_positive_component_work_j': float(positive[3:6].sum()),
              'peak_abs_generalized_force_per_dof': np.max(np.abs(controls[:-1]),axis=0).tolist(),
              'peak_joint_torque_norm_nm': {name: float(np.linalg.norm(controls[:-1,sl],axis=1).max())
                  for name,sl in [('shoulder_rotation',slice(3,6)),('elbow',slice(6,7)),('marker_roll',slice(7,8)),('wrist_marker_rotation',slice(8,11))]},
              'substeps': substeps, 'dt_s': dt/substeps}
    return result, np.asarray(positions), np.asarray(velocities)


def gate(baseline, candidate, epsilon=1e-9):
    """Strict mechanical budget screen; epsilon is floating comparison only."""
    violations = []
    for key in ['positive_work_per_dof_j','negative_work_magnitude_per_dof_j','peak_abs_generalized_force_per_dof']:
        diff = np.asarray(candidate[key])-np.asarray(baseline[key])
        for i in np.flatnonzero(diff > epsilon):
            violations.append({'metric':key,'dof':int(i),'excess':float(diff[i])})
    for key in ['shoulder_translation_net_work_j','shoulder_rotation_net_work_j']:
        diff = candidate[key]-baseline[key]
        if diff > epsilon:
            violations.append({'metric':key,'excess':float(diff)})
    for joint,value in candidate['peak_joint_torque_norm_nm'].items():
        diff = value-baseline['peak_joint_torque_norm_nm'][joint]
        if diff > epsilon:
            violations.append({'metric':'peak_joint_torque_norm_nm','joint':joint,'excess':float(diff)})
    return {'strict_model_budget_pass':not violations,'violations':violations,
            'floating_comparison_epsilon':epsilon,'physiological_capacity_validated':False}


def run(source, out):
    source, out = Path(source), Path(out)
    state = np.load(source)
    q, lengths, loads, dt = state['qpos'],state['lengths'],state['required_loads'],float(state['dt'])
    shifts = [0,-1,1,-round(.001/dt),round(.001/dt)]
    out.mkdir(parents=True,exist_ok=True)
    runs = []
    baselines = {}
    for substeps in [1,2]:
        for shift in shifts:
            controls = shift_applied_intervals(loads,slice(8,11),shift)
            result, pos, vel = rollout(lengths,q,controls,dt,substeps)
            result.update(shift_samples=shift,shift_ms=shift*dt*1000)
            if shift == 0:
                baselines[substeps] = (result,pos)
            baseline, baseline_pos = baselines[substeps]
            result['speed_difference_from_baseline_kmh'] = result['speed_kmh']-baseline['speed_kmh']
            result['budget_gate'] = gate(baseline,result)
            m = model_for(lengths,dt)
            dq = np.zeros(m.nv)
            diffs = []
            for a,b in zip(baseline_pos,pos):
                mujoco.mj_differentiatePos(m,dq,1.,a,b)
                diffs.append(dq.copy())
            diffs = np.asarray(diffs)
            result['max_rotation_difference_from_baseline_deg'] = {name:float(np.rad2deg(np.linalg.norm(diffs[:,sl],axis=1)).max())
                for name,sl in [('shoulder',slice(3,6)),('elbow',slice(6,7)),('marker_roll',slice(7,8)),('wrist_marker',slice(8,11))]}
            np.savez_compressed(out/f'state-shift{shift}-sub{substeps}.npz',qpos=pos,qvel=vel,applied_loads=controls,dt=dt,lengths=lengths)
            runs.append(result)
    resolution = []
    for shift in shifts:
        a,b = [r for r in runs if r['shift_samples']==shift]
        resolution.append({'shift_samples':shift,'speed_difference_fine_minus_coarse_kmh':b['speed_kmh']-a['speed_kmh'],
            'max_positive_work_component_difference_j':float(np.max(np.abs(np.asarray(a['positive_work_per_dof_j'])-b['positive_work_per_dof_j']))),
            'same_strict_gate_result':a['budget_gate']['strict_model_budget_pass']==b['budget_gate']['strict_model_budget_pass']})
    report = {'pitch':'3034_1','source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'protocol':'All 5 predefined shifts; wrist marker-frame 3-axis applied torque shifted together on actually applied intervals only, zero padded. Last unused reference load is excluded before shift. Same initial state and all other controls unchanged. No optimization or condition selection.',
        'work_definition':'Per-DOF applied force times actual generalized velocity, integrated by endpoint trapezoid with linear zero-crossing split. Positive-component work is coordinate-dependent and is NOT muscle work. Controls held constant across substeps.',
        'dof_order':['shoulder_tx','shoulder_ty','shoulder_tz','shoulder_rx','shoulder_ry','shoulder_rz','elbow','marker_roll','wrist_rx','wrist_ry','wrist_rz'],
        'limitations':['Generic inertia and assumed 15cm grip','No muscle force-velocity or anatomical ROM limits','Same applied force does not guarantee same work','Engineering timing changes are not validated comfortable/feasible movements','Budget failure flags are strict; 1e-9 is not an uncertainty bound','Conditional ball speed is not validated actual pitch-speed improvement'],
        'optimization_ready':False,'runs':runs,'resolution_comparison':resolution}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz')
    p.add_argument('--out',default='artifacts/capacity-gate')
    args=p.parse_args(); report=run(args.source,args.out)
    print(json.dumps({'runs':[{k:r[k] for k in ['shift_samples','substeps','speed_kmh','speed_difference_from_baseline_kmh','total_positive_component_work_j','shoulder_translation_net_work_j']} for r in report['runs']], 'resolution':report['resolution_comparison']},indent=2))
