"""Conditional baseball flight, not validated individual coaching predictions.

Coordinates: x lateral, y forward, z up; plate is y=distance.
Constant Cd and Cl are declared scenario assumptions, not fitted ball properties.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def flight(position, velocity, spin_axis=(1, 0, 0), cl=.15, cd=.35,
           distance=16.8, dt=.002, gravity=9.80665):
    p, v, axis = map(lambda a: np.asarray(a, float), (position, velocity, spin_axis))
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (p, v, axis)):
        raise ValueError('Finite three-vectors required')
    if not all(np.isfinite([cl, cd, distance, dt, gravity])) or not (0 <= cl <= .5 and 0 <= cd <= 1 and 0 < dt <= .02 and gravity >= 0):
        raise ValueError('Invalid physical scenario')
    if distance <= p[1] or v[1] <= 0 or (cl and np.linalg.norm(axis) == 0):
        raise ValueError('Forward release and nonzero lift axis required')
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) else axis
    # rho=1.225 kg/m3, radius=.0366 m, mass=.145 kg, constant spin direction.
    k = .5 * 1.225 * np.pi * .0366**2 / .145
    def rhs(s):
        vel = s[3:]; speed = np.linalg.norm(vel)
        # Effective transverse lift: gyro-aligned spin produces zero lift.
        acc = -k*cd*speed*vel + k*cl*speed*np.cross(axis, vel)
        acc[2] -= gravity
        return np.r_[vel, acc]
    s = np.r_[p, v]; path = [s.copy()]; elapsed = 0.
    for _ in range(int(3/dt)):
        a = rhs(s); b = rhs(s+dt*a/2); c = rhs(s+dt*b/2); d = rhs(s+dt*c)
        nxt = s + dt*(a+2*b+2*c+d)/6
        if nxt[1] >= distance:
            f = (distance-s[1])/(nxt[1]-s[1])
            s = s + f*(nxt-s)
            if s[2] < 0: raise ValueError('Ground reached before plate; scenario excluded')
            path.append(s)
            return np.array(path), elapsed + f*dt
        if nxt[2] < 0:
            raise ValueError('Ground reached before plate; scenario excluded')
        s = nxt; path.append(s.copy()); elapsed += dt
    raise ValueError('Plate not reached within 3 seconds')


def evaluate(states, target=(0, .9), cl=.15, cd=.35, distance=16.8,
             jitter_deg=.15, samples=33, seed=20261026, dt=.002, axis=(1, 0, 0), elevation_offset_deg=0):
    if len(target) != 2 or not np.isfinite(target).all() or not 0 <= jitter_deg <= 2 or not 2 <= samples <= 1000:
        raise ValueError('Invalid target or engineering perturbation settings')
    target = np.array(target)
    from scipy.spatial.transform import Rotation
    if not np.isfinite(elevation_offset_deg) or abs(elevation_offset_deg)>30:
        raise ValueError('Common diagnostic direction offset must be within 30 degrees')
    alignment = Rotation.from_rotvec([np.deg2rad(elevation_offset_deg), 0, 0])
    # Common perturbations, not learned athlete variability or confidence intervals.
    noise = np.random.default_rng(seed).normal(0, np.deg2rad(jitter_deg), (samples, 3))
    records = []
    for state in states:
        row = dict(state)
        row['release_speed_kmh'] = float(np.linalg.norm(state['velocity_m_s'])*3.6)
        try:
            p, v = np.array(state['position_m']), alignment.apply(state['velocity_m_s'])
            path, time = flight(p, v, axis, cl, cd, distance, dt)
            try:
                no_lift, _ = flight(p, v, axis, 0, cd, distance, dt)
                lift_displacement = ((path[-1, [0, 2]]-no_lift[-1, [0, 2]])*100).tolist()
            except ValueError:
                lift_displacement = None  # Counterfactual ground impact does not invalidate actual scenario.
            endpoints = []; failed = 0
            for rot in Rotation.from_rotvec(noise):
                try:
                    perturbed, _ = flight(p, rot.apply(v), axis, cl, cd, distance, dt)
                    endpoints.append(perturbed[-1, [0, 2]])
                except ValueError:
                    failed += 1
            ends = np.array(endpoints)
            row.update(status='ok', release_speed_kmh=float(np.linalg.norm(v)*3.6),
                plate_speed_kmh=float(np.linalg.norm(path[-1, 3:])*3.6),
                plate_xz_m=path[-1, [0, 2]].tolist(), flight_s=time,
                target_error_cm=float(np.linalg.norm(path[-1, [0, 2]]-target)*100),
                lift_displacement_cm=lift_displacement,
                perturbation_mean_error_cm=float(np.linalg.norm(ends-target, axis=1).mean()*100) if len(ends) else None,
                perturbation_spread_cm=float(np.sqrt(np.mean(np.sum((ends-ends.mean(0))**2, axis=1)))*100) if len(ends) else None,
                perturbation_failed=failed, endpoints_xz_m=ends.tolist(), path_m=path[:, :3].tolist())
        except ValueError as error:
            row.update(status='excluded', reason=str(error))
        row['human_trial_ready'] = False  # No intervention validation in current model.
        records.append(row)
    baseline = records[0]
    for row in records:
        if row['status'] == baseline['status'] == 'ok':
            row['speed_change_kmh'] = row['release_speed_kmh']-baseline['release_speed_kmh']
            row['target_error_reduction_cm'] = baseline['target_error_cm']-row['target_error_cm']
    return dict(assumptions=dict(target_xz_m=target.tolist(), cl=cl, cd=cd, spin_axis=list(axis),
        distance_m=distance, jitter_deg=jitter_deg, samples=samples, seed=seed, dt_s=dt,
        common_direction_offset_deg=elevation_offset_deg,
        direction_offset_scope='User hypothetical rotation of all release velocities, not observed correction or joint action',
        uncertainty='engineering perturbations, not athlete distribution or confidence interval',
        quality='lift displacement relative to same release without lift; not Stuff+ or quality gain'),
        records=records, human_trial_ready=False)


def build_states(source, output):
    """Generate actual forward arm states for small marker-frame joint changes."""
    import mujoco
    from coupled_payload import model_for
    from constrained_motion_search import modify, endpoints, PARAMETER_NAMES
    from joint_intervention import inverse_controls, rotation_differences
    from torque_arm import derivatives
    from trace_error_audit import integrate_inputs
    from capacity_budget_gate import split_linear_power
    source, output = Path(source), Path(output); output.mkdir(parents=True, exist_ok=True)
    z = np.load(source); q = z['qreference']; dt = float(z['dt']); lengths = z['lengths']
    model = model_for(lengths, dt)
    records = []; reference = None; origin = None; basis = None
    for i, (label, index, amount) in enumerate([('baseline', 0, 0), ('elbow_minus', 2, -.5),
        ('elbow_plus', 2, .5), ('wrist_minus', 3, -.5), ('wrist_plus', 3, .5)]):
        x = np.zeros(6); x[index] = amount
        cq = modify(model, q, dt, x)
        # Preserve audited padded baseline inputs. Add only derivative of change.
        changed_load = inverse_controls(model, cq, dt)
        original_load = inverse_controls(model, q, dt)
        loads = z['controls'] + changed_load-original_load
        pos, vel, _ = integrate_inputs(model, q[0], z['vreference'][0], loads, dt, 1, 'foh')
        points, velocity = endpoints(model, pos, vel)
        if reference is None:
            reference = pos
            forward = velocity.copy(); forward[2] = 0; forward /= np.linalg.norm(forward)
            up = np.array([0., 0., 1.]); right = np.cross(forward, up)
            basis = np.array([right, forward, up]); origin = points[-1].copy()
        positive = np.zeros(model.nv); negative = positive.copy()
        powers = loads*vel
        for a, b in zip(powers[:-1], powers[1:]):
            pp, nn, _ = split_linear_power(a, b, dt); positive += pp; negative += nn
        budget = np.r_[positive, negative, np.max(np.abs(loads), axis=0)]
        if i == 0: base_budget = budget.copy()
        budget_pass = bool(np.all(budget <= base_budget + 1e-6))
        np.savez_compressed(output/f'{label}.npz', qpos=pos, qvel=vel, qreference=q,
                            lengths=lengths, dt=dt, controls=loads, parameters=x)
        records.append(dict(name=label, position_m=(basis@(points[-1]-origin)+[0, 0, 1.8]).tolist(),
            velocity_m_s=(basis@velocity).tolist(), joint_parameters=dict(zip(PARAMETER_NAMES, x.tolist())),
            max_joint_rotation_deg=rotation_differences(model, reference, pos),
            max_ball_shift_mm=float(np.linalg.norm(points-endpoints(model, reference, vel)[0], axis=1).max()*1000),
            sampled_budget_pass=budget_pass, positive_work_j=float(positive.sum()),
            work_change_j=float(positive.sum()-base_budget[:model.nv].sum()),
            state_path=str((output/f'{label}.npz').resolve())))
    data = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        coordinates='Common baseline horizontal velocity frame; assumed release height 1.8m, distance16.8m; not surveyed field alignment',
        dynamics='FOH RK4; inverse force corrections, not fixed-force intervention; sampled work diagnostic, not physiological capacity',
        spin='Unmeasured: user-specified flight lift and axis, not generated by joint changes', states=records)
    (output/'states.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    return data


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path); p.add_argument('--states', type=Path)
    p.add_argument('--output', type=Path, default=Path('artifacts/outcome-simulator')); p.add_argument('--gui', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if a.source: data = build_states(a.source, a.output)
    elif a.states: data = json.loads(a.states.read_text(encoding='utf-8'))
    else: p.error('Supply --source audited state NPZ or --states states.json')
    if a.gui:
        from pitch_outcomes_gui import launch
        launch(data, a.output)
    else:
        result = evaluate(data['states']); result['provenance'] = {k:v for k,v in data.items() if k!='states'}
        (a.output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps([{k:r.get(k) for k in ['name','status','release_speed_kmh','target_error_cm','speed_change_kmh','target_error_reduction_cm','sampled_budget_pass']} for r in result['records']], indent=2))


if __name__ == '__main__': main()
