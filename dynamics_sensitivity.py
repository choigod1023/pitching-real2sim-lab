"""Engineering stress tests, not anthropometric confidence intervals or gains."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from torque_arm import derivatives, make_model


def demand(model, q, dt):
    velocity, acceleration = derivatives(model, q, dt)
    data = mujoco.MjData(model)
    forces, energy, wrist = [], [], []
    for pos, vel, acc in zip(q, velocity, acceleration):
        data.qpos[:] = pos
        data.qvel[:] = vel
        data.qacc[:] = acc
        mujoco.mj_inverse(model, data)
        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)
        forces.append(data.qfrc_inverse.copy())
        energy.append(data.energy.sum())
        wrist.append(data.site_xpos[model.site('wrist').id].copy())
    forces = np.array(forces)
    # Generalized forces and generalized velocities are conjugate; do not mix
    # local free-joint angular velocity with a world-frame torque vector.
    powers = {'shoulder_translation': np.sum(forces[:, :3]*velocity[:, :3], axis=1),
              'shoulder_rotation': np.sum(forces[:, 3:6]*velocity[:, 3:6], axis=1),
              'elbow': forces[:, 6]*velocity[:, 6]}
    powers['total'] = sum(powers.values())
    works = {name: {'net_j': float(np.trapezoid(p, dx=dt)),
                    'positive_j': float(np.trapezoid(np.maximum(p, 0), dx=dt)),
                    'negative_j': float(np.trapezoid(np.minimum(p, 0), dx=dt))}
             for name, p in powers.items()}
    report = {'work': works, 'energy_change_j': float(energy[-1]-energy[0]),
              'work_energy_residual_j': works['total']['net_j']-float(energy[-1]-energy[0]),
              'peak_shoulder_force_n': float(np.linalg.norm(forces[:, :3], axis=1).max()),
              'peak_shoulder_torque_nm': float(np.linalg.norm(forces[:, 3:6], axis=1).max()),
              'peak_elbow_torque_nm': float(np.abs(forces[:, 6]).max())}
    return report, forces, velocity, np.array(wrist)


def fixed_load_rollout(model, q, velocity, reference_wrist, forces):
    """Apply the SAME nominal loads to a perturbed model, without compensation."""
    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    data.qvel[:] = velocity[0]
    errors = [0.]
    for i in range(len(q)-1):
        data.qfrc_applied[:] = forces[i]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if data.warning.number.any() or not np.isfinite(data.qpos).all():
            return {'stable': False, 'wrist_rmse_mm': None, 'wrist_final_error_mm': None}
        errors.append(np.linalg.norm(data.site_xpos[model.site('wrist').id]-reference_wrist[i+1]))
    return {'stable': True, 'wrist_rmse_mm': float(np.sqrt(np.mean(np.square(errors)))*1000),
            'wrist_final_error_mm': float(errors[-1]*1000)}


def run(source, out):
    source, out = Path(source), Path(out)
    meta = json.loads((source/'report.json').read_text(encoding='utf-8'))
    input_file = source/f"rollout-{len(meta['runs'])-1}.npz"
    with np.load(input_file) as arrays:
        q = arrays['reference_qpos'].copy()
        dt = float(np.diff(arrays['time'])[0])
    lengths = meta['segment_lengths_m']
    nominal = make_model(lengths, dt)
    _, nominal_forces, velocity, wrist = demand(nominal, q, dt)
    scenarios = [('nominal', 1., 1.), ('mass_0.8', .8, 1.), ('mass_1.2', 1.2, 1.),
                 ('radius_0.8', 1., .8), ('radius_1.2', 1., 1.2),
                 ('mass_radius_0.8', .8, .8), ('mass_radius_1.2', 1.2, 1.2)]
    rows = []
    for name, mass_scale, radius_scale in scenarios:
        model = make_model(lengths, dt, masses=np.array([2.14154, 1.280182])*mass_scale,
                           radii=np.array([.04, .03])*radius_scale)
        required, _, _, _ = demand(model, q, dt)
        rows.append({'scenario': name, 'mass_scale': mass_scale, 'radius_scale': radius_scale,
                     'required_loads_for_same_reference': required,
                     'fixed_nominal_loads': fixed_load_rollout(model, q, velocity, wrist, nominal_forces)})
    report = {'pitch': meta['pitch'], 'source_sha256': hashlib.sha256(input_file.read_bytes()).hexdigest(),
              'source': str(input_file.resolve()), 'dt_s': dt, 'scenarios': rows,
              'interpretation': 'Plus/minus 20 percent is an engineering stress test, not a measured biological interval. Recomputed loads measure demand; fixed-load rollouts test sensitivity without granting extra ability.',
              'limitations': ['Two-link model excludes hand, ball, muscles, contacts and anatomical limits.',
                             'External shoulder wrench substitutes for the unmodeled torso.',
                             'Positive mechanical work is not metabolic cost or measured athlete capacity.',
                             'No ball speed, improvement or coaching recommendation.']}
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Directory produced by torque_arm.py')
    parser.add_argument('--out', type=Path, default=Path('artifacts/dynamics-sensitivity'))
    args = parser.parse_args()
    result = run(args.source, args.out)
    print(json.dumps(result, indent=2))
