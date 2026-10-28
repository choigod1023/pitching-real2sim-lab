"""Instantaneous muscle capacity audit; fixed chest, no ball or forward tracking."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear


def reduced_coordinates(model):
    """Return affine q = offset + S x for active linear joint equalities."""
    if model.nq != model.nv or np.any(model.jnt_type != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('This audit supports hinge-only models')
    relations = {}
    for k in range(model.neq):
        if not model.eq_active0[k]:
            continue
        if model.eq_type[k] != mujoco.mjtEq.mjEQ_JOINT:
            raise ValueError('Only joint equalities are supported')
        if np.any(np.abs(model.eq_data[k, 2:5]) > 1e-12):
            raise ValueError('Nonlinear joint equality needs a configuration-dependent Jacobian')
        a, b = int(model.eq_obj1id[k]), int(model.eq_obj2id[k])
        if a in relations:
            raise ValueError('Multiple equalities constrain the same joint')
        relations[a] = (b, float(model.eq_data[k, 0]), float(model.eq_data[k, 1]))
    independent = [j for j in range(model.njnt) if j not in relations]
    S = np.zeros((model.nv, len(independent)))
    offset = np.zeros(model.nq)
    done = set(independent)
    for c, j in enumerate(independent):
        S[model.jnt_dofadr[j], c] = 1
    visiting = set()

    def fill(j):
        if j in done:
            return
        if j in visiting:
            raise ValueError('Cyclic joint equalities')
        visiting.add(j)
        b, constant, slope = relations[j]
        if b >= 0:
            fill(b)
            S[model.jnt_dofadr[j]] = slope * S[model.jnt_dofadr[b]]
            constant += slope * offset[model.jnt_qposadr[b]]
        offset[model.jnt_qposadr[j]] = constant
        visiting.remove(j)
        done.add(j)

    for j in relations:
        fill(j)
    return S, offset, independent


def muscle_force_map(model, data):
    """At fixed q,qvel obtain affine actuator forces f0 + activation * delta."""
    if model.na != model.nu or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_MUSCLE):
        raise ValueError('Expected one activation per muscle actuator')
    R = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(R, data.actuator_moment, data.moment_rownnz,
                           data.moment_rowadr, data.moment_colind)
    old_act = data.act.copy()
    data.act[:] = 0
    mujoco.mj_fwdActuation(model, data)
    f0 = data.actuator_force.copy()
    data.act[:] = 1
    mujoco.mj_fwdActuation(model, data)
    delta = data.actuator_force.copy() - f0
    data.act[:] = old_act
    mujoco.mj_fwdActuation(model, data)
    return R, f0, delta


def capacity_for_trajectory(model, qpos, dt):
    """Differentiate and audit every frame, including all independent finger DOFs.

    Inverse dynamics includes contacts. Acceleration residual is a local mass
    projection with constraint forces held fixed, NOT a forward rollout result.
    """
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 3:
        raise ValueError('Need at least three frames with model.nq coordinates')
    if not np.isfinite(qpos).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Finite coordinates and positive dt required')
    S, offset, independent = reduced_coordinates(model)
    reconstructed = offset + qpos[:, model.jnt_qposadr[independent]] @ S.T
    equality_error = float(np.max(np.abs(qpos - reconstructed)))
    if equality_error > 1e-7:
        raise ValueError(f'Trajectory violates joint equalities: {equality_error}')
    qvel = np.gradient(qpos, dt, axis=0, edge_order=2)
    qacc = np.gradient(qvel, dt, axis=0, edge_order=2)
    d = mujoco.MjData(model)
    rows = []
    activations = []
    for frame, q in enumerate(qpos):
        mujoco.mj_resetData(model, d)
        d.qpos[:] = q
        d.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, d)
        R, f0, delta = muscle_force_map(model, d)
        d.qacc[:] = qacc[frame]
        mujoco.mj_inverse(model, d)
        target = S.T @ d.qfrc_inverse
        A = (S.T @ R.T) * delta[None, :]
        b = target - S.T @ R.T @ f0
        solution = lsq_linear(A, b, bounds=(0, 1), method='bvls', tol=1e-8, max_iter=300)
        residual = A @ solution.x - b
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, d, mass)
        acceleration_residual = np.linalg.solve(S.T @ mass @ S, residual)
        lower = model.jnt_range[:, 0] - q[model.jnt_qposadr]
        upper = q[model.jnt_qposadr] - model.jnt_range[:, 1]
        violation = np.maximum(np.maximum(lower, upper), 0) * model.jnt_limited
        rows.append({
            'frame': frame, 'solver_success': bool(solution.success),
            'torque_residual_rmse_nm': float(np.sqrt(np.mean(residual**2))),
            'torque_residual_max_nm': float(np.max(np.abs(residual))),
            'required_reduced_torque_l2_nm': float(np.linalg.norm(target)),
            'passive_reduced_torque_l2_nm': float(np.linalg.norm(S.T @ d.qfrc_passive)),
            'constraint_reduced_torque_l2_nm': float(np.linalg.norm(S.T @ d.qfrc_constraint)),
            'activation_max': float(np.max(solution.x)),
            'activation_saturated_fraction': float(np.mean(solution.x > .999)),
            'contact_count': int(d.ncon),
            'contact_max_penetration_m': float(max([0.] + [-float(c.dist) for c in d.contact])),
            'joint_range_max_violation_rad': float(np.max(violation)),
            'local_acceleration_residual_rmse_rad_s2': float(np.sqrt(np.mean(acceleration_residual**2))),
        })
        activations.append(solution.x.tolist())
    return {
        'scope': 'Fixed-chest instantaneous muscle capacity; no ball; not forward feasible or sports validated.',
        'differentiation': 'Second-order finite differences; endpoint and noise sensitivity remain.',
        'acceleration_residual_scope': 'Local reduced mass solve with inverse constraint forces held fixed; not dynamic tracking error.',
        'contact_caution': 'Inverse torque may rely on contact support; nonzero projected constraint forces must be inspected.',
        'fixed_chest': True, 'ball_model': False, 'forward_tracking_validated': False,
        'sports_prediction_validated': False, 'frame_count': len(qpos), 'dt_s': float(dt),
        'independent_dof_count': len(independent),
        'independent_joints': [model.joint(j).name for j in independent],
        'equality_max_error_rad': equality_error,
        'contact_frame_count': sum(r['contact_count'] > 0 for r in rows),
        'solver_failure_count': sum(not r['solver_success'] for r in rows),
        'torque_residual_rmse_nm_mean': float(np.mean([r['torque_residual_rmse_nm'] for r in rows])),
        'torque_residual_max_nm': max(r['torque_residual_max_nm'] for r in rows),
        'rows': rows, 'instantaneous_activations': activations,
    }


def main():
    from muscle_model_check import load_arm
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, default=Path('artifacts/pitch-fit/trajectory.npz'))
    parser.add_argument('--vendor', type=Path, default=Path('artifacts/vendor'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/pitch-fit/muscle-capacity.json'))
    args = parser.parse_args()
    model, warnings = load_arm(args.vendor)
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        if 'model_joint_ranges' in trajectory and not np.allclose(trajectory['model_joint_ranges'],model.jnt_range):
            raise ValueError('Trajectory was fitted with different joint limits; refusing mismatched model audit')
        time = trajectory['time']
        steps = np.diff(time)
        if len(steps) < 2 or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-9):
            raise ValueError('Regular time samples required')
        report = capacity_for_trajectory(model, trajectory['qpos'], float(steps[0]))
    report['compile_warnings'] = warnings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'instantaneous_activations')}, indent=2))


if __name__ == '__main__':
    main()
