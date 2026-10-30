"""Paired endpoint engineering sensitivity on saved actual rollout states.

Grip changes here move a diagnostic site only. They do not rerun rigid-payload
dynamics or establish a release-time error distribution or anatomical axes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def interpolate_state(model, qpos, qvel, dt, time_s):
    """Interpolate within the closed saved interval, without frame rounding.

    Position uses MuJoCo's quaternion-aware tangent difference/integration;
    generalized velocity is linear. No extrapolation is permitted.
    """
    qpos, qvel = np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float)
    if (qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 2
            or qvel.shape != (len(qpos), model.nv)
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
            or not np.isfinite([dt, time_s]).all() or dt <= 0):
        raise ValueError('Finite matching saved states and a positive dt are required')
    for joint_type, address in zip(model.jnt_type, model.jnt_qposadr):
        if joint_type in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            start = int(address) + (3 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 0)
            norms = np.linalg.norm(qpos[:, start:start + 4], axis=1)
            if not np.allclose(norms, 1., rtol=0., atol=1e-6):
                raise ValueError('Saved quaternion states must have unit norm')
    times = np.arange(len(qpos), dtype=float) * dt
    if time_s < times[0] or time_s > times[-1]:
        raise ValueError('Requested time lies outside the closed saved interval')
    right = int(np.searchsorted(times, time_s, side='left'))
    if times[right] == time_s:
        return qpos[right].copy(), qvel[right].copy()
    left = right - 1
    alpha = (time_s - times[left]) / (times[right] - times[left])
    tangent = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, tangent, 1., qpos[left], qpos[right])
    position = qpos[left].copy()
    mujoco.mj_integratePos(model, position, tangent, float(alpha))
    return position, (1. - alpha) * qvel[left] + alpha * qvel[right]


def site_velocity(model, qpos, qvel, grip_m):
    """Return ball-center velocity for a hand-frame x offset (m), using mj_jacSite.

    A private model copy relocates the diagnostic site. Its rigid payload inertia
    is intentionally unchanged: this is kinematics of saved states, not dynamics.
    The site must lie on the hand or a fixed descendant of the hand.
    """
    qpos, qvel = np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float)
    if (qpos.shape != (model.nq,) or qvel.shape != (model.nv,)
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
            or not np.isfinite(grip_m) or grip_m <= 0):
        raise ValueError('Finite model-sized state and a positive grip are required')
    hand = model.body('hand').id
    site = model.site('ball_center').id
    site_body = int(model.site_bodyid[site])
    ancestor = site_body
    while ancestor != hand:
        if ancestor == 0 or model.body_jntnum[ancestor] != 0:
            raise ValueError('ball_center must be fixed to the hand')
        ancestor = int(model.body_parentid[ancestor])
    local = copy.copy(model)
    data = mujoco.MjData(local)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(local, data)
    point = data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ np.array([grip_m, 0., 0.])
    local.site_pos[site] = data.xmat[site_body].reshape(3, 3).T @ (point - data.xpos[site_body])
    mujoco.mj_forward(local, data)
    linear = np.zeros((3, local.nv))
    angular = np.zeros((3, local.nv))
    mujoco.mj_jacSite(local, data, linear, angular, site)
    return linear @ qvel


def paired_summary(baseline_kmh, candidate_kmh, zero_tolerance_kmh=1e-9):
    """Summarize matched deterministic scenarios; variances are not uncertainty.

    Equal scenario weights are descriptive only. The covariance term documents
    that variation of paired differences is not the sum of independent variances.
    """
    baseline, candidate = np.asarray(baseline_kmh, float), np.asarray(candidate_kmh, float)
    if (baseline.ndim != 1 or baseline.size == 0 or candidate.shape != baseline.shape
            or not np.isfinite(baseline).all() or not np.isfinite(candidate).all()
            or not np.isfinite(zero_tolerance_kmh) or zero_tolerance_kmh < 0):
        raise ValueError('Nonempty finite matched scenarios and nonnegative tolerance required')
    delta = candidate - baseline
    positive, negative = delta > zero_tolerance_kmh, delta < -zero_tolerance_kmh
    sign = ('positive' if positive.all() else 'negative' if negative.all()
            else 'zero' if (np.abs(delta) <= zero_tolerance_kmh).all() else 'mixed_or_zero')
    vb, vc = float(np.var(baseline)), float(np.var(candidate))
    covariance = float(np.mean((baseline - baseline.mean()) * (candidate - candidate.mean())))
    return {
        'scenario_count': int(len(delta)),
        'paired_delta_range_kmh': [float(delta.min()), float(delta.max())],
        'sign_stability': sign,
        'all_positive': bool(positive.all()),
        'positive_count': int(positive.sum()),
        'negative_count': int(negative.sum()),
        'zero_count': int((np.abs(delta) <= zero_tolerance_kmh).sum()),
        'zero_tolerance_kmh': zero_tolerance_kmh,
        'descriptive_paired_delta_variance_kmh2': float(np.var(delta)),
        'descriptive_baseline_variance_kmh2': vb,
        'descriptive_candidate_variance_kmh2': vc,
        'descriptive_paired_covariance_kmh2': covariance,
        'variance_identity_rhs_kmh2': vb + vc - 2. * covariance,
        'variance_scope': 'Equal-weight finite engineering scenarios only; no probability distribution, standard error, confidence interval, or independent-error assumption.',
    }


def paired_sensitivity(model, baseline_qpos, baseline_qvel, candidate_qpos, candidate_qvel,
                       dt, *, grip_lengths=(.14, .15, .16), frame_offsets=(-2, -1, 0, 1, 2),
                       source_frame_dt=1 / 360, endpoint_time=None):
    """Compare baseline/candidate at identical grip and exact requested times.

    Default endpoint is the last baseline state. Positive offsets are explicitly
    excluded when the saved trajectories do not contain post-endpoint states.
    This partial engineering range does not model actual ball-release error.
    """
    baseline_qpos, candidate_qpos = np.asarray(baseline_qpos), np.asarray(candidate_qpos)
    if (baseline_qpos.ndim != 2 or candidate_qpos.ndim != 2
            or not np.isfinite([dt, source_frame_dt]).all() or dt <= 0 or source_frame_dt <= 0):
        raise ValueError('Trajectory arrays and positive finite time steps required')
    end = float((len(baseline_qpos) - 1) * dt if endpoint_time is None else endpoint_time)
    grips, offsets = tuple(grip_lengths), tuple(frame_offsets)
    if (not grips or not offsets or not np.isfinite([end, *grips, *offsets]).all()
            or min(grips) <= 0 or len(set(grips)) != len(grips)
            or len(set(offsets)) != len(offsets) or any(int(x) != x for x in offsets)):
        raise ValueError('Unique positive grips, integer offsets, and finite endpoint required')
    # Validate both complete saved trajectories even if a requested time is excluded.
    interpolate_state(model, baseline_qpos, baseline_qvel, dt, 0.)
    interpolate_state(model, candidate_qpos, candidate_qvel, dt, 0.)
    baseline_end = float((len(baseline_qpos) - 1) * dt)
    candidate_end = float((len(candidate_qpos) - 1) * dt)
    rows, excluded = [], []
    for offset in offsets:
        time = float(end + offset * source_frame_dt)
        reason = []
        if time < 0 or time > baseline_end:
            reason.append('outside_baseline_saved_interval')
        if time < 0 or time > candidate_end:
            reason.append('outside_candidate_saved_interval')
        if reason:
            excluded.append({'frame_offset': int(offset), 'time_s': time, 'reasons': reason,
                             'excluded_grip_lengths_m': list(grips)})
            continue
        bq, bv = interpolate_state(model, baseline_qpos, baseline_qvel, dt, time)
        cq, cv = interpolate_state(model, candidate_qpos, candidate_qvel, dt, time)
        for grip in grips:
            bvelocity = site_velocity(model, bq, bv, grip)
            cvelocity = site_velocity(model, cq, cv, grip)
            bspeed, cspeed = float(np.linalg.norm(bvelocity) * 3.6), float(np.linalg.norm(cvelocity) * 3.6)
            rows.append({'grip_length_m': float(grip), 'frame_offset': int(offset), 'time_s': time,
                         'baseline_velocity_mps': bvelocity.tolist(), 'candidate_velocity_mps': cvelocity.tolist(),
                         'baseline_speed_kmh': bspeed, 'candidate_speed_kmh': cspeed,
                         'paired_delta_kmh': cspeed - bspeed})
    summary = (paired_summary([r['baseline_speed_kmh'] for r in rows], [r['candidate_speed_kmh'] for r in rows])
               if rows else None)
    return {
        'scope': 'Paired endpoint kinematics of saved actual rollout qpos/qvel. Grip and time are shared by baseline and candidate. No rigid-payload mass/inertia dynamics rerun.',
        'time_scope': 'Finite partial engineering offsets; not a measured ball-release timing error, probability distribution, or confidence interval. Missing post-endpoint states are excluded without extrapolation.',
        'axis_scope': 'Grip lies along the hand marker-frame x axis; hand-frame rotation axes are not anatomically calibrated.',
        'interpolation': 'No frame rounding. MuJoCo differentiatePos/integratePos quaternion interpolation and linear generalized velocity.',
        'state_dt_s': float(dt), 'source_frame_dt_s': float(source_frame_dt), 'endpoint_time_s': end,
        'baseline_saved_interval_s': [0., baseline_end], 'candidate_saved_interval_s': [0., candidate_end],
        'requested_grip_lengths_m': list(grips), 'requested_frame_offsets': list(offsets),
        'scenarios': rows, 'excluded_times': excluded, 'summary': summary,
        'coaching_ready': False,
    }


def run(baseline_path, candidate_path, out):
    """Read actual-state NPZs and write a separate, provenance-tagged JSON report."""
    from coupled_payload import model_for

    baseline_path, candidate_path, out = Path(baseline_path), Path(candidate_path), Path(out)
    with np.load(baseline_path, allow_pickle=False) as b, np.load(candidate_path, allow_pickle=False) as c:
        if float(b['dt']) != float(c['dt']) or not np.array_equal(b['lengths'], c['lengths']):
            raise ValueError('Paired NPZs must have identical dt and body lengths')
        model = model_for(b['lengths'], float(b['dt']))
        report = paired_sensitivity(model, b['qpos'], b['qvel'], c['qpos'], c['qvel'], float(b['dt']))
    report['sources'] = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in [('baseline', baseline_path), ('candidate', candidate_path)]}
    if out.resolve() in {baseline_path.resolve(), candidate_path.resolve()}:
        raise ValueError('Report path cannot overwrite either source')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.baseline, args.candidate, args.out)
