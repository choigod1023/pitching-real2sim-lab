"""Train-only world-acceleration regularization of a moving hand-frame fit."""
import hashlib
import json
from pathlib import Path

import ezc3d
import numpy as np
from scipy.spatial.transform import Rotation

from hand_speed_derivative import numerical_rotations, metrics_for_state, improves
from physical_release_comparison import launch_states

LAMBDAS = (.001, .01, .1, 1., 10., 100., 1000.)
ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = ROOT/'artifacts/multiview-reset'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_covariant(times, ball, origin, rotation, gravity, derivative_step=1e-5, forced_lambda=None):
    t, b, o, r, g = map(lambda x: np.asarray(x, float), (times, ball, origin, rotation, gravity))
    n = len(t)
    if (t.shape != (n,) or n < (4 if forced_lambda is None else 3) or b.shape != (n, 3) or o.shape != b.shape
            or r.shape != (n, 3, 3) or g.shape != (3,)
            or not all(np.isfinite(a).all() for a in (t, b, o, r, g))
            or np.any(np.diff(t) <= 0) or not 0 < derivative_step < (t[-1]-t[0])/10):
        raise ValueError('Need finite, increasing training-only observations and a small derivative step')
    if not np.allclose(np.swapaxes(r, 1, 2)@r, np.eye(3), atol=1e-8) or np.any(np.linalg.det(r) <= 0):
        raise ValueError('Proper hand rotations required')
    span = t[-1]-t[0]
    u = (t-t[-1])/span
    z = np.column_stack([u, u*u])
    oc = np.linalg.lstsq(z, o-o[-1], rcond=None)[0]
    logs = Rotation.from_matrix(np.einsum('ij,njk->nik', r[-1].T, r)).as_rotvec()
    if np.linalg.norm(logs, axis=1).max() >= np.pi-.01:
        raise ValueError('Ambiguous rotation log branch')
    rc = np.linalg.lstsq(z, logs, rcond=None)[0]

    def hand_and_design(query):
        q = (np.asarray(query)-t[-1])/span
        zz = np.column_stack([q, q*q])
        oo = o[-1]+zz@oc
        rr = r[-1]@Rotation.from_rotvec(zz@rc).as_matrix()
        m = np.concatenate([rr, rr*q[:, None, None], rr*q[:, None, None]**2], axis=2)
        return oo, m

    oo, design = hand_and_design(t)
    op, mp = hand_and_design(t+derivative_step)
    om, mm = hand_and_design(t-derivative_step)
    oa = (op-2*oo+om)/derivative_step**2
    ma = (mp-2*design+mm)/derivative_step**2
    acc_design = (ma*span**2).reshape(-1, 9)/np.sqrt(n)
    acc_target = ((g-oa)*span**2).reshape(-1)/np.sqrt(n)

    def solve(mask, strength):
        count = int(mask.sum())
        a = design[mask].reshape(-1, 9)/np.sqrt(count)
        y = (b[mask]-oo[mask]).reshape(-1)/np.sqrt(count)
        a = np.vstack([a, np.sqrt(strength)*acc_design])
        y = np.r_[y, np.sqrt(strength)*acc_target]
        coeff, _, rank, _ = np.linalg.lstsq(a, y, rcond=None)
        if rank != 9:
            raise ValueError('Rank deficient transport fit')
        return coeff

    scores = []
    if forced_lambda is None:
        for strength in LAMBDAS:
            errors = []
            for leave in range(n):
                mask = np.arange(n) != leave
                coeff = solve(mask, strength)
                errors.append(float(np.sum((oo[leave]+design[leave]@coeff-b[leave])**2)))
            scores.append(float(np.mean(errors)))
        strength = LAMBDAS[int(np.argmin(scores))]
    else:
        if forced_lambda not in LAMBDAS:
            raise ValueError('Strength outside frozen grid')
        strength = forced_lambda
    coeff = solve(np.ones(n, dtype=bool), strength)
    offset = coeff[:3]
    offset_velocity = coeff[3:6]/span
    origin_velocity = oc[0]/span
    omega = r[-1]@(rc[0]/span)
    rotational = np.cross(omega, r[-1]@offset)
    relative = r[-1]@offset_velocity
    position = o[-1]+r[-1]@offset
    velocity = origin_velocity+rotational+relative
    detail = {
        'lambda': strength, 'lambda_grid': list(LAMBDAS),
        'training_loo_position_mse_m2': scores,
        'relative_coefficients': coeff.reshape(3, 3).tolist(),
        'origin_velocity_mps': origin_velocity.tolist(),
        'rotation_component_mps': rotational.tolist(),
        'relative_component_mps': relative.tolist(),
        'omega_world_rad_s': omega.tolist(),
        'training_frames_only': True, 'future_hand_or_ball_input': False,
        'contact_model': False, 'derivative_step_s': derivative_step}
    return position, velocity, detail


def fit_forward_regularized(times, ball, origin, rotation, gravity, derivative_step=1e-5):
    """Forecast the last *training* frame from the prefix; no future hand in CV."""
    t, b, o, r, g = map(np.asarray, (times, ball, origin, rotation, gravity))
    if len(t) != 4:
        raise ValueError('Frozen forward validation requires four training frames')
    delta = t[-1]-t[-2]
    scores = []
    for strength in LAMBDAS:
        p, v, _ = fit_covariant(t[:-1], b[:-1], o[:-1], r[:-1], g, derivative_step, strength)
        forecast = p+delta*v+.5*delta**2*g
        scores.append(float(np.sum((forecast-b[-1])**2)))
    selected = LAMBDAS[int(np.argmin(scores))]
    p, v, detail = fit_covariant(t, b, o, r, g, derivative_step, selected)
    detail['selection'] = 'Forward prefix3 to last training frame; position only'
    detail['forward_validation_position_mse_m2'] = scores
    detail['forward_validation_future_hand_used'] = False
    return p, v, detail


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    source = LAB/'hand-ball-link/artifacts/hand-ball/observation.json'
    observed = json.loads(source.read_text(encoding='utf-8'))
    for name, entry in observed['source_sha256'].items():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed original source '+name)
    alignment = json.loads(Path(observed['source_sha256']['alignment']['path']).read_text(encoding='utf-8'))
    basis = np.array(alignment['rotation'])
    c = ezc3d.c3d(observed['source_sha256']['body']['path'])
    labels = c['parameters']['ROTATION']['LABELS']['value']
    rr = c['data']['rotations'][:3, :3, labels.index('l_hand_4X4'), :]
    frames = np.arange(973, 980)
    rate = float(observed['rate_hz']); times = (frames-frames[0])/rate
    gravity = basis@np.array([0., 0., -9.81]); label = 62.7*1.609344
    rows = []
    for condition in observed['conditions']:
        ball = np.array([p['ball_rig_m'] for p in condition['positions']])
        origin = np.array([p['hand_origin_rig_m'] for p in condition['positions']])
        rotation, precision = numerical_rotations(np.einsum('ij,jkn->nik', basis, rr[:, :, frames[:4]+condition['c3d_frame_offset']]))
        states = launch_states(times[:4], ball[:4], origin[:4], rotation)
        p, v, detail = fit_covariant(times[:4], ball[:4], origin[:4], rotation, gravity)
        p2, v2, detail2 = fit_covariant(times[:4], ball[:4], origin[:4], rotation, gravity, derivative_step=5e-6)
        states['D'] = (p, v)
        pe, ve, edetail = fit_forward_regularized(times[:4], ball[:4], origin[:4], rotation, gravity)
        pe2, ve2, edetail2 = fit_forward_regularized(times[:4], ball[:4], origin[:4], rotation, gravity, 5e-6)
        states['E'] = (pe, ve)
        metrics = {name: metrics_for_state(pos, vel, gravity, ball[4:], rate, label) for name, (pos, vel) in states.items()}
        row = {'threshold': condition['threshold'], 'offset': condition['c3d_frame_offset'],
               'metrics': metrics, 'fit': detail, 'rotation_precision': precision,
               'half_derivative_step_speed_difference_kmh': float(abs(np.linalg.norm(v2)-np.linalg.norm(v))*3.6),
               'half_derivative_step_position_difference_mm': float(np.linalg.norm(p2-p)*1000),
               'half_derivative_step_same_lambda': detail['lambda'] == detail2['lambda'],
               'forward_fit': edetail,
               'E_half_derivative_step_speed_difference_kmh': float(abs(np.linalg.norm(ve2)-np.linalg.norm(ve))*3.6),
               'E_half_derivative_step_same_lambda': edetail['lambda'] == edetail2['lambda'],
               'D_beats_B_all_three': improves(metrics['D'], metrics['B']),
               'D_beats_C_all_three': improves(metrics['D'], metrics['C']),
               'E_beats_B_all_three': improves(metrics['E'], metrics['B']),
               'E_beats_C_all_three': improves(metrics['E'], metrics['C'])}
        rows.append(row)
    primary = next(r for r in rows if r['threshold'] == 80 and r['offset'] == 1)
    result = {
        'trial': observed['trial'], 'dataset': 'OBP/Prime_multiview_demo',
        'reference_changed_by_user': True, 'comparison_with3034_permitted': False,
        'training_frames': frames[:4].tolist(), 'evaluation_frames': frames[4:].tolist(),
        'rate_hz_declared': rate, 'capture_clock_independently_verified': False,
        'source_label_kmh': label, 'source_label_measurement_definition_verified': False,
        'primary_condition': {'threshold': 80, 'offset': 1}, 'primary': primary, 'conditions': rows,
        'gravity_rig_mps2': gravity.tolist(),
        'sources': {'observation': {'path': str(source.resolve()), 'sha256': sha(source)}, **observed['source_sha256']},
        'protocol_sha256': sha(ROOT/'MULTIVIEW_RESET_PROTOCOL.md'),
        'forward_protocol_sha256': sha(ROOT/'FORWARD_SELECTION_PROTOCOL.md'),
        'fit_code_sha256': sha(ROOT/'covariant_release.py'),
        'physical_body_motion_changed': False, 'posture_difference_deg': 0., 'posture_difference_mm': 0.,
        'joint_force_work_verified': False, 'force_change_n': None, 'work_change_j': None,
        'actual_release_time_verified': False, 'scientific_stage_passed': False,
        'performance_improvement_kmh': None,
        'interpretation': 'Exploratory conditional release-state reconstruction; observed training ball calibrates local state. Future hand excluded. Not a validated response to joint changes, athlete improvement, or a fix of3034.'}
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (ROOT/'multiview_reset_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'primary': {name: {k: m[k] for k in ['launch_speed_kmh', 'mean_absolute_speed_discrepancy_kmh', 'mean_velocity_vector_discrepancy_kmh', 'position_rmse_mm', 'mean_absolute_radar_scalar_discrepancy_kmh']} for name, m in primary['metrics'].items()},
        'lambda': primary['fit']['lambda'], 'D_beats_B_count': sum(r['D_beats_B_all_three'] for r in rows),
        'D_beats_C_count': sum(r['D_beats_C_all_three'] for r in rows),
        'E_beats_B_count': sum(r['E_beats_B_all_three'] for r in rows),
        'E_beats_C_count': sum(r['E_beats_C_all_three'] for r in rows)}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
