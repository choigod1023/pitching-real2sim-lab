"""Target-free paired joint-to-ball kinematic mapping, not capacity validation."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from coupled_payload import model_for
from grip_identifiability import velocity_map
from torque_arm import derivatives


def endpoint_speed(lengths, dt, position, velocity, grip):
    model = model_for(lengths, dt, grip=(grip, 0., 0.))
    data = mujoco.MjData(model)
    data.qpos[:] = position
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    jac = np.zeros((3, model.nv))
    rotjac = np.zeros_like(jac)
    site = model.site('ball_center').id
    mujoco.mj_jacSite(model, data, jac, rotjac, site)
    v = jac @ velocity
    # Independent rigid-offset identity and position finite-difference checks.
    wrist, angular_map, _, _ = velocity_map(lengths, dt, position, velocity)
    identity_error = np.linalg.norm(v - wrist - angular_map @ [grip, 0., 0.])
    positions = []
    epsilon = 1e-7
    for sign in (-1, 1):
        data.qpos[:] = position
        mujoco.mj_integratePos(model, data.qpos, velocity, sign * epsilon)
        mujoco.mj_forward(model, data)
        positions.append(data.site_xpos[site].copy())
    finite_difference_error = np.linalg.norm(v - (positions[1] - positions[0]) / (2 * epsilon))
    if identity_error > 1e-9 or finite_difference_error > 1e-6:
        raise AssertionError('Independent kinematic mapping check failed')
    return dict(speed_kmh=float(np.linalg.norm(v) * 3.6),
                velocity_mps=v.tolist(),
                rigid_identity_error_mps=float(identity_error),
                position_finite_difference_error_mps=float(finite_difference_error))


def run(source, out):
    source, out = Path(source), Path(out)
    files = sorted(source.glob('state-*-sub1.npz'))
    if len(files) != 17:
        raise ValueError(f'Require all 17 preregistered cases, found {len(files)}')
    rows, provenance = [], []
    for file in files:
        state = np.load(file)
        case = file.name.removeprefix('state-').removesuffix('-sub1.npz')
        q = state['candidate_q']
        lengths, dt = state['lengths'], float(state['dt'])
        model = model_for(lengths, dt)
        velocity, _ = derivatives(model, q, dt)
        provenance.append(dict(case=case, source=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        for grip in (.14, .15, .16):
            row = endpoint_speed(lengths, dt, q[-1], velocity[-1], grip)
            rows.append(dict(case=case, grip_m=grip, **row))
    baseline = {row['grip_m']: row['speed_kmh'] for row in rows if row['case'] == 'baseline'}
    if len(baseline) != 3:
        raise ValueError('Missing baseline')
    for row in rows:
        row['paired_gain_kmh'] = row['speed_kmh'] - baseline[row['grip_m']]
    summaries = []
    for case in sorted({row['case'] for row in rows} - {'baseline'}):
        gain = [row['paired_gain_kmh'] for row in rows if row['case'] == case]
        signs = np.sign(gain).astype(int)
        summaries.append(dict(case=case, min_gain_kmh=min(gain), max_gain_kmh=max(gain),
                              gain_span_kmh=max(gain)-min(gain), sign_stable=bool(len(set(signs)) == 1)))
    report = dict(rows=rows, cases=summaries, sources=provenance,
                  scope='Prescribed trajectory endpoint Jacobian speeds at unchanged q/qvel for three assumed grips. No force recalculation or forward rollout for alternative grips. Not fixed capacity, measured improvement, or identified true grip.',
                  derivative_convention='torque_arm.derivatives: centered quaternion differences internally, one-sided at endpoints; identical for all grips.',
                  ball_only='A predictor F(ball_history) with fixed history and no joint input returns the same forecast for every joint intervention. This structural invariance neither falsifies ball forecasting nor validates this hand mapping.',
                  all_signs_stable=all(x['sign_stable'] for x in summaries),
                  maximum_finite_difference_error_mps=max(row['position_finite_difference_error_mps'] for row in rows))
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(baselines=baseline, cases=summaries,
                         maximum_finite_difference_error_mps=report['maximum_finite_difference_error_mps']), indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='artifacts/joint-intervention')
    parser.add_argument('--out', default='artifacts/joint-mapping')
    args = parser.parse_args()
    run(args.source, args.out)
