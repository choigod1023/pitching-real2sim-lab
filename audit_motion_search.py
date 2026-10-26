"""Independent saved-state replay and actual mechanical constraint audit."""
import json
import hashlib
from pathlib import Path
import mujoco
import numpy as np
from scipy.stats import qmc
from audit_multi_joint_timing import independent_budget, replay, rotations
from coupled_payload import model_for


def geometry(state):
    model = model_for(state['lengths'], float(state['dt']))
    data = mujoco.MjData(model)
    points = []
    for q, v in zip(state['qpos'], state['qvel']):
        data.qpos[:] = q; data.qvel[:] = v
        mujoco.mj_forward(model, data)
        points.append(data.site_xpos[model.site('ball_center').id].copy())
    jac = np.zeros((3, model.nv)); jr = np.zeros_like(jac)
    mujoco.mj_jacSite(model, data, jac, jr, model.site('ball_center').id)
    return np.asarray(points), jac @ data.qvel


def run(folder='artifacts/constrained-motion-search'):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    baseline = report['baseline_fine']
    assert hashlib.sha256(Path(report['source']).read_bytes()).hexdigest() == report['source_sha256']
    assert hashlib.sha256(Path('MOTION_SEARCH_PROTOCOL.md').read_bytes()).hexdigest() == report['protocol_sha256']
    rows = report['records']
    assert len(rows) <= report['evaluation_budget']
    assert len({tuple(r['parameters']) for r in rows}) == len(rows)
    expected_sobol = 2*qmc.Sobol(6,scramble=True,seed=report['seed']).random_base2(6)-1
    np.testing.assert_array_equal(expected_sobol, np.array([r['parameters'] for r in rows[1:65]]))
    coarse_base = rows[0]
    for row in rows:
        assert np.max(np.abs(row['parameters'])) <= 1
        if 'error' in row:
            assert not row['all_pass']
            continue
        assert row['initial_position_equal'] and row['initial_velocity_equal']
        assert independent_budget(coarse_base, row) == row['budget_gate']['strict_model_budget_pass']
        assert row['all_pass'] == (row['pose_pass'] and independent_budget(coarse_base, row))
    base = np.load(folder/'state-baseline-sub2.npz')
    base_points, base_v = geometry(base)
    checks = []
    for name in ('baseline', 'selected', 'diagnostic'):
        row = report[f'{name}_fine']
        state = np.load(folder/f'state-{name}-sub2.npz')
        assert np.array_equal(base['qpos'][0], state['qpos'][0])
        assert np.array_equal(base['qvel'][0], state['qvel'][0])
        assert np.array_equal(base['lengths'], state['lengths'])
        pos, vel, positive, negative, speed = replay(state['candidate_q'], state['controls'], state['lengths'], float(state['dt']))
        errors = dict(qpos=float(np.max(np.abs(pos-state['qpos']))),
                      qvel=float(np.max(np.abs(vel-state['qvel']))),
                      positive_work_j=float(np.max(np.abs(positive-row['positive_work_per_dof_j']))),
                      negative_work_j=float(np.max(np.abs(negative-row['negative_work_magnitude_per_dof_j']))),
                      speed_kmh=abs(speed-row['speed_kmh']))
        assert max(errors.values()) < 1e-8, errors
        points, final_v = geometry(state)
        angular = rotations(base['qpos'], pos)
        displacement = float(np.linalg.norm(points-base_points,axis=1).max()*1000)
        direction = float(np.rad2deg(np.arccos(np.clip(np.dot(final_v,base_v)/(np.linalg.norm(final_v)*np.linalg.norm(base_v)),-1,1))))
        budget = independent_budget(baseline,row)
        pose = max(angular.values()) <= 3 and displacement <= 20 and direction <= 1
        assert budget == row['budget_gate']['strict_model_budget_pass']
        assert pose == row['pose_pass']
        tracking = rotations(state['candidate_q'], pos)
        checks.append(dict(case=name,replay_errors=errors,budget_pass=budget,pose_pass=pose,
            actual_rotation_difference_deg=angular,actual_ball_center_max_difference_mm=displacement,
            release_direction_difference_deg=direction,reference_tracking_rotation_deg=tracking,
            combined_pass=budget and pose,model_speed_kmh=speed,
            speed_gain_kmh=speed-baseline['speed_kmh']))
    assert checks[1]['combined_pass'], 'Selected result must be feasible, including baseline fallback'
    result=dict(selected_replays=checks,initial_states_and_geometry_identical=True,
                audited_coarse_evaluations=len(rows),coarse_budget_flags_verified=True,
                recorded_speed_kmh=79.8*1.609344,
                baseline_recorded_minus_model_kmh=79.8*1.609344-baseline['speed_kmh'],
                scope='Mechanical screen only, not physiological capacity, comfort or real intervention validation.')
    (folder/'independent-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    run()
