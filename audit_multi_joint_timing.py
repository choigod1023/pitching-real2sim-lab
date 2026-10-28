"""Independent control reconstruction, mechanical accounting, and selected replay."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from coupled_payload import model_for
from torque_arm import derivatives


def reconstruct(loads, shifts):
    controls = loads.copy()
    count = len(loads)-1
    for columns, shift in zip((slice(3,6), slice(6,7), slice(8,11)), shifts):
        controls[:-1, columns] = 0
        for target in range(count):
            source = target-shift
            if 0 <= source < count:
                controls[target, columns] = loads[source, columns]
    return controls


def independent_budget(base, run):
    checks = []
    for key in ('positive_work_per_dof_j', 'negative_work_magnitude_per_dof_j', 'peak_abs_generalized_force_per_dof'):
        checks.extend((np.asarray(run[key])-base[key] <= 1e-9).tolist())
    for key in ('shoulder_translation_net_work_j', 'shoulder_rotation_net_work_j'):
        checks.append(run[key]-base[key] <= 1e-9)
    checks.extend(value-base['peak_joint_torque_norm_nm'][key] <= 1e-9
                  for key,value in run['peak_joint_torque_norm_nm'].items())
    return all(checks)


def rotations(base, q):
    def angle(a,b):
        a=a/np.linalg.norm(a,axis=1,keepdims=True)
        b=b/np.linalg.norm(b,axis=1,keepdims=True)
        return float(np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=1)),0,1))).max())
    return dict(shoulder=angle(base[:,3:7],q[:,3:7]),
                elbow=float(np.rad2deg(np.abs(q[:,7]-base[:,7])).max()),
                marker_roll=float(np.rad2deg(np.abs(q[:,8]-base[:,8])).max()),
                wrist=angle(base[:,9:13],q[:,9:13]))


def replay(q, controls, lengths, dt):
    model=model_for(lengths,dt/2)
    model.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    data=mujoco.MjData(model)
    velocity,_=derivatives(model,q,dt)
    data.qpos[:]=q[0];data.qvel[:]=velocity[0]
    mujoco.mj_forward(model,data)
    positions=[data.qpos.copy()];velocities=[data.qvel.copy()]
    positive=np.zeros(model.nv);negative=positive.copy()
    for force in controls[:-1]:
        data.qfrc_applied[:]=force
        for _ in range(2):
            before=force*data.qvel.copy()
            mujoco.mj_step(model,data)
            after=force*data.qvel
            for j,(a,b) in enumerate(zip(before,after)):
                if a*b < 0:
                    pos=dt/4*max(a,b)**2/abs(b-a)
                    neg=dt/4*min(a,b)**2/abs(b-a)
                else:
                    pos=dt/4*(max(a,0)+max(b,0))
                    neg=dt/4*(max(-a,0)+max(-b,0))
                positive[j]+=pos;negative[j]+=neg
        positions.append(data.qpos.copy());velocities.append(data.qvel.copy())
    mujoco.mj_forward(model,data)
    jac=np.zeros((3,model.nv));rotation_jac=np.zeros_like(jac)
    mujoco.mj_jacSite(model,data,jac,rotation_jac,model.site('ball_center').id)
    return np.array(positions),np.array(velocities),positive,negative,float(np.linalg.norm(jac@data.qvel)*3.6)


def run(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    source=Path(report['source']);state=np.load(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==report['source_sha256']
    fine={r['case_id']:r for r in report['runs'] if r['substeps']==2}
    coarse={r['case_id']:r for r in report['runs'] if r['substeps']==1}
    assert len(fine)==len(coarse)==125
    baseline=fine[report['baseline_case_id']]
    base_state=np.load(folder/f"state-{report['baseline_case_id']}-sub2.npz")
    source_q=state['qpos'];dt=float(state['dt']);loads=state['required_loads']
    max_rotation_error=0.;max_peak_increase=0.;passing=[]
    for name,case in fine.items():
        saved=np.load(folder/f'state-{name}-sub2.npz')
        controls=reconstruct(loads,case['shifts_samples'])
        assert np.array_equal(controls,saved['controls'])
        assert np.array_equal(saved['qpos'][0],base_state['qpos'][0])
        assert np.array_equal(saved['qvel'][0],base_state['qvel'][0])
        peak=np.max(np.abs(controls[:-1]),axis=0)
        max_peak_increase=max(max_peak_increase,float(np.max(peak-np.max(np.abs(loads[:-1]),axis=0))))
        assert np.array_equal(peak,case['peak_abs_generalized_force_per_dof'])
        observed=rotations(base_state['qpos'],saved['qpos'])
        max_rotation_error=max(max_rotation_error,max(abs(observed[k]-case['max_rotation_difference_from_baseline_deg'][k]) for k in observed))
        assert (max(observed.values())<=3.) == case['small_motion_pass']
        assert independent_budget(baseline,case)==case['budget_gate']['strict_model_budget_pass']
        assert independent_budget(coarse[report['baseline_case_id']],coarse[name])==coarse[name]['budget_gate']['strict_model_budget_pass']
        if name!=report['baseline_case_id'] and case['small_motion_pass'] and coarse[name]['small_motion_pass'] and independent_budget(baseline,case) and coarse[name]['budget_gate']['strict_model_budget_pass']:
            threshold=abs(baseline['speed_kmh']-coarse[report['baseline_case_id']]['speed_kmh'])+abs(case['speed_kmh']-coarse[name]['speed_kmh'])
            if case['speed_difference_from_baseline_kmh']>threshold:passing.append(name)
    assert max_peak_increase<=1e-9
    assert max_rotation_error<1e-4
    assert len(passing)==report['accepted_model_candidate_count']
    replay_checks=[]
    replay_names=[report['baseline_case_id'],report['selected_case_id']]
    replay_names.extend(name for name,case in fine.items()
                        if case['small_motion_pass'] and sum(v!=0 for v in case['shifts_samples'])>=2)
    for name in dict.fromkeys(replay_names):
        saved=np.load(folder/f'state-{name}-sub2.npz')
        pos,vel,positive,negative,speed=replay(source_q,reconstruct(loads,fine[name]['shifts_samples']),state['lengths'],dt)
        check=dict(case_id=name,qpos_max_abs_difference=float(np.max(np.abs(pos-saved['qpos']))),
                   qvel_max_abs_difference=float(np.max(np.abs(vel-saved['qvel']))),
                   positive_work_max_abs_difference_j=float(np.max(np.abs(positive-fine[name]['positive_work_per_dof_j']))),
                   negative_work_max_abs_difference_j=float(np.max(np.abs(negative-fine[name]['negative_work_magnitude_per_dof_j']))),
                   speed_abs_difference_kmh=abs(speed-fine[name]['speed_kmh']))
        assert max(v for k,v in check.items() if k!='case_id')<1e-9
        replay_checks.append(check)
    result=dict(condition_count=125,independent_gate_checks=250,accepted_candidates=passing,
                max_force_peak_increase=max_peak_increase,max_rotation_measurement_difference_deg=max_rotation_error,
                all_initial_states_exactly_equal=True,all_controls_exactly_reconstructed=True,
                replay_checks=replay_checks)
    (folder/'independent-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',default='artifacts/multi-joint-timing')
    run(parser.parse_args().folder)
