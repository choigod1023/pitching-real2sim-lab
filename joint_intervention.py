"""Preregistered joint trajectory demand experiment, not validated intervention prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from coupled_payload import model_for
from torque_arm import derivatives
from capacity_budget_gate import rollout, gate

JOINTS = {'shoulder':slice(3,6),'elbow':slice(6,7),'marker_roll':slice(7,8),'wrist_marker':slice(8,11)}


def modify_trajectory(model, q, dt, dof, angle_deg, window=.030):
    if dt <= 0 or window <= 0 or dof not in (6,8,9,10):
        raise ValueError('Positive time and declared joint required')
    time=np.arange(len(q))*dt
    phase=np.clip((time-(time[-1]-window))/window,0,1)
    angles=np.deg2rad(angle_deg)*phase**3
    changed=np.array(q,copy=True)
    for p,angle in zip(changed,angles):
        tangent=np.zeros(model.nv); tangent[dof]=angle
        mujoco.mj_integratePos(model,p,tangent,1.)
    return changed,angles


def inverse_controls(model,q,dt):
    velocity,acceleration=derivatives(model,q,dt)
    d=mujoco.MjData(model); loads=[]
    for pos,vel,acc in zip(q,velocity,acceleration):
        d.qpos[:]=pos; d.qvel[:]=vel; d.qacc[:]=acc
        mujoco.mj_inverse(model,d)
        loads.append(d.qfrc_inverse.copy())
    return np.asarray(loads)


def rotation_differences(model, a,b):
    differences=np.zeros((len(a),model.nv))
    for row,p,r in zip(differences,a,b):
        mujoco.mj_differentiatePos(model,row,1.,p,r)
    return {name:float(np.rad2deg(np.linalg.norm(differences[:,sl],axis=1)).max()) for name,sl in JOINTS.items()}


def initial_energy(model,q,v):
    d=mujoco.MjData(model); d.qpos[:]=q; d.qvel[:]=v
    mujoco.mj_forward(model,d); mujoco.mj_energyPos(model,d); mujoco.mj_energyVel(model,d)
    return float(d.energy.sum())


def run(source,out):
    source,out=Path(source),Path(out); out.mkdir(parents=True,exist_ok=True)
    state=np.load(source); q=state['qpos']; dt=float(state['dt']); lengths=state['lengths']
    model=model_for(lengths,dt); baseline_v,_=derivatives(model,q,dt)
    specs=[('baseline',6,0)]
    for name,dof in [('elbow',6),('wrist_x',8),('wrist_y',9),('wrist_z',10)]:
        specs.extend((f'{name}_{"m" if angle<0 else "p"}{abs(angle)}',dof,angle) for angle in (-2,-1,1,2))
    baselines={}; runs=[]
    for case,dof,amplitude in specs:
        changed,angles=modify_trajectory(model,q,dt,dof,amplitude)
        controls=inverse_controls(model,changed,dt)
        v,_=derivatives(model,changed,dt)
        for substeps in (1,2):
            result,pos,vel=rollout(lengths,changed,controls,dt,substeps)
            if case=='baseline': baselines[substeps]=(result,pos)
            baseline,baseline_pos=baselines[substeps]
            changes=rotation_differences(model,baseline_pos,pos)
            d=mujoco.MjData(model); d.qpos[:]=pos[-1]; mujoco.mj_forward(model,d)
            endpoint=d.site_xpos[model.site('ball_center').id].copy()
            d.qpos[:]=changed[-1]; mujoco.mj_forward(model,d)
            result.update(case=case,dof=dof,amplitude_deg=amplitude,
                speed_difference_from_baseline_kmh=result['speed_kmh']-baseline['speed_kmh'],
                budget_gate=gate(baseline,result),
                max_actual_joint_rotation_change_deg=changes,
                max_actual_rotation_change_deg=max(changes.values()),
                small_change_screen_pass=max(changes.values())<=3.,
                endpoint_ball_tracking_error_mm=float(np.linalg.norm(endpoint-d.site_xpos[model.site('ball_center').id])*1000),
                initial_qpos_exactly_equal=bool(np.array_equal(changed[0],q[0])),
                initial_qvel_exactly_equal=bool(np.array_equal(v[0],baseline_v[0])),
                initial_energy_j=initial_energy(model,changed[0],v[0]),
                positive_component_work_change_j=result['total_positive_component_work_j']-baseline['total_positive_component_work_j'],
                shoulder_translation_net_work_change_j=result['shoulder_translation_net_work_j']-baseline['shoulder_translation_net_work_j'],
                shoulder_rotation_net_work_change_j=result['shoulder_rotation_net_work_j']-baseline['shoulder_rotation_net_work_j'])
            np.savez_compressed(out/f'state-{case}-sub{substeps}.npz',qreference=q,candidate_q=changed,controls=controls,qpos=pos,qvel=vel,dt=dt,lengths=lengths,requested_angles=angles)
            runs.append(result)
        print(case,round(result['speed_difference_from_baseline_kmh'],6),result['budget_gate']['strict_model_budget_pass'],flush=True)
    resolution=[]
    for case,_,_ in specs:
        a,b=[r for r in runs if r['case']==case]
        resolution.append(dict(case=case,speed_difference_fine_minus_coarse_kmh=b['speed_kmh']-a['speed_kmh'],
            max_positive_work_component_difference_j=float(np.max(np.abs(np.asarray(a['positive_work_per_dof_j'])-b['positive_work_per_dof_j']))),
            same_budget_result=a['budget_gate']['strict_model_budget_pass']==b['budget_gate']['strict_model_budget_pass']))
    report=dict(pitch='3034_1',source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        protocol='JOINT_INTERVENTION_PROTOCOL.md',protocol_sha256=hashlib.sha256(Path('JOINT_INTERVENTION_PROTOCOL.md').read_bytes()).hexdigest(),
        scope='Prescribed joint movement, recalculated inverse forces, actual RK4 rollout. Not equal ability, not intervention truth, not prior speed-gap correction.',
        initial_energy_baseline_j=initial_energy(model,q[0],baseline_v[0]),runs=runs,resolution_comparison=resolution)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz')
    p.add_argument('--out',default='artifacts/joint-intervention')
    args=p.parse_args(); run(args.source,args.out)
