"""Bounded conditional motion search with strict mechanical acceptance."""
import argparse, hashlib, json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from coupled_payload import model_for
from torque_arm import derivatives
from joint_intervention import inverse_controls, rotation_differences
from observed_candidate_simulation import warp_elbow
from capacity_budget_gate import rollout, gate

PARAMETER_NAMES=['shoulder_x_deg','shoulder_y_deg','elbow_deg','wrist_marker_y_deg','wrist_marker_z_deg','elbow_delay_ms']
SCALES=np.array([1.,1.,1.,1.,1.,.7])


def modify(model,q,dt,x):
    x=np.asarray(x,float)
    if x.shape!=(6,) or not np.isfinite(x).all() or np.max(np.abs(x))>1+1e-12: raise ValueError('Six bounded parameters required')
    changed=warp_elbow(q,dt,x[5]*.0007)[0] if x[5] else q.copy()
    time=np.arange(len(q))*dt
    ramp=np.clip((time-time[-1]+.030)/.030,0,1)**3
    for row,a in zip(changed,ramp):
        tangent=np.zeros(model.nv); tangent[[3,4,6,9,10]]=np.deg2rad(x[:5])*a
        mujoco.mj_integratePos(model,row,tangent,1.)
    changed[:4]=q[:4]
    return changed


def endpoints(model,pos,vel):
    d=mujoco.MjData(model); out=[]
    for p in pos:
        d.qpos[:]=p; mujoco.mj_forward(model,d);out.append(d.site_xpos[model.site('ball_center').id].copy())
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    mujoco.mj_jacSite(model,d,jp,jr,model.site('ball_center').id)
    return np.array(out),jp@vel[-1]


def budget_vector(r):
    return np.r_[r['positive_work_per_dof_j'],r['negative_work_magnitude_per_dof_j'],r['peak_abs_generalized_force_per_dof'],r['shoulder_translation_net_work_j'],r['shoulder_rotation_net_work_j'],list(r['peak_joint_torque_norm_nm'].values())]


def annotate(model,base,bpos,bvel,result,pos,vel):
    changes=rotation_differences(model,bpos,pos)
    bpoints,bvelocity=endpoints(model,bpos,bvel);points,velocity=endpoints(model,pos,vel)
    cosine=np.dot(bvelocity,velocity)/(np.linalg.norm(bvelocity)*np.linalg.norm(velocity))
    direction=float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    displacement=float(np.linalg.norm(points-bpoints,axis=1).max()*1000)
    result.update(speed_gain_kmh=result['speed_kmh']-base['speed_kmh'],max_rotation_deg=max(changes.values()),joint_rotation_deg=changes,max_ball_displacement_mm=displacement,release_direction_deg=direction,budget_gate=gate(base,result),positive_work_change_j=result['total_positive_component_work_j']-base['total_positive_component_work_j'],shoulder_translation_work_change_j=result['shoulder_translation_net_work_j']-base['shoulder_translation_net_work_j'])
    result['pose_pass']=bool(result['max_rotation_deg']<=3 and displacement<=20 and direction<=1)
    result['all_pass']=bool(result['pose_pass'] and result['budget_gate']['strict_model_budget_pass'])
    result['normalized_constraints']=np.r_[(budget_vector(base)-budget_vector(result))/np.maximum(np.abs(budget_vector(base)),1.),(3-result['max_rotation_deg'])/3,(20-displacement)/20,1-direction].tolist()
    return result



def select_resolved_candidate(replays):
    """Accept positive gain only above candidate plus baseline step discrepancy."""
    coarse={r['evaluation_id']:r for r in replays if r['substeps']==1}
    fine={r['evaluation_id']:r for r in replays if r['substeps']==2}
    base=fine[0]
    baseline_difference=abs(base['speed_kmh']-coarse[0]['speed_kmh'])
    resolved=[base]
    for index,r in fine.items():
        threshold=baseline_difference+abs(r['speed_kmh']-coarse[index]['speed_kmh'])
        if r['all_pass'] and coarse[index]['all_pass'] and r['speed_gain_kmh']>threshold:
            resolved.append(r)
    return max(resolved,key=lambda r:r['speed_gain_kmh'])


def run(source,out,max_evaluations=500):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    z=np.load(source);q=z['qpos'];dt=float(z['dt']);lengths=z['lengths'];model=model_for(lengths,dt)
    records=[];cache={}; baseline_states={}; optimizer_runs=[]
    base_controls=inverse_controls(model,q,dt)
    for sub in (1,2):
        r,p,v=rollout(lengths,q,base_controls,dt,sub)
        annotate(model,r,p,v,r,p,v);r.update(parameters=[0.]*6,physical_parameters=dict(zip(PARAMETER_NAMES,[0.]*6)),evaluation_id=0)
        baseline_states[sub]=(r,p,v)
        np.savez_compressed(out/f'state-baseline-sub{sub}.npz',qreference=q,candidate_q=q,controls=base_controls,qpos=p,qvel=v,dt=dt,lengths=lengths,parameters=np.zeros(6))
    base,bpos,bvel=baseline_states[1]
    class BudgetReached(Exception):pass
    def evaluate(x):
        x=np.clip(np.asarray(x,float),-1,1);key=tuple(x.tolist())
        if key in cache:return cache[key]
        if len(records)>=max_evaluations:raise BudgetReached()
        index=len(records)
        try:
            changed=modify(model,q,dt,x);controls=inverse_controls(model,changed,dt)
            result,pos,vel=rollout(lengths,changed,controls,dt,1)
            annotate(model,base,bpos,bvel,result,pos,vel)
            vv,_=derivatives(model,changed,dt);v0,_=derivatives(model,q,dt)
            result.update(initial_position_equal=bool(np.array_equal(changed[0],q[0])),initial_velocity_equal=bool(np.array_equal(vv[0],v0[0])))
        except (ValueError,RuntimeError) as e:
            result=dict(error=str(e),speed_gain_kmh=-1e6,normalized_constraints=[-1e6]*42,all_pass=False,pose_pass=False)
        result.update(evaluation_id=index,parameters=x.tolist(),physical_parameters=dict(zip(PARAMETER_NAMES,(x*SCALES).tolist())))
        records.append(result);cache[key]=result
        if index%25==0: print('evaluation',index,'gain',round(result['speed_gain_kmh'],5),'pass',result['all_pass'],flush=True)
        return result
    evaluate(np.zeros(6))
    for x in 2*qmc.Sobol(6,scramble=True,seed=20261026).random_base2(6)-1:evaluate(x)
    for i in range(6):
        for sign in (-1,1):
            x=np.zeros(6);x[i]=sign;evaluate(x)
    candidates=sorted(records[1:],key=lambda r:sum(min(0,c)**2 for c in r['normalized_constraints']))
    starts=[np.zeros(6),np.array(candidates[0]['parameters']),np.array(candidates[1]['parameters'])]
    for start in starts:
        try:
            opt=minimize(lambda x:-evaluate(x)['speed_gain_kmh'],start,method='SLSQP',bounds=[(-1,1)]*6,constraints=[dict(type='ineq',fun=lambda x:np.array(evaluate(x)['normalized_constraints']))],options=dict(maxiter=35,ftol=1e-9,eps=.002))
            optimizer_runs.append(dict(start=start.tolist(),success=bool(opt.success),message=str(opt.message),x=opt.x.tolist(),iterations=int(opt.nit)))
        except BudgetReached:
            optimizer_runs.append(dict(start=start.tolist(),success=False,message='500-evaluation budget reached'));break
    coarse_feasible=sorted([r for r in records if r['all_pass']],key=lambda r:r['speed_gain_kmh'],reverse=True)
    pose_candidates=[r for r in records if r['pose_pass']]
    diagnostic=max(pose_candidates,key=lambda r:r['speed_gain_kmh'])
    replay_ids=set([0,diagnostic['evaluation_id']]+[r['evaluation_id'] for r in coarse_feasible[:4]])
    replay=[];states={}
    for index in sorted(replay_ids):
        r=records[index];x=np.array(r['parameters']);changed=modify(model,q,dt,x);controls=inverse_controls(model,changed,dt)
        for sub in (1,2):
            result,pos,vel=rollout(lengths,changed,controls,dt,sub)
            b,bp,bv=baseline_states[sub];annotate(model,b,bp,bv,result,pos,vel)
            result.update(evaluation_id=index,parameters=x.tolist(),physical_parameters=r['physical_parameters'])
            replay.append(result);states[(index,sub)]=(changed,controls,pos,vel)
    selected=select_resolved_candidate(replay)
    diagnostic_fine=next(r for r in replay if r['substeps']==2 and r['evaluation_id']==diagnostic['evaluation_id'])
    for label,index in [('selected',selected['evaluation_id']),('diagnostic',diagnostic['evaluation_id'])]:
        for sub in (1,2):
            changed,controls,pos,vel=states[(index,sub)]
            np.savez_compressed(out/f'state-{label}-sub{sub}.npz',qreference=q,candidate_q=changed,controls=controls,qpos=pos,qvel=vel,dt=dt,lengths=lengths,parameters=np.array(records[index]['parameters']))
    report=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(Path('MOTION_SEARCH_PROTOCOL.md').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=20261026,evaluation_budget=max_evaluations,parameter_names=PARAMETER_NAMES,records=records,optimizer_runs=optimizer_runs,replays=replay,baseline_fine=baseline_states[2][0],selected_fine=selected,diagnostic_fine=diagnostic_fine,coarse_feasible_count=len(coarse_feasible),coarse_positive_feasible_count=sum(r['speed_gain_kmh']>1e-6 for r in coarse_feasible),recorded_speed_kmh=79.8*1.609344,interpretation='Conditional model search; neither athlete gain nor global optimum established.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:report[k] for k in ['coarse_feasible_count','coarse_positive_feasible_count','selected_fine','diagnostic_fine']},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='../within-athlete-candidates/artifacts/observed-candidate/reference-3034_4/reference-0.npz');p.add_argument('--out',default='artifacts/constrained-motion-search');a=p.parse_args();run(a.source,a.out)
