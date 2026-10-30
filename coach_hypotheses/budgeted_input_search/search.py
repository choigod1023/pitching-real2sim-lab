"""Single-shooting input timing search with work in the optimizer constraints."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from capacity_budget_gate import gate, split_linear_power
from coupled_payload import model_for
from joint_intervention import inverse_controls
from coach_hypotheses.coordinated_tracking.controller import force_limits, GROUPS
from coach_hypotheses.gain_feasibility.profiles import movement_metrics
from coach_hypotheses.gain_feasibility.shield import shield_rollout

PARTS = (slice(0,3),slice(3,6),slice(6,7),slice(7,8),slice(8,11))
POLICIES = ('A_strict_components', 'B_internal_redistribution')


def temporal_basis(count, dt):
    """Three disjoint Hann pulses covering exactly the final sixty ms."""
    if count < 3 or not np.isfinite(dt) or dt <= 0 or (count-2)*dt <= .060:
        raise ValueError('Positive dt and enough unchanged initial support required')
    time = (np.arange(count)-(count-1))*dt
    phase = (time[:,None]-np.array([-.050,-.030,-.010]))/.010
    return np.where(np.abs(phase) < 1., .5*(1+np.cos(np.pi*phase)), 0.)


def applied_controls(baseline, basis, parameters):
    """Change the input sequence, never the pose, inertia or force envelope."""
    b,x,h = np.asarray(baseline,float),np.asarray(parameters,float),np.asarray(basis,float)
    if b.ndim != 2 or b.shape[1] != 11 or h.shape != (len(b),3) or x.shape != (33,):
        raise ValueError('N by 11 controls, N by 3 basis and 33 parameters required')
    if not all(np.isfinite(a).all() for a in (b,x,h)) or np.max(np.abs(x)) > 1+1e-12:
        raise ValueError('Finite bounded parameters required')
    if not np.any(x):
        return b.copy()
    limits = force_limits(b)
    change = h@x.reshape(11,3).T * (.02*limits['components'])
    out = np.clip(b+change,-limits['components'],limits['components'])
    for name,part in GROUPS.items():
        norm = np.linalg.norm(out[:-1,part],axis=1)
        ratio = np.divide(limits['groups'][name],norm,out=np.ones_like(norm),where=norm>0)
        out[:-1,part] *= np.minimum(1.,ratio)[:,None]
    out[~np.any(h,axis=1)] = b[~np.any(h,axis=1)]
    # The unused terminal row must not enter actuation or its peak envelope.
    out[-1] = b[-1]
    return out


def group_work(applied, velocities, timestep):
    """Dot within each group before splitting positive/negative actual power."""
    forces,v = np.asarray(applied,float),np.asarray(velocities,float)
    if forces.ndim != 2 or forces.shape[1] != 11 or v.shape != (len(forces)+1,11):
        raise ValueError('One eleven-vector per actual state interval required')
    if not np.isfinite(timestep) or timestep <= 0 or not np.isfinite(forces).all() or not np.isfinite(v).all():
        raise ValueError('Finite forces and velocities required')
    p0 = np.column_stack([np.sum(forces[:,part]*v[:-1,part],axis=1) for part in PARTS])
    p1 = np.column_stack([np.sum(forces[:,part]*v[1:,part],axis=1) for part in PARTS])
    positive,negative,net = split_linear_power(p0,p1,timestep)
    positive,negative,net = positive.sum(0),negative.sum(0),net.sum(0)
    return {'group_positive_work_j':positive.tolist(),'group_negative_work_j':negative.tolist(),
            'group_net_work_j':net.tolist(),'group_total_positive_work_j':float(positive.sum()),
            'group_total_negative_work_j':float(negative.sum())}


def cost_vector(result, policy):
    """Return declared policy costs; B never substitutes for original A."""
    peaks = np.r_[result['peak_abs_generalized_force_per_dof'],
                   [result['peak_joint_torque_norm_nm'][k] for k in GROUPS]]
    shoulder_net = [result['shoulder_translation_net_work_j'],result['shoulder_rotation_net_work_j']]
    if policy == POLICIES[0]:
        return np.r_[result['positive_work_per_dof_j'],result['negative_work_magnitude_per_dof_j'],peaks,shoulder_net]
    if policy == POLICIES[1]:
        return np.r_[peaks,result['group_positive_work_j'][:2],result['group_negative_work_j'][:2],
                     shoulder_net,result['group_total_positive_work_j'],result['group_total_negative_work_j'],
                     result['total_net_work_j']]
    raise ValueError('Unknown analysis policy')


def policy_gate(baseline,candidate,policy,epsilon=1e-9):
    """Use physical costs, never optimizer success/slack, for final judgement."""
    if policy == POLICIES[0]:
        return gate(baseline,candidate,epsilon)
    diff = cost_vector(candidate,policy)-cost_vector(baseline,policy)
    return {'policy_budget_pass':bool(np.all(diff<=epsilon)),
            'violation_indices':np.flatnonzero(diff>epsilon).tolist(),
            'excess_values':diff[diff>epsilon].tolist(),'floating_comparison_epsilon':epsilon,
            'physiological_capacity_validated':False,'original_strict_gate':gate(baseline,candidate,epsilon)}


def passes_policy(result, policy):
    """Keep legacy and exploratory acceptance explicitly separate."""
    return result['budget_gate']['strict_model_budget_pass'] if policy == POLICIES[0] else result['budget_gate']['policy_budget_pass']


def normalized_margins(baseline,candidate,policy):
    """Actual integrated work inequalities are evaluated inside the optimizer."""
    cap,cost = cost_vector(baseline,policy),cost_vector(candidate,policy)
    return np.r_[(cap+1e-9-cost)/np.maximum(np.abs(cap),1.),
                 (3-candidate['max_rotation_change_deg'])/3,
                 (20-candidate['max_ball_position_change_mm'])/20,
                 1-candidate['release_direction_change_deg']]


def evaluate(lengths, reference_q, applied, dt, substeps, baseline, policy, parameters):
    """Actual force-driven simulation, sampled-substep work and pose checks."""
    result,pos,vel,trace = shield_rollout(lengths,reference_q,applied,dt,substeps,budgets=None)
    base,bp,bv = baseline
    model = model_for(lengths,dt/substeps)
    result.update(group_work(trace['applied_loads'],vel,dt/substeps))
    result.update(movement_metrics(model,bp,bv,pos,vel))
    result.update(parameters=np.asarray(parameters).tolist(),substeps=substeps,
                  paired_gain_kmh=result['speed_kmh']-base['speed_kmh'],
                  positive_component_work_change_j=result['total_positive_component_work_j']-base['total_positive_component_work_j'],
                  group_positive_work_change_j=result['group_total_positive_work_j']-base['group_total_positive_work_j'],
                  initial_state_equal=bool(np.array_equal(pos[0],bp[0]) and np.array_equal(vel[0],bv[0])),
                  budget_gate=policy_gate(base,result,policy))
    result['policy_pose_pass'] = bool(passes_policy(result,policy) and result['pose_screen_pass'] and result['initial_state_equal'])
    return result,trace


def select_resolved(replays,policy):
    """Zero-input-change feasible fallback and fixed coarse/fine criterion."""
    baseline = next(r for r in replays if not np.any(r['parameters']))
    choices=[baseline]
    for pair in replays:
        if not np.any(pair['parameters']):
            continue
        coarse,fine=pair['coarse'],pair['fine']
        bound=abs(baseline['coarse']['speed_kmh']-baseline['fine']['speed_kmh'])+abs(coarse['speed_kmh']-fine['speed_kmh'])
        discrepancy=abs(coarse['paired_gain_kmh']-fine['paired_gain_kmh'])
        if all(r['policy_pose_pass'] for r in (coarse,fine)) and fine['paired_gain_kmh']>bound and fine['paired_gain_kmh']>0 and coarse['paired_gain_kmh']>0 and discrepancy<=.001:
            choices.append(pair)
    return max(choices,key=lambda pair:pair['fine']['paired_gain_kmh'])


def run_search(source,reference_report,out,max_evaluations=360):
    """Bounded multistart single-shooting, preserving all trials and failures."""
    source,reference_report,out=Path(source),Path(reference_report),Path(out)
    prior=json.loads(reference_report.read_text(encoding='utf-8'))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=prior['source_sha256']:
        raise ValueError('Reference input hash mismatch')
    if max_evaluations!=360:
        raise ValueError('Use the frozen evaluation budget')
    z=np.load(source);q,dt,lengths=z['qpos'],float(z['dt']),z['lengths']
    model=model_for(lengths,dt);controls=inverse_controls(model,q,dt);basis=temporal_basis(len(q),dt)
    out.mkdir(parents=True,exist_ok=True)
    baselines={}
    for sub in (1,2):
        b,bp,bv,trace=shield_rollout(lengths,q,controls,dt,sub)
        np.testing.assert_allclose(b['speed_kmh'],prior['baseline'][str(sub)]['speed_kmh'],atol=1e-10,rtol=0)
        b.update(group_work(trace['applied_loads'],bv,dt/sub));baselines[sub]=b,bp,bv
    policies=[]
    for policy in POLICIES:
        records=[];cache={};messages=[]
        class EvaluationLimit(Exception):
            pass
        def ev(x):
            x=np.asarray(x,float);key=tuple(x)
            if key in cache:return cache[key]
            if len(records)>=max_evaluations:raise EvaluationLimit()
            applied=applied_controls(controls,basis,x)
            r,_=evaluate(lengths,q,applied,dt,1,baselines[1],policy,x)
            r['evaluation_id']=len(records);records.append(r);cache[key]=r
            if len(records)%25==0:
                (out/f'{policy}-partial.json').write_text(json.dumps(records,allow_nan=False),encoding='utf-8')
                print(policy,'evaluations',len(records),'best feasible',max(v['paired_gain_kmh'] for v in records if v['policy_pose_pass']),flush=True)
            return r
        ev(np.zeros(33))
        starts=[np.zeros(33),.1*(-1.)**np.arange(33),-.1*(-1.)**np.arange(33)]
        for start in starts:
            try:
                opt=minimize(lambda x:-ev(x)['paired_gain_kmh'],start,method='SLSQP',bounds=[(-1.,1.)]*33,
                    constraints=[{'type':'ineq','fun':lambda x:normalized_margins(baselines[1][0],ev(x),policy)}],
                    options={'maxiter':25,'eps':.001,'ftol':1e-9})
                ev(opt.x)
                messages.append({'start':start.tolist(),'success':bool(opt.success),'message':str(opt.message),'nit':int(opt.nit)})
            except EvaluationLimit:
                messages.append({'start':start.tolist(),'success':False,'message':'360 distinct-evaluation budget reached'})
                break
        eligible=sorted([r for r in records if r['policy_pose_pass'] and np.any(r['parameters'])],key=lambda r:r['paired_gain_kmh'],reverse=True)[:3]
        selected_records=[records[0]]+eligible
        replays=[]
        for coarse in selected_records:
            x=np.array(coarse['parameters']);applied=applied_controls(controls,basis,x)
            fine,trace=evaluate(lengths,q,applied,dt,2,baselines[2],policy,x)
            replays.append({'parameters':x.tolist(),'coarse':coarse,'fine':fine})
            np.savez_compressed(out/f'{policy}-replay-{coarse["evaluation_id"]}.npz',**trace,source_controls=applied)
        chosen=select_resolved(replays,policy)
        selected={'policy':policy,'gain_kmh':chosen['fine']['paired_gain_kmh'],
                  'parameters':chosen['parameters'],'fine':chosen['fine'],
                  'is_baseline_fallback':not bool(np.any(chosen['parameters'])),
                  'actual_effect_verified':False,'physiological_capacity_validated':False}
        policy_report={'policy':policy,'evaluation_count':len(records),'optimizer_messages':messages,'records':records,'replays':replays,'selected':selected,
                       'coarse_policy_pose_pass_count':sum(r['policy_pose_pass'] for r in records),
                       'coarse_positive_policy_pose_pass_count':sum(r['policy_pose_pass'] and r['paired_gain_kmh']>0 for r in records)}
        (out/f'{policy}.json').write_text(json.dumps(policy_report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        policies.append(policy_report)
        print(policy,'selected',selected['gain_kmh'],'baseline',selected['is_baseline_fallback'],flush=True)
    report={'pitch':'3034_1','source':str(source.resolve()),'source_sha256':digest,
            'reference_report':str(reference_report.resolve()),'reference_report_sha256':hashlib.sha256(reference_report.read_bytes()).hexdigest(),
            'protocol_sha256':hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'mujoco_version':mujoco.__version__,
            'policies':policies,'primary_policy':POLICIES[0],'exploratory_policy':POLICIES[1],
            'real_athlete_effect_verified':False,'empirical_effect_95ci':None,'spin_or_command_effect_verified':False,
            'scope':'Applied force timing search; B is an explicit mechanical redistribution hypothesis, never legacy A or physiological validation.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--reference-report',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run_search(a.source,a.reference_report,a.out)
