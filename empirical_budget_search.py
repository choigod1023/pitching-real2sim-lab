import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from coupled_payload import run as prepare, model_for
from joint_intervention import inverse_controls
from capacity_budget_gate import rollout, split_linear_power
from constrained_motion_search import modify, annotate, budget_vector

OUT=Path('artifacts/empirical-budget'); OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('../openbiomechanics/baseball_pitching/data')
SOURCE=Path('../within-athlete-candidates/artifacts/observed-candidate/reference-3034_4/reference-0.npz')

def group_costs(r, controls, velocities, dt):
    groups=[slice(0,3),slice(3,6),slice(6,7),slice(7,8),slice(8,11)]
    pos=np.zeros(5);neg=np.zeros(5)
    for force,v0,v1 in zip(controls[:-1],velocities[:-1],velocities[1:]):
        p0=np.array([force[g]@v0[g] for g in groups]);p1=np.array([force[g]@v1[g] for g in groups])
        a,b,_=split_linear_power(p0,p1,dt);pos+=a;neg+=b
    peaks=np.array([np.linalg.norm(controls[:-1,g],axis=1).max() for g in groups])
    r['group_positive_work_j']=pos.tolist();r['group_negative_work_j']=neg.tolist();r['group_peak_loads']=peaks.tolist()
    r['group_total_positive_work_j']=float(pos.sum());r['group_total_negative_work_j']=float(neg.sum())
    return r

def metrics(r):
    return np.r_[r['group_positive_work_j'],r['group_negative_work_j'],r['group_peak_loads'],r['shoulder_translation_net_work_j'],r['shoulder_rotation_net_work_j'],r['group_total_positive_work_j'],r['group_total_negative_work_j']]

def margins(r,observed,weights,base,policy):
    costs=metrics(r); cap=np.asarray(weights)@observed
    c=(cap-costs)/np.maximum(np.abs(cap),1.)
    if policy=='conserve_external_total':
        keys=['shoulder_translation_net_work_j','shoulder_rotation_net_work_j','group_total_positive_work_j','group_total_negative_work_j']
        c=np.r_[c,[(base[k]-r[k])/max(abs(base[k]),1.) for k in keys]]
    return np.r_[c,(3-r['max_rotation_deg'])/3,(20-r['max_ball_displacement_mm'])/20,1-r['release_direction_deg']]

def run():
    s=np.load(SOURCE); q=s['qpos']; lengths=s['lengths'];dt=float(s['dt']);model=model_for(lengths,dt)
    refs={}; obs={1:[],2:[]}; source_hashes={}
    for i in range(1,6):
        pitch=f'3034_{i}'
        p=Path('../within-athlete-candidates/artifacts/observed-candidate')/f'reference-{pitch}'/'reference-0.npz'
        if not p.exists():
            p=OUT/f'reference-{pitch}'/'reference-0.npz'
            if not p.exists(): prepare(DATA,pitch,p.parent,20.,(1/11520,))
        z=np.load(p); qi=z['qpos'];ti=float(z['dt']);mi=model_for(lengths,ti);control=inverse_controls(mi,qi,ti)
        source_hashes[pitch]=hashlib.sha256(p.read_bytes()).hexdigest()
        for sub in (1,2):
            r,pos,vel=rollout(lengths,qi,control,ti,sub);group_costs(r,control,vel,ti);r['pitch']=pitch;obs[sub].append(r)
            if i==4: refs[sub]=(r.copy(),pos,vel)
        print('observed',pitch,flush=True)
    matrices={sub:np.array([metrics(r) for r in obs[sub]]) for sub in (1,2)}
    base,bp,bv=refs[1];records=[]; cache={}; results=[]
    def evaluate(x):
        key=tuple(np.asarray(x,float))
        if key in cache:return cache[key]
        changed=modify(model,q,dt,x);controls=inverse_controls(model,changed,dt)
        r,p,v=rollout(lengths,changed,controls,dt,1);group_costs(r,controls,v,dt);annotate(model,base,bp,bv,r,p,v)
        r['x']=list(key);r['index']=len(records);cache[key]=r;records.append(r)
        if len(records)%25==0:print('evaluated',len(records),flush=True)
        return r
    evaluate(np.zeros(6))
    for policy in ['conserve_external_total','observed_interpolation']:
        start=len(records); attempts=[]; initial=np.r_[np.zeros(6),[0,0,0,1,0]]
        class Limit(Exception):pass
        def ev(z):
            if len(records)-start>=160 and tuple(np.asarray(z[:6],float)) not in cache:raise Limit()
            r=evaluate(z[:6]); c=margins(r,matrices[1],z[6:],base,policy)
            attempts.append(dict(index=r['index'],weights=z[6:].tolist(),min_margin=float(c.min())))
            return r,c
        messages=[]
        for starting in [initial,np.r_[np.full(6,.25),np.full(5,.2)],np.r_[np.full(6,-.25),np.full(5,.2)]]:
            try:
                opt=minimize(lambda z:-ev(z)[0]['speed_gain_kmh'],starting,method='SLSQP',bounds=[(-1,1)]*6+[(0,1)]*5,constraints=[dict(type='eq',fun=lambda z:z[6:].sum()-1),dict(type='ineq',fun=lambda z:ev(z)[1])],options=dict(maxiter=30,eps=.002,ftol=1e-9))
                messages.append(str(opt.message)); ev(opt.x)
            except Limit:
                messages.append('160 distinct-motion budget reached');break
        message=messages
        eligible=[a for a in attempts if a['min_margin']>=-1e-7 and abs(sum(a['weights'])-1)<1e-7 and min(a['weights'])>=0]
        # Keep baseline as fallback independently of optimizer sampling.
        eligible.append(dict(index=0,weights=[0,0,0,1,0],min_margin=0))
        best=max(eligible,key=lambda a:records[a['index']]['speed_gain_kmh']); x=np.array(records[best['index']]['x'])
        changed=modify(model,q,dt,x);controls=inverse_controls(model,changed,dt)
        r,p,v=rollout(lengths,changed,controls,dt,2);group_costs(r,controls,v,dt); b,bpp,bvv=refs[2];annotate(model,b,bpp,bvv,r,p,v)
        fine_margin=margins(r,matrices[2],best['weights'],b,policy)
        resolution=abs(r['speed_kmh']-records[best['index']]['speed_kmh'])+abs(b['speed_kmh']-base['speed_kmh'])
        accepted=bool(fine_margin.min()>=-1e-7 and r['speed_gain_kmh']>resolution)
        np.savez_compressed(OUT/f'{policy}.npz',qpos=p,qvel=v,controls=controls,candidate_q=changed,qreference=q,dt=dt,lengths=lengths)
        results.append(dict(policy=policy,optimizer_message=message,attempts=attempts,best_coarse=best,fine=r,fine_min_margin=float(fine_margin.min()),gain_resolution_bound=resolution,accepted_positive_gain=accepted,accepted_gain_kmh=r['speed_gain_kmh'] if accepted else 0.,weights=best['weights']))
        print(policy,'candidate',r['speed_gain_kmh'],'margin',fine_margin.min(),'accepted',accepted,flush=True)
    meta=pd.read_csv(DATA/'poi/poi_metrics.csv');actual=meta[meta.session_pitch.str.startswith('3034_')][['session_pitch','elbow_varus_moment','shoulder_internal_rotation_moment']]
    report=dict(observed=obs,records=records,policies=results,source_hashes=source_hashes,common_lengths=lengths.tolist(),actual_poi=actual.to_dict('records'),protocol_sha256=hashlib.sha256(Path('EMPIRICAL_BUDGET_PROTOCOL.md').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Observed model-demand interpolation, not physiological capacity or safety threshold')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
if __name__=='__main__':run()
