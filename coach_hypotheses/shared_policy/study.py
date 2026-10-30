"""Joint multi-pitch constrained pilot; frozen recipient evaluation follows fit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from coach_hypotheses.pitch_transfer.study import load_frozen_candidate,pair_evaluation,independent_check
from .policy import (TRAIN,EVALUATION,state_features,fit_scaler,policy_controls,work_costs,
                     C_gate,assess,choose)


def load_context(source,pitch):
    """Verify archived receipts and reuse completed baseline/input calculations."""
    source=Path(source);root=Path(__file__).parents[1]/'pitch_transfer/results'
    summary=json.loads((root/'summary.json').read_text())
    raw=gzip.decompress((root/f'{pitch}.json.gz').read_bytes())
    expected=summary['archives'][pitch]['raw_sha256']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Archived report hash mismatch')
    report=json.loads(raw);folder=source/pitch
    if hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()!=expected:
        raise ValueError('Local report differs from archived report')
    if hashlib.sha256((folder/'input.npz').read_bytes()).hexdigest()!=report['ledger']['input_sha256']:
        raise ValueError('Input receipt mismatch')
    input_state=dict(np.load(folder/'input.npz'));baselines={};traces={}
    for sub,key in ((1,'coarse'),(2,'fine')):
        trace=dict(np.load(folder/f'baseline-sub{sub}.npz'))
        independent_check(trace,report['baseline'][key])
        np.testing.assert_array_equal(trace['lengths'],input_state['lengths'])
        np.testing.assert_allclose(float(trace['dt'])*sub,float(input_state['dt']),atol=0,rtol=0)
        baselines[sub]=report['baseline'][key],trace['qpos'],trace['qvel'];traces[sub]=trace
    features,index=state_features(traces[2])
    return {'pitch':pitch,'input':input_state,'baselines':baselines,'baseline_pair':report['baseline'],
            'baseline_traces':traces,'features':features,'feature_index':index,
            'receipt':{'prior_report_sha256':expected,'input_sha256':report['ledger']['input_sha256'],
                       'baseline_state_sha256':{str(sub):hashlib.sha256((folder/f'baseline-sub{sub}.npz').read_bytes()).hexdigest() for sub in (1,2)}}}


def evaluate(weights,contexts,scaler,direction,save=None):
    """Same weights for each pitch; each uses its own frozen baseline capacities."""
    record={'weights':np.asarray(weights).tolist(),'pitches':{}}
    for pitch,ctx in contexts.items():
        state=ctx['input'];q,dt,lengths,controls=state['qpos'],float(state['dt']),state['lengths'],state['controls']
        applied,coeff,metadata=policy_controls(controls,dt,ctx['features'],scaler,weights,direction)
        pair,traces=pair_evaluation(q,dt,lengths,applied,ctx['baselines'],coeff)
        for sub,key in ((1,'coarse'),(2,'fine')):
            root=lambda trace:float(np.linalg.norm(trace['applied_loads'][:,:3],axis=1).max())
            pair[key]['C']=C_gate(ctx['baselines'][sub][0],pair[key],root(ctx['baseline_traces'][sub]),root(traces[sub]))
        pair['assessment']=assess(pair,ctx['baseline_pair']);pair['policy']=metadata
        if save is not None:
            path=Path(save)/pitch;path.mkdir(parents=True,exist_ok=True)
            pair['verification']=independent_check(traces[2],pair['fine'])
            np.savez_compressed(path/'selected-sub2.npz',**traces[2])
        record['pitches'][pitch]=pair
    return record


def margins(contexts,record,t):
    """Dual-resolution physical work, pose and worst-pitch epigraph constraints."""
    out=[]
    for pitch,ctx in contexts.items():
        for sub,key in ((1,'coarse'),(2,'fine')):
            base=work_costs(ctx['baselines'][sub][0]);r=record['pitches'][pitch][key]
            out.extend((base-work_costs(r))/np.maximum(np.abs(base),1.))
            out.extend(((3-r['max_rotation_change_deg'])/3,(20-r['max_ball_position_change_mm'])/20,
                        1-r['release_direction_change_deg'],r['paired_gain_kmh']-t))
    return np.array(out)


def optimize(contexts,scaler,direction,out):
    """Bounded two-start maximin fit without access to evaluation contexts."""
    if set(contexts)!=set(TRAIN):raise ValueError('Only preregistered training contexts permitted')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[];messages=[]
    class Limit(Exception):pass
    starts=[np.zeros(13),np.r_[np.array([[.05,0,0,0],[-.05,0,0,0],[.05,0,0,0]]).ravel(),-1.]]
    for start_index,start in enumerate(starts):
        cache={};count=0
        def ev(w):
            nonlocal count
            key=tuple(np.asarray(w,float))
            if key in cache:return cache[key]
            if count>=80:raise Limit()
            row=evaluate(w,contexts,scaler,direction);row.update(evaluation_id=len(records),start_index=start_index)
            records.append(row);cache[key]=row;count+=1
            if count%20==0:
                print('shared policy start',start_index,'policies',count,flush=True)
                (out/'partial.json').write_text(json.dumps({'records':records,'messages':messages},allow_nan=False),encoding='utf8')
            return row
        try:
            result=minimize(lambda x:-x[-1]+1e-6*np.sum(x[:-1]**2),start,method='SLSQP',
                            bounds=[(-1.,1.)]*12+[(None,None)],
                            constraints=[{'type':'ineq','fun':lambda x:margins(contexts,ev(x[:-1]),x[-1])}],
                            options={'maxiter':12,'eps':1e-5,'ftol':1e-10})
            ev(result.x[:-1]);msg={'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit)}
        except Limit:
            msg={'success':False,'message':'80 distinct-policy evaluation limit; not an impossibility proof'}
        messages.append(dict(start_index=start_index,unique_policies=count,**msg))
    selected=choose(records)
    return selected,{'records':records,'messages':messages,'policy_evaluations':len(records),
                     'paired_pitch_evaluations':len(records)*len(TRAIN),
                     'baseline_fallback':not bool(np.any(selected['weights']))}


def run(source,out):
    """Fit once, freeze, then load and evaluate two excluded development pitches."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError('Preserve completed results')
    direction,_,source_hash=load_frozen_candidate()
    train={p:load_context(source,p) for p in TRAIN}
    scaler=fit_scaler([train[p]['features'] for p in TRAIN])
    selected,optimization=optimize(train,scaler,direction,out)
    weights=np.array(selected['weights']);fixed=evaluate(weights,train,scaler,direction,out/'states')
    # Selection/scaler are frozen on disk before any recipient contexts are loaded.
    freeze={'weights':weights.tolist(),'scaler':scaler,'training_ids':list(TRAIN),
            'selected_evaluation_id':selected['evaluation_id'],'direction_source_sha256':source_hash,
            'protocol_sha256':hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest()}
    frozen_bytes=(json.dumps(freeze,indent=2,allow_nan=False)+'\n').encode('utf8')
    (out/'frozen-policy.json').write_bytes(frozen_bytes)
    excluded={p:load_context(source,p) for p in EVALUATION}
    evaluation=evaluate(weights,excluded,scaler,direction,out/'states')
    report={'frozen_policy':freeze,'frozen_policy_sha256':hashlib.sha256(frozen_bytes).hexdigest(),
            'training':fixed,'evaluation':evaluation,'optimization':optimization,
            'inputs':{p:{'features':c['features'].tolist(),'feature_index':c['feature_index'],**c['receipt']} for p,c in (train|excluded).items()},
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('policy.py'))},
            'real_effect_verified':False,'physiological_capacity_validated':False,'effect_95ci':None,
            'independent_unseen_evaluation':False,'global_research_gate_advanced':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('chosen baseline fallback',optimization['baseline_fallback'],flush=True)
    print('evaluation',[(p,r['fine']['paired_gain_kmh'],r['assessment']['pass'],r['policy']['abstained']) for p,r in evaluation['pitches'].items()],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)
