"""Dual-resolution feasibility restoration; all original strict caps remain."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from coupled_payload import model_for
from joint_intervention import inverse_controls
from coach_hypotheses.gain_feasibility.shield import shield_rollout
from .search import (POLICIES,temporal_basis,applied_controls,group_work,evaluate,
                     normalized_margins,select_resolved,cost_vector)


def choose_start(records):
    """Fixed engineering recovery target, not a declaration of feasibility."""
    eligible=[r for r in records if r['paired_gain_kmh']>=.5 and r['pose_screen_pass']]
    if not eligible:
        raise ValueError('No prespecified restoration start')
    def excess(row):
        return max([v['excess'] for v in row['budget_gate']['violations'] if 'work' in v['metric']] or [0.])
    return min(eligible,key=excess)


def restoration_margins(baselines,pair):
    """Both resolutions and the fixed gain floor must hold simultaneously."""
    margins=[]
    for sub,key in ((1,'coarse'),(2,'fine')):
        base=baselines[sub][0]
        raw=normalized_margins(base,pair[key],POLICIES[0])
        cap=cost_vector(base,POLICIES[0])
        raw[:len(cap)]=(cap-cost_vector(pair[key],POLICIES[0]))/np.maximum(np.abs(cap),1.)
        # Input generation enforces these peak limits analytically. Keep all
        # original peak checks in final gate, without redundant flat inequalities.
        margins.append(np.r_[raw[:22],raw[37:]])
    return np.r_[margins[0],margins[1],
                 pair['coarse']['paired_gain_kmh']-.5,pair['fine']['paired_gain_kmh']-.5]


def run_restore(source,prior_report,out):
    """Restore a recorded near-feasible point without increasing budget bounds."""
    source,prior_report,out=Path(source),Path(prior_report),Path(out)
    prior=json.loads(prior_report.read_text(encoding='utf-8'))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=prior['source_sha256']:
        raise ValueError('Reference hash mismatch')
    original=next(p for p in prior['policies'] if p['policy']==POLICIES[0])
    start_row=choose_start(original['records'])
    if start_row['evaluation_id']!=296:
        raise ValueError('Recorded prespecified start changed')
    start=np.array(start_row['parameters'])
    state=np.load(source);q,dt,lengths=state['qpos'],float(state['dt']),state['lengths']
    model=model_for(lengths,dt);controls=inverse_controls(model,q,dt);basis=temporal_basis(len(q),dt)
    baselines={};baseline_rows={};out.mkdir(parents=True,exist_ok=True)
    for sub in (1,2):
        b,bp,bv,trace=shield_rollout(lengths,q,controls,dt,sub)
        b.update(group_work(trace['applied_loads'],bv,dt/sub));baselines[sub]=b,bp,bv
        row,actual=evaluate(lengths,q,controls,dt,sub,baselines[sub],POLICIES[0],np.zeros(33))
        baseline_rows[sub]=row
        np.savez_compressed(out/f'baseline-sub{sub}.npz',**actual,source_controls=controls)
    fallback={'parameters':[0.]*33,'coarse':baseline_rows[1],'fine':baseline_rows[2]}
    records=[];cache={}
    class Limit(Exception):
        pass
    def ev(x):
        key=tuple(np.asarray(x,float))
        if key in cache:return cache[key]
        if len(records)>=400:raise Limit()
        applied=applied_controls(controls,basis,np.asarray(x))
        coarse,_=evaluate(lengths,q,applied,dt,1,baselines[1],POLICIES[0],x)
        fine,_=evaluate(lengths,q,applied,dt,2,baselines[2],POLICIES[0],x)
        pair={'parameters':list(key),'coarse':coarse,'fine':fine,
              'distance_squared_from_start':float(np.sum((np.asarray(x)-start)**2)),
              'evaluation_id':len(records)}
        records.append(pair);cache[key]=pair
        if len(records)%25==0:
            print('restore paired evaluations',len(records),'smallest margin',float(restoration_margins(baselines,pair).min()),flush=True)
            (out/'partial.json').write_text(json.dumps(records,allow_nan=False),encoding='utf-8')
        return pair
    ev(start)
    try:
        opt=minimize(lambda x:.5*np.sum((x-start)**2),start,jac=lambda x:x-start,method='SLSQP',
                     bounds=[(-1.,1.)]*33,constraints=[{'type':'ineq','fun':lambda x:restoration_margins(baselines,ev(x))}],
                     options={'maxiter':25,'ftol':1e-12,'eps':1e-5})
        ev(opt.x)
        message={'success':bool(opt.success),'message':str(opt.message),'nit':int(opt.nit)}
    except Limit:
        message={'success':False,'message':'400 paired-evaluation limit reached'}
    eligible=[]
    for pair in records:
        selected=select_resolved([fallback,pair],POLICIES[0])
        if selected is pair and pair['coarse']['paired_gain_kmh']>=.5 and pair['fine']['paired_gain_kmh']>=.5:
            eligible.append(pair)
    selected=min(eligible,key=lambda p:p['distance_squared_from_start']) if eligible else fallback
    applied=applied_controls(controls,basis,np.array(selected['parameters']))
    for sub in (1,2):
        result,trace=evaluate(lengths,q,applied,dt,sub,baselines[sub],POLICIES[0],selected['parameters'])
        np.savez_compressed(out/f'selected-sub{sub}.npz',**trace,source_controls=applied)
    report={'source':str(source.resolve()),'source_sha256':digest,
            'prior_report':str(prior_report.resolve()),'prior_report_sha256':hashlib.sha256(prior_report.read_bytes()).hexdigest(),
            'protocol_sha256':hashlib.sha256(Path(__file__).with_name('RECOVERY_PROTOCOL.md').read_bytes()).hexdigest(),
            'numerical_target_note_sha256':hashlib.sha256(Path(__file__).with_name('NUMERICAL_TARGET_NOTE.md').read_bytes()).hexdigest(),
            'optimization_target_uses_acceptance_epsilon':False,
            'redundant_peak_note_sha256':hashlib.sha256(Path(__file__).with_name('REDUNDANT_PEAK_NOTE.md').read_bytes()).hexdigest(),
            'optimizer_peak_caps_enforced_by_input_generation':True,
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'start_evaluation_id':start_row['evaluation_id'],'start_gain_kmh':start_row['paired_gain_kmh'],
            'optimizer':message,'evaluations':len(records),'records':records,'eligible_count':len(eligible),
            'selected':selected,'baseline_fallback':not bool(eligible),
            'real_effect_validated':False,'physiological_capacity_validated':False,'empirical_effect_95ci':None}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('restored selected',selected['fine']['paired_gain_kmh'],'fallback',not bool(eligible),message,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--prior-report',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run_restore(a.source,a.prior_report,a.out)
