"""Preregistered cross-pitch fixed-policy transfer with optional local refits."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from coupled_payload import model_for
from joint_intervention import inverse_controls
from trace_error_audit import ObservedPitch
from coach_hypotheses.coordinated_tracking.controller import GROUPS, force_limits
from coach_hypotheses.budgeted_input_search.search import (
    POLICIES, temporal_basis, applied_controls, group_work, evaluate,
    cost_vector, normalized_margins, select_resolved, policy_gate)
from coach_hypotheses.gain_feasibility.shield import (
    shield_rollout, replay_controls, actual_accounting)

COHORT = ('3034_1','3034_2','3034_3','3034_4','3034_5',
          '2996_1','2996_2','3044_1','3044_2')
CACHED = {
    '3034_1':'grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz',
    '3034_2':'within-athlete-candidates/artifacts/observed-candidate/reference-3034_2/reference-0.npz',
    '3034_3':'empirical-budget-search/artifacts/empirical-budget/reference-3034_3/reference-0.npz',
    '3034_4':'within-athlete-candidates/artifacts/observed-candidate/reference-3034_4/reference-0.npz',
    '3034_5':'empirical-budget-search/artifacts/empirical-budget/reference-3034_5/reference-0.npz'}


def load_frozen_candidate():
    """Read the immutable archived source result, verifying its original hash."""
    folder=Path(__file__).parents[1]/'budgeted_input_search/results'
    expected=json.loads((folder/'summary.json').read_text())['archives']['clean_restore']['raw_sha256']
    raw=gzip.decompress((folder/'clean_restore.json.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Frozen candidate archive hash mismatch')
    report=json.loads(raw)
    if report['baseline_fallback']:
        raise ValueError('Frozen source is baseline fallback')
    return np.array(report['selected']['parameters']),report['selected'],expected


def absolute_transfer(source_controls,source_modified,source_dt,recipient,recipient_dt):
    """Transfer absolute force delta on BR-relative time, then recipient caps."""
    a,b,r=map(lambda v:np.asarray(v,float),(source_controls,source_modified,recipient))
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=11 or r.ndim!=2 or r.shape[1]!=11:
        raise ValueError('Matching source and eleven-component recipient required')
    if min(source_dt,recipient_dt)<=0 or not all(np.isfinite(x).all() for x in (a,b,r)):
        raise ValueError('Finite inputs and positive timesteps required')
    st=(np.arange(len(a))-len(a)+1)*source_dt
    rt=(np.arange(len(r))-len(r)+1)*recipient_dt
    delta=np.column_stack([np.interp(rt,st,b[:,i]-a[:,i],left=0,right=0) for i in range(11)])
    active=(rt>=-.060)&(rt<0)
    delta[~active]=0
    desired=r+delta
    limits=force_limits(r)
    out=np.clip(desired,-limits['components'],limits['components'])
    for name,part in GROUPS.items():
        norm=np.linalg.norm(out[:-1,part],axis=1)
        factor=np.minimum(1,np.divide(limits['groups'][name],norm,out=np.ones_like(norm),where=norm>0))
        out[:-1,part]*=factor[:,None]
    out[~active]=r[~active]
    return out,{'clipping_max_per_component':np.max(np.abs(out-desired),axis=0).tolist(),
                'changed_source_intervals':int(np.count_nonzero(np.any(out[:-1]!=r[:-1],axis=1)))}


def raw_margins(baselines,pair):
    """Raw work/pose bounds at both resolutions; peak caps remain final gates."""
    margins=[]
    for sub,key in ((1,'coarse'),(2,'fine')):
        base=baselines[sub][0];candidate=pair[key]
        raw=normalized_margins(base,candidate,POLICIES[0])
        cap=cost_vector(base,POLICIES[0])
        raw[:len(cap)]=(cap-cost_vector(candidate,POLICIES[0]))/np.maximum(np.abs(cap),1.)
        margins.extend(np.r_[raw[:22],raw[37:]])
    return np.array(margins)


def transfer_pass(pair,baseline_pair):
    """Apply full two-resolution budget, pose, gain and numerical rules."""
    c,f=pair['coarse'],pair['fine']
    bound=abs(baseline_pair['coarse']['speed_kmh']-baseline_pair['fine']['speed_kmh'])+abs(c['speed_kmh']-f['speed_kmh'])
    discrepancy=abs(c['paired_gain_kmh']-f['paired_gain_kmh'])
    return {'pass':bool(c['policy_pose_pass'] and f['policy_pose_pass'] and
                        c['paired_gain_kmh']>0 and f['paired_gain_kmh']>bound and discrepancy<=.001),
            'numerical_speed_bound_kmh':bound,'gain_resolution_difference_kmh':discrepancy}


def extra_diagnostics(base,trace,candidate):
    """Expose work and root-vector metrics absent from legacy A."""
    return {'B_gate':policy_gate(base['summary'],candidate,POLICIES[1]),
            'net_work_change_j':candidate['total_net_work_j']-base['summary']['total_net_work_j'],
            'root_force_norm_peak_change_n':float(np.linalg.norm(trace['applied_loads'][:,:3],axis=1).max()-base['root_peak']),
            'positive_component_work_change_j':candidate['total_positive_component_work_j']-base['summary']['total_positive_component_work_j'],
            'negative_component_work_change_j':candidate['total_negative_component_work_magnitude_j']-base['summary']['total_negative_component_work_magnitude_j']}


def prepare(pitch,lab,out,subject_lengths=None):
    """Reuse observed references; freeze subject geometry before inverse inputs."""
    lab,out=Path(lab),Path(out);out.mkdir(parents=True,exist_ok=True)
    if pitch in CACHED:
        path=lab/CACHED[pitch];z=np.load(path)
        q,dt,native=z['qpos'],float(z['dt']),z['lengths']
        ledger={'reference_path':str(path),'reference_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'provenance':'Existing 20Hz reference; exact cached poses preserved'}
    else:
        obs=ObservedPitch(pitch,lab/'openbiomechanics/baseball_pitching/data')
        t=np.linspace(obs.a,obs.b,int(np.ceil((obs.b-obs.a)*11520))+1)
        q,dt,native=obs.sample(t),float(t[1]-t[0]),obs.lengths
        ledger=obs.ledger
    lengths=np.array(native if subject_lengths is None else subject_lengths)
    controls=inverse_controls(model_for(lengths,dt),q,dt)
    np.savez_compressed(out/'input.npz',qpos=q,dt=dt,lengths=lengths,native_lengths=native,controls=controls)
    ledger.update(pitch=pitch,native_lengths_m=native.tolist(),fixed_lengths_m=lengths.tolist(),
                  fixed_minus_native_mm=((lengths-native)*1000).tolist(),source_dt_s=dt,
                  input_sha256=hashlib.sha256((out/'input.npz').read_bytes()).hexdigest())
    return q,dt,lengths,controls,ledger


def pair_evaluation(q,dt,lengths,controls,baselines,parameters):
    """Evaluate actual applied inputs at both integration resolutions."""
    pair={'parameters':np.asarray(parameters).tolist()};traces={}
    for sub,key in ((1,'coarse'),(2,'fine')):
        row,trace=evaluate(lengths,q,controls,dt,sub,baselines[sub],POLICIES[0],parameters)
        pair[key]=row;traces[sub]=trace
    return pair,traces


def independent_check(trace,expected):
    """Reintegrate saved forces and recompute physical costs independently."""
    q,v=replay_controls(trace['lengths'],trace['qpos'][0],trace['qvel'][0],trace['applied_loads'],float(trace['dt']))
    np.testing.assert_array_equal(q,trace['qpos']);np.testing.assert_array_equal(v,trace['qvel'])
    accounting=actual_accounting(trace['lengths'],q,v,trace['applied_loads'],float(trace['dt']))
    for key,value in accounting.items():
        if key in expected and not isinstance(value,dict):
            np.testing.assert_allclose(value,expected[key],atol=1e-9,rtol=0)
    return {'bitwise_replay_pass':True,'accounting_pass':True,
            'speed_kmh':accounting['speed_kmh'],'total_net_work_j':accounting['total_net_work_j']}


def refit(q,dt,lengths,controls,baselines,fallback,frozen,out):
    """Two independently budgeted starts; retain every failed iterate."""
    records=[];messages=[];basis=temporal_basis(len(q),dt)
    class Limit(Exception):
        pass
    for start_index,start in enumerate((np.zeros(33),frozen)):
        cache={};count=0
        def ev(x):
            nonlocal count
            key=tuple(np.asarray(x,float))
            if key in cache:return cache[key]
            if count>=120:raise Limit()
            pair,_=pair_evaluation(q,dt,lengths,applied_controls(controls,basis,x),baselines,x)
            pair.update(evaluation_id=len(records),start_index=start_index)
            records.append(pair);cache[key]=pair;count+=1
            if count%40==0:print('refit',out.name,'start',start_index,'pairs',count,flush=True)
            return pair
        try:
            opt=minimize(lambda x:-ev(x)['fine']['paired_gain_kmh'],start,method='SLSQP',
                         bounds=[(-1.,1.)]*33,
                         constraints=[{'type':'ineq','fun':lambda x:raw_margins(baselines,ev(x))}],
                         options={'maxiter':12,'ftol':1e-12,'eps':1e-5})
            ev(opt.x);message={'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit)}
        except Limit:
            message={'success':False,'message':'120 paired-evaluation limit reached; not an impossibility proof'}
        messages.append(dict(start_index=start_index,pairs=count,**message))
        (out/'refit-partial.json').write_text(json.dumps({'messages':messages,'records':records},allow_nan=False),encoding='utf8')
    selected=select_resolved([fallback]+records,POLICIES[0])
    return selected,{'messages':messages,'paired_evaluations':len(records),'records':records,
                     'selected_evaluation_id':selected.get('evaluation_id'),
                     'baseline_fallback':not bool(np.any(selected['parameters']))}


def run_pitch(pitch,lab,out,frozen,source_controls,source_modified,source_dt,subject_lengths,do_refit):
    """One isolated recipient calculation with fixed-transfer and refit reports."""
    out=Path(out);q,dt,lengths,controls,ledger=prepare(pitch,lab,out,subject_lengths)
    baselines={};extras={}
    for sub in (1,2):
        b,bp,bv,tr=shield_rollout(lengths,q,controls,dt,sub)
        b.update(group_work(tr['applied_loads'],bv,dt/sub));baselines[sub]=b,bp,bv
        extras[sub]={'summary':b,'root_peak':float(np.linalg.norm(tr['applied_loads'][:,:3],axis=1).max())}
        np.savez_compressed(out/f'baseline-sub{sub}.npz',**tr)
    fallback,_=pair_evaluation(q,dt,lengths,controls,baselines,np.zeros(33))
    absolute,clip=absolute_transfer(source_controls,source_modified,source_dt,controls,dt)
    transfers={}
    for name,applied in (('normalized',applied_controls(controls,temporal_basis(len(q),dt),frozen)),('absolute',absolute)):
        pair,traces=pair_evaluation(q,dt,lengths,applied,baselines,frozen)
        pair['judgement']=transfer_pass(pair,fallback)
        pair['extra']=extra_diagnostics(extras[2],traces[2],pair['fine'])
        pair['verification']=independent_check(traces[2],pair['fine'])
        np.savez_compressed(out/f'{name}-sub2.npz',**traces[2])
        if name=='absolute':pair['clipping']=clip
        transfers[name]=pair
    report={'pitch':pitch,'ledger':ledger,'baseline':fallback,'transfers':transfers,
            'real_athlete_effect_verified':False,'effect_95ci':None,'physiological_capacity_validated':False}
    # Save transfer first so interrupted refitting cannot erase completed evidence.
    (out/'direct.json').write_text(json.dumps(report,allow_nan=False,indent=2),encoding='utf8')
    print('direct',pitch,[(k,round(v['fine']['paired_gain_kmh'],6),v['judgement']['pass']) for k,v in transfers.items()],flush=True)
    if do_refit:
        selected,optimization=refit(q,dt,lengths,controls,baselines,fallback,frozen,out)
        pair,traces=pair_evaluation(q,dt,lengths,applied_controls(controls,temporal_basis(len(q),dt),selected['parameters']),baselines,selected['parameters'])
        pair['judgement']=transfer_pass(pair,fallback)
        pair['extra']=extra_diagnostics(extras[2],traces[2],pair['fine'])
        pair['verification']=independent_check(traces[2],pair['fine'])
        np.savez_compressed(out/'refit-sub2.npz',**traces[2])
        report['refit']={'selected':pair,'optimization':optimization}
    (out/'report.json').write_text(json.dumps(report,allow_nan=False,indent=2),encoding='utf8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--pitch',choices=COHORT);parser.add_argument('--refit',action='store_true')
    a=parser.parse_args();lab,out=Path(a.lab),Path(a.out)
    frozen,source_pair,source_hash=load_frozen_candidate()
    source=np.load(lab/CACHED['3034_1']);sq,sd,sl=source['qpos'],float(source['dt']),source['lengths']
    sc=inverse_controls(model_for(sl,sd),sq,sd);sm=applied_controls(sc,temporal_basis(len(sq),sd),frozen)
    pitches=(a.pitch,) if a.pitch else COHORT
    for pitch in pitches:
        if pitch.startswith('3034'):lengths=sl
        else:
            first=pitch.split('_')[0]+'_1';cached=out/first/'input.npz'
            if cached.exists():lengths=np.load(cached)['lengths']
            else:lengths=ObservedPitch(first,lab/'openbiomechanics/baseball_pitching/data').lengths
        dest=out/pitch
        if (dest/'report.json').exists():raise FileExistsError('Completed result must not be overwritten: '+str(dest))
        report=run_pitch(pitch,lab,dest,frozen,sc,sm,sd,lengths,a.refit and pitch in COHORT[1:5])
        report.update(protocol_sha256=hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      source_report_sha256=source_hash)
        if pitch=='3034_1':
            for key in ('coarse','fine'):
                np.testing.assert_allclose(report['transfers']['normalized'][key]['speed_kmh'],source_pair[key]['speed_kmh'],rtol=0,atol=1e-8)
        (dest/'report.json').write_text(json.dumps(report,allow_nan=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
