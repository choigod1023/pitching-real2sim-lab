"""Small state-conditioned input policy; no claim of anatomical prescription."""
import numpy as np

from coach_hypotheses.budgeted_input_search.search import (
    temporal_basis,applied_controls,cost_vector,policy_gate,POLICIES)
from coach_hypotheses.pitch_transfer.study import transfer_pass

TRAIN=('3034_1','3034_2','3034_3')
EVALUATION=('3034_4','3034_5')
PARTS=(slice(6,7),slice(7,8),slice(8,11))


def state_features(trace):
    """Observe only the last saved baseline state at/before BR minus 60 ms."""
    q,v=np.asarray(trace['qpos']),np.asarray(trace['qvel'])
    dt=float(trace['dt'])
    if q.ndim!=2 or q.shape[1]!=13 or v.shape!=(len(q),11) or dt<=0 or not np.isfinite(dt):
        raise ValueError('Matching thirteen/eleven state arrays and positive dt required')
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Finite states required')
    times=(np.arange(len(q))-len(q)+1)*dt
    valid=np.flatnonzero(times<=-.060)
    if not len(valid):raise ValueError('No unchanged preintervention state')
    i=int(valid[-1])
    return np.array([q[i,7],v[i,6],np.linalg.norm(v[i,8:11])]),i


def fit_scaler(training):
    """Fit three feature scales and support bounds from training only."""
    x=np.asarray(training,float)
    if x.ndim!=2 or x.shape[1]!=3 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('At least two finite three-feature training rows required')
    return {'mean':x.mean(0).tolist(),'scale':np.maximum(x.std(0),1e-6).tolist(),
            'minimum':x.min(0).tolist(),'maximum':x.max(0).tolist(),'training_count':len(x)}


def transform(features,scaler):
    """Return bounded features and abstain flag without refitting on recipients."""
    x=np.asarray(features,float)
    mean,scale,lo,hi=[np.asarray(scaler[k],float) for k in ('mean','scale','minimum','maximum')]
    if any(a.shape!=(3,) for a in (x,mean,scale,lo,hi)) or not all(np.isfinite(a).all() for a in (x,mean,scale,lo,hi)) or np.any(scale<=0) or np.any(lo>hi):
        raise ValueError('Finite three-feature values and valid frozen scaler required')
    supported=bool(np.all(x>=lo-1e-12) and np.all(x<=hi+1e-12))
    return np.clip((x-mean)/scale,-2,2),supported


def policy_controls(controls,dt,features,scaler,weights,direction):
    """One twelve-weight rule; external root inputs remain bitwise unchanged."""
    w,direction=np.asarray(weights,float),np.asarray(direction,float)
    if w.shape!=(12,) or direction.shape!=(33,) or not np.isfinite(w).all() or not np.isfinite(direction).all() or np.max(np.abs(w))>1+1e-12 or np.max(np.abs(direction))>1+1e-12:
        raise ValueError('Bounded twelve weights and frozen thirty-three direction required')
    z,supported=transform(features,scaler)
    amplitude=np.tanh(w.reshape(3,4)@np.r_[1.,z]) if supported else np.zeros(3)
    coefficients=np.zeros((11,3));source=direction.reshape(11,3)
    for part,a in zip(PARTS,amplitude):coefficients[part]=a*source[part]
    out=applied_controls(controls,temporal_basis(len(controls),dt),coefficients.ravel())
    out[:,:6]=controls[:,:6]
    return out,coefficients.ravel(),{'supported':supported,'abstained':not supported,
                                   'standardized_features':z.tolist(),'amplitudes':amplitude.tolist()}


def work_costs(result):
    """Union A/B actual work costs, excluding analytically capped peaks."""
    a,b=cost_vector(result,POLICIES[0]),cost_vector(result,POLICIES[1])
    return np.r_[a[:22],a[-2:],b[15:19],b[21:24]]


def C_gate(base,candidate,root_base,root_candidate):
    """Full A AND full B AND root-force-vector peak, never physiology validation."""
    a=policy_gate(base,candidate,POLICIES[0]);b=policy_gate(base,candidate,POLICIES[1])
    delta=float(root_candidate-root_base)
    if not np.isfinite(delta):raise ValueError('Finite root peaks required')
    return {'pass':bool(a['strict_model_budget_pass'] and b['policy_budget_pass'] and delta<=1e-9),
            'A':a,'B':b,'root_norm_peak_change_n':delta,
            'total_net_work_change_j':candidate['total_net_work_j']-base['total_net_work_j'],
            'physiological_capacity_validated':False}


def assess(pair,baseline_pair):
    """Keep C, numerical convergence and positive gain required together."""
    result=transfer_pass(pair,baseline_pair)
    result['pass']=bool(result['pass'] and pair['coarse']['C']['pass'] and pair['fine']['C']['pass'])
    return result


def training_pass(record):
    """Only all-training feasibility can select a policy; evaluation is absent."""
    return bool(set(record['pitches'])==set(TRAIN) and np.any(record['weights']) and
                all(r['assessment']['pass'] for r in record['pitches'].values()))


def choose(records):
    """Select maximin training improvement or the unchanged-policy fallback."""
    baseline=next(r for r in records if not np.any(r['weights']))
    eligible=[r for r in records if training_pass(r)]
    return max(eligible,key=lambda r:min(p['fine']['paired_gain_kmh'] for p in r['pitches'].values())) if eligible else baseline
