"""Observed elbow timing bridge: conditional mechanics, not a causal predictor."""
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from coupled_payload import model_for
from torque_arm import derivatives
from capacity_budget_gate import rollout, gate
from joint_intervention import inverse_controls, rotation_differences, initial_energy
from coach_hypotheses.model_retirement.policy import legacy_diagnostic_only, model_usage_metadata, RetiredPredictionModelError


def elbow_peak(q, dt):
    return int(np.argmax(np.abs(np.gradient(q[:,7],dt))))


def warp_elbow(q,dt,delay):
    """Endpoint-anchored warp with frozen first three samples; no extrapolation."""
    if len(q)<8 or dt<=0 or not np.isfinite(delay): raise ValueError('Invalid inputs')
    t=np.arange(len(q))*dt; s=np.clip((t-3*dt)/(t[-1]-3*dt),0,1)
    peak=elbow_peak(q,dt); p=s[peak]
    if not 0<p<1: raise ValueError('Interior baseline peak required')
    h=s**3*(1-s)/(p**3*(1-p)); mapped=t-delay*h
    if np.any(np.diff(mapped)<=0) or mapped.min()<0 or mapped.max()>t[-1]:
        raise ValueError('Time warp is non-monotonic or outside measured support')
    changed=q.copy(); changed[:,7]=CubicSpline(t,q[:,7])(mapped)
    changed[:4]=q[:4]; changed[-1]=q[-1]
    return changed,mapped


@legacy_diagnostic_only
def run(out='artifacts/observed-candidate',data='../openbiomechanics/baseball_pitching/data'):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    from coupled_payload import run as prepare
    for pitch in ('3034_4','3034_2'):
        reference=out/f'reference-{pitch}'/'reference-0.npz'
        if not reference.exists(): prepare(Path(data),pitch,reference.parent,20.,(1/11520,))
    records=[]; baselines={}; baseline_positions={}
    specs=[('slow_baseline','3034_4',0),('fast_observed','3034_2',0)]
    specs += [(f'elbow_fraction_{fraction:+g}','3034_4',fraction/360) for fraction in (-1,-.5,-.25,.25,.5,1,2)]
    for case,pitch,delay in specs:
        state=np.load(out/f'reference-{pitch}'/'reference-0.npz')
        q=state['qpos']; dt=float(state['dt']); lengths=state['lengths']
        model=model_for(lengths,dt)
        try:
            changed,mapped=warp_elbow(q,dt,delay) if delay else (q.copy(),np.arange(len(q))*dt)
        except ValueError as e:
            records.append(dict(case=case,invalid_reason=str(e))); continue
        controls=inverse_controls(model,changed,dt)
        v,_=derivatives(model,changed,dt); v0,_=derivatives(model,q,dt)
        for substeps in (1,2):
            result,pos,vel=rollout(lengths,changed,controls,dt,substeps)
            if case=='slow_baseline': baselines[substeps]=result.copy();baseline_positions[substeps]=pos.copy()
            base=baselines[substeps]
            comparison=baseline_positions[substeps] if pitch=='3034_4' else pos
            rotations=rotation_differences(model,comparison,pos)
            result.update(case=case,pitch=pitch,requested_delay_ms=delay*1000,
                measured_reference_speed_kmh={'3034_4':79.8,'3034_2':81.4}[pitch]*1.609344,
                speed_difference_from_slow_kmh=result['speed_kmh']-base['speed_kmh'],
                positive_work_change_from_slow_j=result['total_positive_component_work_j']-base['total_positive_component_work_j'],
                shoulder_translation_work_change_from_slow_j=result['shoulder_translation_net_work_j']-base['shoulder_translation_net_work_j'],
                budget_gate=gate(base,result),
                max_actual_joint_rotation_change_deg=rotations,
                max_reference_tracking_rotation_error_deg=rotation_differences(model,changed,pos),
                small_change_screen_pass=max(rotations.values())<=3 if pitch=='3034_4' else None,
                baseline_model_peak_before_br_ms=(elbow_peak(q,dt)-(len(q)-1))*dt*1000,
                achieved_reference_peak_shift_ms=(elbow_peak(changed,dt)-elbow_peak(q,dt))*dt*1000,
                achieved_rollout_peak_before_br_ms=(int(np.argmax(np.abs(vel[:,6])))-(len(q)-1))*dt*1000,
                initial_qpos_equal=bool(np.array_equal(changed[0],q[0])),initial_qvel_equal=bool(np.array_equal(v[0],v0[0])),
                initial_energy_j=initial_energy(model,changed[0],v[0]),
                peak_comparison_scope='Geometric model hinge; distinct from measured Visual3D elbow x',
                baseline_comparison_valid_for_candidate=pitch=='3034_4')
            if pitch!='3034_4':
                result['budget_gate']=None
                result['max_actual_joint_rotation_change_deg']=None
                result['initial_state_comparison_scope']='Different observed pitch: no same-ability comparison'
            np.savez_compressed(out/f'state-{case}-sub{substeps}.npz',qreference=q,candidate_q=changed,qpos=pos,qvel=vel,controls=controls,dt=dt,lengths=lengths,mapped_time=mapped)
            records.append(result)
        print(case,round(result['speed_kmh'],6),round(result['speed_difference_from_slow_kmh'],6),result['budget_gate']['strict_model_budget_pass'] if result['budget_gate'] else 'not_applicable',flush=True)
    report=dict(protocol='OBSERVED_CANDIDATE_SIMULATION.md',measured_fast_minus_slow_kmh=(81.4-79.8)*1.609344,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('OBSERVED_CANDIDATE_SIMULATION.md'),*[out/f'reference-{pitch}'/'reference-0.npz' for pitch in ('3034_4','3034_2')]]},
        scope='Inverse-demand trajectory interpolation, actual RK4; NOT equal force inputs or intervention truth.',runs=records)
    report.update(model_usage_metadata())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',default='artifacts/observed-candidate'); p.add_argument('--data',default='../openbiomechanics/baseball_pitching/data')
    p.add_argument('--legacy-diagnostic',action='store_true');args=p.parse_args()
    try:run(args.out,args.data,legacy_diagnostic=args.legacy_diagnostic)
    except RetiredPredictionModelError as error:p.error(str(error))
