"""Predeclared multi-joint applied-input timing grid, without amplitude fitting."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import mujoco
import numpy as np
from capacity_budget_gate import shift_applied_intervals, rollout, gate
from coupled_payload import model_for
from coach_hypotheses.model_retirement.policy import legacy_diagnostic_only, model_usage_metadata, RetiredPredictionModelError

GROUPS = [('shoulder',slice(3,6)),('elbow',slice(6,7)),('wrist',slice(8,11))]
ROTATIONS = GROUPS[:2]+[('marker_roll',slice(7,8)),GROUPS[2]]


def shifted_controls(loads, shifts):
    if len(shifts)!=3:
        raise ValueError('Three group shifts required')
    controls=np.array(loads,copy=True)
    for (_,sl),shift in zip(GROUPS,shifts):
        controls=shift_applied_intervals(controls,sl,shift)
    return controls


def rotation_differences(model, baseline, candidate):
    dq=np.zeros(model.nv); differences=[]
    for a,b in zip(baseline,candidate):
        mujoco.mj_differentiatePos(model,dq,1.,a,b)
        differences.append(dq.copy())
    differences=np.asarray(differences)
    return {name:float(np.rad2deg(np.linalg.norm(differences[:,sl],axis=1)).max()) for name,sl in ROTATIONS}


def case_id(shifts):
    return '_'.join(f'{name}{shift}' for name,shift in zip('sew',shifts))


def select_case(runs):
    coarse={r['case_id']:r for r in runs if r['substeps']==1}
    fine={r['case_id']:r for r in runs if r['substeps']==2}
    baseline='s0_e0_w0'
    numerical_baseline=abs(fine[baseline]['speed_kmh']-coarse[baseline]['speed_kmh'])
    eligible=[]; small=[]
    for name,b in fine.items():
        if name==baseline: continue
        a=coarse[name]
        if a['small_motion_pass'] and b['small_motion_pass']:
            small.append(b)
            if a['budget_gate']['strict_model_budget_pass'] and b['budget_gate']['strict_model_budget_pass']:
                bound=numerical_baseline+abs(a['speed_kmh']-b['speed_kmh'])
                if b['speed_difference_from_baseline_kmh']>bound:
                    eligible.append(b)
    pool=eligible or small or [r for k,r in fine.items() if k!=baseline]
    selected=sorted(pool,key=lambda r:(-r['speed_kmh'],r['case_id']))[0]
    return selected['case_id'],('accepted_model_candidate' if eligible else 'rejected_diagnostic'),len(eligible)


@legacy_diagnostic_only
def run(source,out):
    source,out=Path(source),Path(out); out.mkdir(parents=True,exist_ok=True)
    data=np.load(source); q=data['qpos']; lengths=data['lengths']; loads=data['required_loads']; dt=float(data['dt'])
    model=model_for(lengths,dt)
    grid=[(0,0,0)]+[s for s in itertools.product(range(-2,3),repeat=3) if s!=(0,0,0)]
    runs=[]; bases={}
    for substeps in (1,2):
        for i,shifts in enumerate(grid):
            controls=shifted_controls(loads,shifts)
            result,pos,vel=rollout(lengths,q,controls,dt,substeps)
            name=case_id(shifts)
            if shifts==(0,0,0): bases[substeps]=(result.copy(),pos)
            base,base_pos=bases[substeps]
            rotations=rotation_differences(model,base_pos,pos)
            result.update(case_id=name,shifts_samples=list(shifts),shifts_ms=[s*dt*1000 for s in shifts],
                speed_difference_from_baseline_kmh=result['speed_kmh']-base['speed_kmh'],
                budget_gate=gate(base,result),max_rotation_difference_from_baseline_deg=rotations,
                small_motion_pass=max(rotations.values())<=3.,
                aggregate_only_diagnostic_pass=all(result[key]<=base[key]+1e-9 for key in (
                    'total_positive_component_work_j','total_negative_component_work_magnitude_j',
                    'shoulder_translation_net_work_j','shoulder_rotation_net_work_j')))
            result['record_minus_model_kmh']=130.1959296-result['speed_kmh']
            if substeps==2:
                np.savez_compressed(out/f'state-{name}-sub2.npz',qpos=pos,qvel=vel,controls=controls,dt=dt,lengths=lengths,qreference=q)
            runs.append(result)
            if i%25==0: print(f'substeps={substeps} completed={i+1}/125',flush=True)
    selected,status,count=select_case(runs)
    coarse={r['case_id']:r for r in runs if r['substeps']==1}
    fine={r['case_id']:r for r in runs if r['substeps']==2}
    resolution=[dict(case_id=name,speed_difference_fine_minus_coarse_kmh=b['speed_kmh']-coarse[name]['speed_kmh'],
        max_positive_work_component_difference_j=float(np.max(np.abs(np.array(b['positive_work_per_dof_j'])-coarse[name]['positive_work_per_dof_j']))),
        same_strict_gate_result=b['budget_gate']['strict_model_budget_pass']==coarse[name]['budget_gate']['strict_model_budget_pass']) for name,b in fine.items()]
    multi=[r for r in fine.values() if sum(s!=0 for s in r['shifts_samples'])>=2]
    report=dict(pitch='3034_1',athlete_id=2,source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(Path('TIMING_COMBINATION_PROTOCOL.md').read_bytes()).hexdigest(),
        dt_source_s=dt,recorded_speed_kmh=130.1959296,baseline_case_id='s0_e0_w0',selected_case_id=selected,selection_status=status,
        accepted_model_candidate_count=count,condition_count=125,modified_condition_count=124,
        strict_budget_pass_modified_fine=sum(r['budget_gate']['strict_model_budget_pass'] for n,r in fine.items() if n!='s0_e0_w0'),
        small_motion_pass_modified_fine=sum(r['small_motion_pass'] for n,r in fine.items() if n!='s0_e0_w0'),
        aggregate_only_and_small_motion_pass_modified_fine=sum(r['small_motion_pass'] and r['aggregate_only_diagnostic_pass'] for n,r in fine.items() if n!='s0_e0_w0'),
        simultaneous_multi_joint_condition_count=len(multi),
        simultaneous_multi_joint_small_motion_pass_count=sum(r['small_motion_pass'] for r in multi),
        runs=runs,resolution_comparison=resolution,
        limitations=['±2 source intervals is a very local ±0.1736 ms grid, not all possible timing changes.',
          'Generic inertia, 15cm rigid grip, no muscle capacity/contact or anatomical limits.',
          'Mechanical component work is coordinate dependent, not muscle work.',
          'No calibrated actual release/radar comparison and no zero-error claim.',
          'Zero-padding clips waveform boundaries; unchanged peaks do not guarantee unchanged work.'])
    report.update(model_usage_metadata())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','resolution_comparison')},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz')
    p.add_argument('--out',default='artifacts/multi-joint-timing')
    p.add_argument('--legacy-diagnostic',action='store_true')
    a=p.parse_args()
    try:run(a.source,a.out,legacy_diagnostic=a.legacy_diagnostic)
    except RetiredPredictionModelError as error:p.error(str(error))
