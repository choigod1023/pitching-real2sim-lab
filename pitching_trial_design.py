"""Executable trial condition/order templates and explicitly hypothetical power sensitivity."""
import json
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parent


def williams_order(seed=20261028):
    labels=[''.join(str(b) for b in bits) for bits in product((0,1),repeat=3)]
    rng=np.random.default_rng(seed);labels=np.array(labels)[rng.permutation(8)]
    base=np.array([0,1,7,2,6,3,5,4])
    return labels[np.array([(base+i)%8 for i in rng.permutation(8)])]


def assess_daily_contrasts(contrasts,speed_threshold=1.0):
    """Three co-endpoints, conservative one-sided Bonferroni bounds on independent days.

    Columns: speed km/h, elbow Nm, shoulder Nm. This does not establish unchanged
    muscle force/total work and cannot replace randomization or reliable measurements.
    """
    x=np.asarray(contrasts,float)
    if x.ndim!=2 or x.shape[1]!=3 or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('Need at least three finite independent-day contrasts for three endpoints')
    mean=x.mean(axis=0);se=x.std(axis=0,ddof=1)/np.sqrt(len(x));critical=student_t.ppf(1-.05/3,len(x)-1)
    lower=mean-critical*se;upper=mean+critical*se
    return dict(mean=mean.tolist(),lower=lower.tolist(),upper=upper.tolist(),
                speed_threshold_kmh=speed_threshold,
                limited_proxy_joint_criterion_pass=bool(lower[0]>speed_threshold and upper[1]<=0 and upper[2]<=0),
                full_strength_work_hypothesis_verified=False)


def power_sensitivity(sigma,n_days,effect,repetitions=10000,seed=20261028):
    sigma,effect=np.asarray(sigma,float),np.asarray(effect,float)
    if sigma.shape!=(3,) or effect.shape!=(3,) or np.any(sigma<=0) or not np.isfinite(sigma).all() or not np.isfinite(effect).all() or n_days<3:raise ValueError('Invalid scenario')
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(repetitions,n_days,3))*sigma+effect
    mean=x.mean(axis=1);se=x.std(axis=1,ddof=1)/np.sqrt(n_days);critical=student_t.ppf(1-.05/3,n_days-1)
    passed=(mean[:,0]-critical*se[:,0]>1)&(mean[:,1]+critical*se[:,1]<=0)&(mean[:,2]+critical*se[:,2]<=0)
    return float(passed.mean())


def analyze_measurements(frame,candidate='111'):
    """Aggregate pitches→paired block contrasts→days, never treating pitches as days.

    Only one prespecified final candidate vs 000 is tested. The input cannot
    prove randomization, sensor reliability, muscle work or absence of carryover.
    """
    columns=['athlete_id','day_id','paired_block_id','condition','pitch_id','measurement_qc',
             'speed_kmh','elbow_peak_nm','shoulder_peak_nm']
    if any(c not in frame for c in columns):raise ValueError('Missing measurement columns')
    if candidate not in {'001','010','011','100','101','110','111'}:raise ValueError('Invalid fixed candidate')
    f=frame.copy()
    if f.empty or f.athlete_id.nunique()!=1 or f[columns[:5]].isna().any().any():raise ValueError('Need one athlete and complete IDs')
    f['condition']=f.condition.astype(str).str.zfill(3)
    if not f.condition.isin([''.join(map(str,b)) for b in product((0,1),repeat=3)]).all():raise ValueError('Unknown condition label')
    if f.duplicated(['athlete_id','day_id','pitch_id']).any():raise ValueError('Duplicate pitch identity')
    qa=f.measurement_qc.eq('pass');excluded=int((~qa).sum());f=f[qa & f.condition.isin(['000',candidate])]
    endpoints=['speed_kmh','elbow_peak_nm','shoulder_peak_nm']
    if not np.isfinite(f[endpoints].to_numpy(float)).all():raise ValueError('Nonfinite accepted measurements')
    block=[];incomplete=0
    for (day,pair),g in f.groupby(['day_id','paired_block_id']):
        means=g.groupby('condition')[endpoints].mean()
        if not {'000',candidate}.issubset(means.index):incomplete+=1;continue
        delta=(means.loc[candidate]-means.loc['000']).to_dict()
        block.append(dict(day_id=day,paired_block_id=pair,**delta))
    if not block:raise ValueError('No complete baseline/candidate paired blocks')
    daily=pd.DataFrame(block).groupby('day_id')[endpoints].mean()
    result=dict(candidate=candidate,qc_excluded_pitches=excluded,complete_blocks=len(block),
                incomplete_blocks=incomplete,independent_days_assumed=len(daily),
                daily_contrasts=daily.reset_index().to_dict('records'),
                inference=assess_daily_contrasts(daily.to_numpy()) if len(daily)>=3 else None,
                raw_pitches_are_not_independent_days=True,
                randomization_reliability_and_carryover_require_external_validation=True,
                muscle_force_total_work_and_causal_hypothesis_verified=False)
    return result


def main():
    out=ROOT/'artifacts/hypothesis-test';out.mkdir(parents=True,exist_ok=True)
    conditions=[]
    for bits in product((0,1),repeat=3):
        conditions.append(dict(condition=''.join(map(str,bits)),A_knee_extension_deg=1.8573*bits[0],
                               B_trunk_lateral_deg=-1.948*bits[1],B_forearm_global_deg=2.0313*bits[1],
                               C_rear_knee_delay_ms=1000/360*2*bits[2],scope='target reference only, not proven coaching prescription'))
    pd.DataFrame(conditions).to_csv(out/'conditions.csv',index=False)
    schedule=[]
    for cycle,order in enumerate(williams_order(),1):
        for position,condition in enumerate(order,1):
            schedule.append(dict(cycle=cycle,position=position,condition=condition))
    pd.DataFrame(schedule).to_csv(out/'balanced_order.csv',index=False)
    columns=['athlete_id','day_id','paired_block_id','condition','assigned_order','pitch_id','video_id','speed_kmh',
             'elbow_peak_nm','shoulder_peak_nm','knee_extension_deg','trunk_lateral_br_deg','forearm_global_deg',
             'rear_knee_peak_br_ms','target_x_m','target_z_m','arrival_x_m','arrival_z_m','spin_rpm','spin_axis_3d',
             'joint_work_j','total_work_validated','effort_rating','discomfort_rating','measurement_qc','exclusion_reason']
    pd.DataFrame(columns=columns).to_csv(out/'measurement_template.csv',index=False)
    p=pd.read_csv(ROOT.parent/'openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv')
    v=p[p.session.eq(3034)][['pitch_speed_mph','elbow_varus_moment','shoulder_internal_rotation_moment']].copy()
    v.pitch_speed_mph*=1.609344
    sigma=v.std(ddof=1).to_numpy()*np.sqrt(2)
    scenarios=[]
    for days in (8,12,16,24):
        for effect in ((0,0,0),(1,0,0),(1.77,0,0),(1.77,-1,-2),(1.77,-3,-3)):
            scenarios.append(dict(independent_days=days,assumed_effect=list(effect),
                                  simulated_joint_pass_fraction=power_sensitivity(sigma,days,effect)))
    report=dict(scope='Design sensitivity only: synthetic independent normal daily contrasts, not athlete trial results or required sample size',
                variance_source='sqrt(2) times sample SD of five observed 3034 pitches; NOT device error or measured day-level variance',
                assumed_daily_contrast_sd=dict(zip(['speed_kmh','elbow_nm','shoulder_nm'],sigma.tolist())),
                normal_errors_and_independent_endpoints_assumed=True,
                criterion='lower speed bound >1 km/h AND upper elbow/shoulder bounds <=0; one-sided alpha .05/3 each',
                repetitions_per_scenario=10000,scenarios=scenarios,
                schedule_is_pitch_volume_recommendation=False,human_trial_executed=False)
    (out/'trial_design.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('hypothetical daily contrast SD',sigma.tolist())
    for s in scenarios:
        if s['independent_days'] in (8,24):print(s)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--measurements',type=Path)
    ap.add_argument('--candidate',default='111')
    ap.add_argument('--result',type=Path,default=Path('artifacts/hypothesis-test/measured_trial_analysis.json'))
    args=ap.parse_args()
    if args.measurements:
        r=analyze_measurements(pd.read_csv(args.measurements,dtype={'condition':str}),args.candidate)
        args.result.parent.mkdir(parents=True,exist_ok=True)
        args.result.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(r['inference']))
    else:main()
