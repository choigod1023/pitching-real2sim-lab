"""Observed whole-body/energy differences under three arm-load proxy screens."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from within_athlete_candidates import extract as original_validity, PEAKS as ORIGINAL_PEAKS
from full_body_comparison import sample, peak
from pilot import regular_time

POI = {
 'lead_knee_extension_deg':'lead_knee_extension_from_fp_to_br',
 'separation_fp_deg':'rotation_hip_shoulder_separation_fp',
 'arm_slot_deg':'arm_slot', 'trunk_forward_br_deg':'torso_anterior_tilt_br',
 'trunk_lateral_br_deg':'torso_lateral_tilt_br', 'elbow_flexion_fp_deg':'elbow_flexion_fp',
 'max_shoulder_external_rotation_deg':'max_shoulder_external_rotation',
 'max_pelvis_deg_s':'max_pelvis_rotational_velo', 'max_trunk_deg_s':'max_torso_rotational_velo',
 'trunk_after_pelvis_poi_ms':'timing_peak_torso_to_peak_pelvis_rot_velo',
 'trunk_to_arm_transfer_j':'thorax_distal_transfer_fp_br',
 'pelvis_to_trunk_transfer_j':'pelvis_lumbar_transfer_fp_br',
 'shoulder_transfer_j':'shoulder_transfer_fp_br', 'elbow_transfer_j':'elbow_transfer_fp_br',
 'lead_knee_generation_j':'lead_knee_generation_fp_br', 'lead_hip_generation_j':'lead_hip_generation_fp_br',
 'rear_knee_generation_j':'rear_knee_generation_pkh_fp', 'rear_hip_generation_j':'rear_hip_generation_pkh_fp',
}
LOADS=['elbow_varus_moment','shoulder_internal_rotation_moment']
GEN=['shoulder_generation_fp_br','elbow_generation_fp_br','lead_knee_generation_fp_br',
     'lead_hip_generation_fp_br','rear_knee_generation_pkh_fp','rear_hip_generation_pkh_fp']
CURVES={'lead_knee_deg':'lead_knee_angle_x','rear_knee_deg':'rear_knee_angle_x',
        'pelvis_z_deg':'pelvis_angle_z','trunk_z_deg':'torso_angle_z','separation_z_deg':'torso_pelvis_angle_z'}
VELS={'pelvis':'pelvis_velo_z','trunk':'torso_velo_z','elbow':'elbow_velo_x','rear_knee':'rear_knee_velo_x'}


def read_selected(path, ids, columns=None):
    parts=[]
    for g in pd.read_csv(path,usecols=columns,chunksize=40000):
        parts.append(g[g.session_pitch.isin(ids)])
    return pd.concat(parts,ignore_index=True)


def pair_records(pitches):
    if pitches.session_pitch.duplicated().any():raise ValueError('Duplicate pitch identity')
    numeric=['speed_kmh',*LOADS,'arm_generation_j','partial_six_joint_generation_j',*POI,'stride_cm']
    if not np.isfinite(pitches[numeric].to_numpy(float)).all():raise ValueError('Nonfinite pitch features')
    pairs=[]
    for key,g in pitches.groupby(['user','session','pitch_type'],sort=True):
        for a,b in combinations(g.sort_values('session_pitch').to_dict('records'),2):
            if a['speed_kmh']==b['speed_kmh']:continue
            fast,slow=(a,b) if a['speed_kmh']>b['speed_kmh'] else (b,a)
            row=dict(user=int(key[0]),session=int(key[1]),pitch_type=key[2],
                     fast=fast['session_pitch'],slow=slow['session_pitch'],
                     speed_gain_kmh=fast['speed_kmh']-slow['speed_kmh'])
            for k in [*LOADS,'arm_generation_j','partial_six_joint_generation_j',*POI,'stride_cm']:
                row[k]=fast[k]-slow[k]
            row['eligible']=(row['speed_gain_kmh']>=1.0 and row[LOADS[0]]<=0
                             and row[LOADS[1]]<=0 and row['arm_generation_j']<=0)
            pairs.append(row)
    return pd.DataFrame(pairs)


def clustered_summary(pairs, features):
    """Equal weight per athlete. No pair independence assumption or causal inference."""
    out=[]
    for feature in features:
        x=pairs.groupby('user',sort=True)[feature].mean().to_numpy(float)
        if len(x)==0 or not np.isfinite(x).all():raise ValueError('Empty/nonfinite comparison')
        rng=np.random.default_rng(20261028)
        boot=x[rng.integers(0,len(x),(2000,len(x)))].mean(axis=1)
        tol=1000/360 if feature.endswith('_ms') else 1e-8
        out.append(dict(feature=feature,n_athletes=len(x),mean_delta=float(x.mean()),
                        median_athlete_delta=float(np.median(x)),
                        bootstrap_mean_95=np.percentile(boot,[2.5,97.5]).tolist(),
                        positive_athletes=int((x>tol).sum()),negative_athletes=int((x<-tol).sum()),
                        unresolved_athletes=int((np.abs(x)<=tol).sum()),direction_tolerance=tol))
    return out


def run(data, out):
    data,out=Path(data),Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(data/'metadata.csv');poi=pd.read_csv(data/'poi/poi_metrics.csv')
    if meta.session_pitch.duplicated().any() or poi.session_pitch.duplicated().any():raise ValueError('Duplicate ID')
    merged=meta.merge(poi.drop(columns=['pitch_speed_mph']),on=['session_pitch','session'],validate='one_to_one',how='left',indicator=True)
    if not merged['_merge'].eq('both').all():raise ValueError('Missing POI join')
    vcols=list(dict.fromkeys(['session_pitch','time','fp_10_time','fp_100_time','BR_time',
                             *ORIGINAL_PEAKS.values(),'wrist_velo_x','wrist_velo_y','wrist_velo_z',*VELS.values()]))
    velocities=read_selected(data/'full_sig/joint_velos.csv',set(meta.session_pitch),vcols)
    valid=[];excluded=[]
    for pid,g in velocities.groupby('session_pitch',sort=True):
        try:valid.append(dict(session_pitch=pid,**original_validity(g)))
        except ValueError as e:excluded.append(dict(pitch=pid,reason=str(e)))
    missing=set(meta.session_pitch)-set(velocities.session_pitch)
    excluded.extend(dict(pitch=p,reason='missing velocity records') for p in sorted(missing))
    pitches=merged[merged.session_pitch.isin([r['session_pitch'] for r in valid])].copy()
    pitches['speed_kmh']=pitches.pitch_speed_mph*1.609344
    pitches['arm_generation_j']=pitches[GEN[:2]].sum(axis=1,skipna=False)
    pitches['partial_six_joint_generation_j']=pitches[GEN].sum(axis=1,skipna=False)
    for name,col in POI.items():pitches[name]=pitches[col]*(1000 if name.endswith('_ms') else 1)
    pitches['stride_cm']=pitches.stride_length*pitches.session_height_m*100
    required=['speed_kmh',*LOADS,*POI,'stride_cm','arm_generation_j','partial_six_joint_generation_j']
    if not np.isfinite(pitches[required].to_numpy(float)).all():raise ValueError('Missing POI features')
    pairs=pair_records(pitches);selected=pairs[pairs.eligible].copy().sort_values(['user','slow','fast']).reset_index(drop=True)
    if selected.empty:raise ValueError('No eligible comparisons')
    ids=set(selected.fast)|set(selected.slow)
    angles=read_selected(data/'full_sig/joint_angles.csv',ids,['session_pitch','time',*CURVES.values()])
    vby={pid:g.sort_values('time').reset_index(drop=True) for pid,g in velocities[velocities.session_pitch.isin(ids)].groupby('session_pitch')}
    aby={pid:g.sort_values('time').reset_index(drop=True) for pid,g in angles.groupby('session_pitch')}
    events=[];profiles={};time_features={};phase=np.linspace(0,1,101)
    for pid in sorted(ids):
        g=vby[pid];a=aby[pid];t=regular_time(g);ta=regular_time(a)
        fp,br=float(g.fp_10_time.iloc[0]),float(g.BR_time.iloc[0]);q=fp+phase*(br-fp)
        # Preserve all actual times; normalized profiles do not rescale angular velocity.
        r=dict(fp_to_br_ms=(br-fp)*1000);curves={}
        for k,col in CURVES.items():
            y=a[col].to_numpy(float);near=(ta>=fp-1/360)&(ta<=br+1/360)
            if not np.isfinite(y[near]).all():raise ValueError(f'Nonfinite angle window {pid} {col}')
            curves[k]=sample(ta,np.degrees(np.unwrap(np.radians(y))),q)
        for name,col in VELS.items():
            pt,pv=peak(t,g[col],fp,br);r[name+'_peak_br_ms']=(pt-br)*1000
            r[name+'_peak_deg_s']=pv
            window=(t>=fp-1/360)&(t<=br+1/360)
            if not np.isfinite(g.loc[window,col]).all():raise ValueError(f'Nonfinite velocity window {pid}')
            curves[name+'_velocity_deg_s']=sample(t,g[col],q)
            events.append(dict(pitch=pid,segment=name,boundary_peak=min(abs(pt-fp),abs(pt-br))<=1/360+.000051))
        r['trunk_after_pelvis_recomputed_ms']=r['trunk_peak_br_ms']-r['pelvis_peak_br_ms']
        time_features[pid]=r;profiles[pid]=curves
    for k in next(iter(time_features.values())):
        selected[k]=[time_features[f][k]-time_features[s][k] for f,s in zip(selected.fast,selected.slow)]
    curve_summary={}
    for name in next(iter(profiles.values())):
        diffs=np.array([profiles[f][name]-profiles[s][name] for f,s in zip(selected.fast,selected.slow)])
        by_user=np.array([diffs[selected.user.to_numpy()==u].mean(axis=0) for u in sorted(selected.user.unique())])
        curve_summary[name]=dict(mean_delta=by_user.mean(axis=0).tolist(),median_delta=np.median(by_user,axis=0).tolist(),
                                 positive_athletes=(by_user>1e-8).sum(axis=0).tolist())
    features=['speed_gain_kmh',*LOADS,'arm_generation_j','partial_six_joint_generation_j',*POI,'stride_cm',*next(iter(time_features.values()))]
    summaries=clustered_summary(selected,features)
    sensitivity=[]
    sensitivity_features=['lead_knee_extension_deg','separation_fp_deg','trunk_lateral_br_deg','trunk_to_arm_transfer_j']
    for margin in (0,1,2):
        subset=selected[(selected[LOADS[0]]<=-margin)&(selected[LOADS[1]]<=-margin)]
        sensitivity.append(dict(required_peak_reduction_nm=margin,pairs=len(subset),athletes=subset.user.nunique(),
                                feature_deltas=clustered_summary(subset,sensitivity_features) if len(subset) else []))
    partial=selected[selected.partial_six_joint_generation_j<=0]
    first=selected.iloc[0];pb=pitches.set_index('session_pitch')
    example={key:pb.loc[first[key],required].to_dict() for key in ('slow','fast')}
    report=dict(scope='Exploratory observed same-athlete pairs; not causal, independent validation or equal strength/total work',
                original_pitches=len(meta),valid_pitches=len(pitches),excluded=excluded,
                all_distinct_speed_pairs=len(pairs),selected_pairs=len(selected),selected_athletes=selected.user.nunique(),
                unique_selected_pitches=len(ids),selection_fixed_before_feature_analysis=True,
                arm_peak_sensitivity=sensitivity,
                partial_generation_nonincrease=dict(pairs=len(partial),athletes=partial.user.nunique(),scope='six reported joints, mixed source integration windows, not total work'),
                partial_generation_subset_deltas=clustered_summary(partial,sensitivity_features) if len(partial) else [],
                descriptive_athlete_balanced_deltas=summaries,
                example=dict(user=int(first.user),slow=first.slow,fast=first.fast,values=example,
                             time_features={k:time_features[first[k]] for k in ('slow','fast')}),
                selected_pair_records=selected.to_dict('records'),
                boundary_peaks=events,phase_percent=(phase*100).tolist(),phase_differences=curve_summary,
                input_sha256={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [data/'metadata.csv',data/'poi/poi_metrics.csv',data/'full_sig/joint_velos.csv',data/'full_sig/joint_angles.csv']},
                muscle_force_or_total_work_validated=False,causal_velocity_gain_demonstrated=False,
                spin_or_command_evaluated=False,universal_motion_prescription_adopted=False)
    # Complete numerical and video inputs stay local. Compact results remain reviewable in Git.
    selected.to_csv(out/'pairs.csv',index=False)
    pitches[pitches.session_pitch.isin(ids)].to_csv(out/'pitches.csv',index=False)
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    compact={k:v for k,v in report.items() if k not in ('phase_differences','boundary_peaks')}
    (out/'summary.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'profiles.npz',phase=phase,**{f'{pid}__{name}':v for pid,c in profiles.items() for name,v in c.items()})
    print(json.dumps({k:report[k] for k in ('valid_pitches','all_distinct_speed_pairs','selected_pairs','selected_athletes','unique_selected_pitches','partial_generation_nonincrease')},ensure_ascii=False))
    display={'speed_gain_kmh',*LOADS,'arm_generation_j','lead_knee_extension_deg','separation_fp_deg','trunk_lateral_br_deg','trunk_to_arm_transfer_j'}
    for r in summaries:
        if r['feature'] in display:print(r['feature'],round(r['mean_delta'],3),[round(x,3) for x in r['bootstrap_mean_95']],r['positive_athletes'],r['negative_athletes'],r['unresolved_athletes'])
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    ap.add_argument('--out',type=Path,default=Path('artifacts/load-matched'))
    a=ap.parse_args();run(a.data,a.out)
