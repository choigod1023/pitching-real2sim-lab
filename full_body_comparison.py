"""Fixed-athlete observational full-body comparison; no causal gain claims."""
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from pilot import regular_time, vector

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'openbiomechanics/baseball_pitching/data'
OUT=ROOT/'artifacts/full-body'
PEAKS={'lead_knee':'lead_knee_velo_x','rear_knee':'rear_knee_velo_x',
       'lead_hip':'lead_hip_velo_x','rear_hip':'rear_hip_velo_x',
       'pelvis':'pelvis_velo_z','torso':'torso_velo_z'}
POI=['stride_length','lead_knee_extension_from_fp_to_br','max_pelvis_rotational_velo',
     'max_torso_rotational_velo','rotation_hip_shoulder_separation_fp']
ANGLES=['lead_knee_angle_x','rear_knee_angle_x','pelvis_angle_z','torso_angle_z','torso_pelvis_angle_z']

def sample(t,y,q):
    q=np.asarray(q)
    if np.any(q<t[0]) or np.any(q>t[-1]): raise ValueError('Extrapolation prohibited')
    y=np.asarray(y)
    if y.ndim==1:return np.interp(q,t,y)
    return np.stack([np.interp(q,t,y[:,k]) for k in range(y.shape[1])],axis=-1)

def peak(t,y,start,end):
    ids=np.flatnonzero((t>=start-.000051)&(t<=end+.000051))
    if not len(ids):raise ValueError('Empty event window')
    k=ids[np.argmax(np.abs(np.asarray(y)[ids]))]
    return float(t[k]),float(y[k])

def pair_summary(features,cols,fast,slow):
    by=features.set_index('session_pitch');rows=[]
    for c in cols:
        delta=float(by.loc[fast,c]-by.loc[slow,c]); signs=[]
        tolerance=1000/360 if c.endswith('_ms') else 1e-8
        for _,(a,b) in enumerate(combinations(features.to_dict('records'),2)):
            f,s=(a,b) if a['speed_kmh']>b['speed_kmh'] else (b,a)
            if f['speed_kmh']==s['speed_kmh']:continue
            diff=f[c]-s[c];signs.append(0. if abs(diff)<=tolerance else float(np.sign(diff)))
        rows.append(dict(feature=c,slow=float(by.loc[slow,c]),fast=float(by.loc[fast,c]),
                         fast_minus_slow=delta,same_direction_pairs=int(sum(x==np.sign(delta) for x in signs)) if abs(delta)>tolerance else 0,unresolved_pairs=int(sum(x==0 for x in signs)),direction_tolerance=tolerance,pairs=len(signs)))
    return rows

def load():
    meta=pd.read_csv(DATA/'metadata.csv');meta=meta[(meta.user==2)&(meta.session==3034)].copy()
    ids=meta.session_pitch.tolist(); tables={}
    for name in ['joint_angles','joint_velos','landmarks']:
        parts=[]
        for d in pd.read_csv(DATA/f'full_sig/{name}.csv',chunksize=50000):
            parts.append(d[d.session_pitch.isin(ids)])
        tables[name]=pd.concat(parts,ignore_index=True)
    poi=pd.read_csv(DATA/'poi/poi_metrics.csv');return meta,poi[poi.session_pitch.isin(ids)],tables

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    meta,poi,tables=load();rows=[];sensitivity=[]
    for m in meta.to_dict('records'):
        pid=m['session_pitch'];g=tables['joint_velos'].query('session_pitch == @pid').reset_index(drop=True)
        a=tables['joint_angles'].query('session_pitch == @pid').reset_index(drop=True)
        t=regular_time(g);ta=regular_time(a);fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0]);fp100=float(g.fp_100_time.iloc[0])
        r=dict(session_pitch=pid,speed_kmh=m['pitch_speed_mph']*1.609344,fp_to_br_ms=(br-fp)*1000,fp100_minus_fp10_ms=(fp100-fp)*1000)
        p=poi.set_index('session_pitch').loc[pid]
        for c in POI:r[c]=float(p[c])
        r['stride_cm']=r['stride_length']*m['session_height_m']*100
        for name,col in PEAKS.items():
            pt,v=peak(t,g[col],fp,br);r[name+'_peak_br_ms']=(pt-br)*1000;r[name+'_peak_fp_ms']=(pt-fp)*1000;r[name+'_peak_deg_s']=v;r[name+'_peak_fp100_ms']=(pt-fp100)*1000;r[name+'_boundary_peak']=float(min(abs(pt-fp),abs(pt-br))<=1/360+.000051)
            for shift in [-1,0,1]:
                st,sv=peak(t,g[col],fp+shift/360,br+shift/360)
                sensitivity.append(dict(session_pitch=pid,segment=name,boundary_shift_frames=shift,peak_time=st,peak_br_ms=(st-br)*1000))
        r['torso_minus_pelvis_ms']=r['torso_peak_br_ms']-r['pelvis_peak_br_ms']
        for c in ANGLES:
            for anchor,q in [('fp',fp),('br',br)]:r[c+'_'+anchor]=float(sample(ta,a[c],q))
        rows.append(r)
    f=pd.DataFrame(rows).sort_values('session_pitch');fast=f.loc[f.speed_kmh.idxmax(),'session_pitch'];slow=f.loc[f.speed_kmh.idxmin(),'session_pitch']
    cols=[c for c in f if c not in ['session_pitch','speed_kmh']]
    summary=pair_summary(f,cols,fast,slow)
    report=dict(athlete_user=2,session=3034,n_pitches=len(f),fast=fast,slow=slow,
                speed_difference_kmh=float(f.speed_kmh.max()-f.speed_kmh.min()),sample_ms=1000/360,
                comparison=summary,interpretation='Observational association only; no causal gain or equal work/force claim')
    f.to_csv(OUT/'all_five_pitches.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'fast_slow.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'window_sensitivity.csv',index=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report,tables

if __name__=='__main__':
    r,_=run()
    print(json.dumps({k:v for k,v in r.items() if k!='comparison'}))
    for c in r['comparison']:
        if c['feature'] in ['stride_cm','lead_knee_extension_from_fp_to_br','torso_minus_pelvis_ms'] or c['feature'].endswith('_peak_br_ms'):print(c)

