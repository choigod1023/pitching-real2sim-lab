"""Functional kinematic prediction audit, not an intervention simulator."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from speed_association_audit import load, DATA, ridge_fit
from pilot import regular_time

OUT=Path(__file__).resolve().parent/'artifacts/performance-restart'
CHANNELS=['pelvis_velo_z','torso_velo_z']+[f'{joint}_velo_{axis}' for joint in ['shoulder','elbow','wrist'] for axis in 'xyz']


def phase_samples(t,values,start,end):
    t=np.asarray(t);values=np.asarray(values)
    if start<t[0] or end>t[-1] or start>=end:
        raise ValueError('Invalid phase interval or non-finite motion')
    lo=max(0,np.searchsorted(t,start,side='right')-1)
    hi=min(len(t),np.searchsorted(t,end,side='left')+1)
    t,values=t[lo:hi],values[lo:hi]
    if not np.isfinite(values).all():raise ValueError('Missing samples inside phase interpolation support')
    times=np.linspace(start,end,21)
    return np.stack([np.interp(times,t,values[:,i]) for i in range(values.shape[1])],axis=1)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    frame,counts=load();wanted=set(frame.session_pitch)
    chunks=[g[g.session_pitch.isin(wanted)] for g in pd.read_csv(DATA/'full_sig/joint_velos.csv',chunksize=50000)]
    data=pd.concat(chunks,ignore_index=True); features={};bad=[]
    for pid,g in data.groupby('session_pitch'):
        g=g.sort_values('time');t=regular_time(g);fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0])
        try:x=phase_samples(t,g[CHANNELS].to_numpy(float),fp,br)
        except ValueError as e:bad.append(dict(pitch=pid,reason=str(e)));continue
        features[pid]=np.r_[x.T.ravel(),(br-fp)*1000]
    frame=frame[frame.session_pitch.isin(features)].copy().reset_index(drop=True)
    x=np.stack([features[p] for p in frame.session_pitch]);y=frame.speed_kmh.to_numpy(float)
    groups=frame[['user','session']]
    cx=pd.DataFrame(x)-pd.DataFrame(x).groupby([groups.user,groups.session]).transform('mean')
    cy=y-frame.groupby(['user','session']).speed_kmh.transform('mean').to_numpy()
    models={'trunk_waveform':list(range(42))+[231],
            'arm_wrist_waveform':list(range(42,231))+[231],
            'full_waveform':list(range(232))}
    predictions=[]
    for user in sorted(frame.user.unique()):
        train=frame.user.to_numpy()!=user
        coefficients={name:ridge_fit(cx.to_numpy()[train][:,ix],cy[train],alpha=100.) for name,ix in models.items()}
        for session,g in frame[frame.user==user].groupby('session'):
            ids=g.sort_values('session_pitch').index.to_numpy();anchor=ids[0]
            for i in ids[1:]:
                actual=float(y[i]-y[anchor])
                for name,ix in models.items():
                    delta=float((x[i,ix]-x[anchor,ix])@coefficients[name])
                    predictions.append(dict(user=int(user),session=int(session),pitch=frame.iloc[i].session_pitch,model=name,
                        actual_delta_kmh=actual,predicted_delta_kmh=delta,error=delta-actual,baseline_error=-actual))
    p=pd.DataFrame(predictions);summary=[]
    for name,g in p.groupby('model'):
        by=g.assign(a=g.error.abs(),b=g.baseline_error.abs()).groupby('user')[['a','b']].mean()
        summary.append(dict(model=name,n_pitches=len(g),athletes=len(by),mae_kmh=float(g.error.abs().mean()),
            baseline_mae_kmh=float(g.baseline_error.abs().mean()),bias_kmh=float(g.error.mean()),
            athlete_macro_mae_kmh=float(by.a.mean()),baseline_macro_mae_kmh=float(by.b.mean()),
            athletes_improved=int((by.a<by.b).sum())))
    result=dict(counts=counts,additional_exclusions=bad,channels=CHANNELS,phase_points=21,ridge_alpha=100,
                summaries=summary,independent_external_validation=False,causal_effect=False,
                scope='Same reused OBP cohort; excluded athlete, measured anchor speed. Joint angular velocity channels are coordinate conventions, not muscle forces. Peak-vs-waveform comparison is a new retrospective model comparison.')
    p.to_csv(OUT/'waveform_predictions.csv',index=False)
    (OUT/'waveform_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));return result


if __name__=='__main__':run()
