"""Exploratory kinematic counterfactuals. No ball-speed or causal claims."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sport_outcomes import attach_observed_speed, evidence_contract

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'openbiomechanics/baseball_pitching/data'
OUT=ROOT/'artifacts/pilot'
# Fixed before evaluation. These are engineering screens, not safety limits.
GRID=[(a,d) for a in [-5.,-2.5,0.,2.5,5.] for d in [-10.,-5.,0.,5.,10.]]


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def rotate(v,axis,angle):
    c=np.cos(angle)[:,None];s=np.sin(angle)[:,None]
    return v*c+np.cross(axis,v)*s+axis*np.sum(axis*v,axis=1,keepdims=True)*(1-c)


def vector(df,name):return df[[name+'_'+c for c in 'xyz']].to_numpy(float)


def regular_time(df):
    recorded=df.time.to_numpy(float)
    nominal=recorded[0]+np.arange(len(recorded))/360.
    if np.max(np.abs(recorded-nominal))>.000051:
        raise ValueError('Time samples do not match documented 360Hz within CSV rounding')
    return nominal


def change(t,shoulder,elbow,wrist,hand,release,angle_deg,delay_ms):
    # Smooth local envelope; positive delay reads earlier forearm orientation.
    weight=np.exp(-.5*((t-release)/.1)**2)
    shift=t-delay_ms*.001*weight
    # Shift directions relative to elbow, not whole-body translation.
    def retime(v):
        sampled=np.column_stack([np.interp(shift,t,v[:,k]) for k in range(3)])
        return unit(sampled)*np.linalg.norm(v,axis=1,keepdims=True)
    fore=wrist-elbow; hv=hand-elbow
    shifted=retime(fore)
    # Use a common minimal rotation for wrist and hand to keep their geometry.
    u=unit(fore);v=unit(shifted)
    cross=np.cross(u,v); axis=unit(cross)
    ang=np.arctan2(np.linalg.norm(cross,axis=1),np.sum(u*v,axis=1))
    moved_hand=rotate(hv,axis,ang)
    moved_fore=shifted
    # Positive bend increases angle between upper-arm-forward and forearm.
    bend_axis=unit(np.cross(unit(elbow-shoulder),unit(moved_fore)))
    bend=np.deg2rad(angle_deg)*weight
    return elbow+rotate(moved_fore,bend_axis,bend),elbow+rotate(moved_hand,bend_axis,bend)


def speed_at(t,p,event):
    v=np.gradient(p,t,axis=0,edge_order=2)
    return float(np.linalg.norm([np.interp(event,t,v[:,k]) for k in range(3)]))


def evaluate(df,angle,delay):
    t=regular_time(df); release=float(df.BR_time.iloc[0])
    shoulder=vector(df,'shoulder_jc');elbow=vector(df,'elbow_jc')
    wrist=vector(df,'wrist_jc');hand=vector(df,'hand_jc')
    cw,ch=change(t,shoulder,elbow,wrist,hand,release,angle,delay)
    roi=(t>=release-.2)&(t<=release+.05)
    deviation=np.rad2deg(np.arccos(np.clip(np.sum(unit(cw-elbow)*unit(wrist-elbow),axis=1),-1,1)))
    acc=lambda p:np.gradient(np.gradient(p,t,axis=0,edge_order=2),t,axis=0,edge_order=2)
    base_acc=np.linalg.norm(acc(hand),axis=1)[roi].max()
    ratio=float(np.linalg.norm(acc(ch),axis=1)[roi].max()/base_acc)
    displacement=float(np.linalg.norm(ch-hand,axis=1)[roi].max())
    delta=speed_at(t,ch,release)-speed_at(t,hand,release)
    step=float(np.median(np.diff(t)))
    offsets=[speed_at(t,ch,release+q*step)-speed_at(t,hand,release+q*step) for q in [-1,0,1]]
    return {'angle_deg':angle,'delay_ms':delay,'baseline_hand_speed_m_s':speed_at(t,hand,release),
            'delta_hand_speed_m_s':delta,'min_delta_release_pm1frame':min(offsets),
            'max_forearm_direction_change_deg':float(deviation[roi].max()),
            'max_hand_displacement_m':displacement,'peak_hand_acceleration_ratio':ratio,
            'eligible':bool(deviation[roi].max()<=5.+1e-7 and displacement<=.05+1e-9 and ratio<=1.1+1e-9)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(DATA/'metadata.csv').sort_values(['user','session_pitch'])
    data=pd.read_csv(DATA/'full_sig/landmarks.csv')
    groups={k:g.sort_values('time') for k,g in data.groupby('session_pitch',sort=False)}
    rows=[];skipped=[]
    for idx,m in enumerate(meta.itertuples()):
        df=groups.get(m.session_pitch)
        if df is None:skipped.append([m.session_pitch,'no landmarks']);continue
        cols=['time','BR_time']+[n+'_'+c for n in ['shoulder_jc','elbow_jc','wrist_jc','hand_jc'] for c in 'xyz']
        if not np.isfinite(df[cols].to_numpy(float)).all() or np.any(np.diff(df.time)<=0):
            skipped.append([m.session_pitch,'nonfinite or time order']);continue
        if not df.time.min()+.22<float(df.BR_time.iloc[0])<df.time.max()-.08:
            skipped.append([m.session_pitch,'release near edge']);continue
        for a,d in GRID:
            row=evaluate(df,a,d);row.update(user=int(m.user),session_pitch=m.session_pitch)
            rows.append(row)
        if idx%50==0:print(f'Evaluated {idx+1}/{len(meta)} pitches',flush=True)
    candidates=attach_observed_speed(pd.DataFrame(rows),meta)
    first=candidates[['user','session_pitch']].drop_duplicates().groupby('user',sort=True).first().session_pitch
    candidates['split']=np.where(candidates.session_pitch.isin(first.values),'selection','heldout')
    train=candidates[candidates.split=='selection']
    # Global setting learned only from selection pitches, not evaluation pitches.
    common=train.groupby(['angle_deg','delay_ms']).min_delta_release_pm1frame.mean().idxmax()
    common_stats=train.groupby(['angle_deg','delay_ms']).agg(score=('min_delta_release_pm1frame','mean'),pass_rate=('eligible','mean'))
    common_screened=common_stats[common_stats.pass_rate>=.95].score.idxmax()
    choices={}
    for uid,g in train.groupby('user'):
        g=g.copy();g['cost']=g.angle_deg.abs()/5+g.delay_ms.abs()/10
        constrained=g[g.eligible & (g.min_delta_release_pm1frame>0.05)]
        # Abstain unless at least .05 m/s proxy increase at all three timing samples.
        best=constrained.sort_values(['min_delta_release_pm1frame','cost'],ascending=[False,True]).iloc[0] if len(constrained) else g[(g.angle_deg==0)&(g.delay_ms==0)].iloc[0]
        free=g.sort_values(['min_delta_release_pm1frame','cost'],ascending=[False,True]).iloc[0]
        choices[int(uid)]={'personalized':(float(best.angle_deg),float(best.delay_ms)),
                           'unconstrained':(float(free.angle_deg),float(free.delay_ms))}
    evaluations=[]
    for (uid,pitch),g in candidates[candidates.split=='heldout'].groupby(['user','session_pitch']):
        for method,setting in {'unchanged':(0.,0.),'common':common,'common_screened':common_screened,**choices[int(uid)]}.items():
            row=g[(g.angle_deg==setting[0])&(g.delay_ms==setting[1])].iloc[0].to_dict()
            row['method']=method;evaluations.append(row)
    results=pd.DataFrame(evaluations)
    summary={}
    for method,g in results.groupby('method'):
        # Athlete-level average prevents those with more pitches dominating.
        per=g.groupby('user').delta_hand_speed_m_s.mean()
        rng=np.random.default_rng(20261025)
        means=np.array([rng.choice(per.to_numpy(),len(per),replace=True).mean() for _ in range(2000)])
        summary[method]={'athletes':len(per),'pitches':len(g),'athlete_mean_proxy_delta_m_s':float(per.mean()),
                         'bootstrap95_mean':[float(x) for x in np.quantile(means,[.025,.975])],
                         'athletes_positive_fraction':float((per>.05).mean()),
                         'heldout_eligible_fraction':float(g.eligible.mean()),
                         'heldout_timing_robust_positive_fraction':float((g.min_delta_release_pm1frame>.05).mean())}
    candidates.to_csv(OUT/'candidates.csv',index=False);results.to_csv(OUT/'heldout.csv',index=False)
    report={'evidence':evidence_contract(),
            'study_type':'offline kinematic counterfactual feasibility pilot, not causal or dynamic',
            'screen_limits':{'forearm_direction_deg':5,'hand_displacement_m':.05,'peak_hand_acceleration_ratio':1.1},
            'candidates_per_pitch':len(GRID),'evaluated_pitches':candidates.session_pitch.nunique(),
            'selection_athletes':len(choices),'skipped':skipped,'common_setting':list(common),'common_screened_setting':list(common_screened),
            'protocol_note':'Initial +/-250ms edge exclusion retained only 126 pitches. Required coverage aligned to -200/+50ms metric window plus margins. Subsequently rounded CSV timestamps were replaced by verified nominal 360Hz times before derivative calculations. Both revisions are documented; this is exploratory, not preregistered.',
            'abstention_athletes':sum(v['personalized']==(0.,0.) for v in choices.values()),
            'summary':summary,'choices':choices}
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='choices'},indent=2),flush=True)


if __name__=='__main__':main()
