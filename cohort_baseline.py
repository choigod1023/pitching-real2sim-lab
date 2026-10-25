"""Deterministic development-cohort evaluation; no outcome-selected settings."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from coupled_payload import run as prepare, model_for, inverse_and_rollout

FILTERS = [0.,10.,20.,30.]


def select_trials(meta, athlete_limit=10, pitches_per_athlete=3):
    right=meta[meta.p_throws.eq('R')].sort_values(['user','session_pitch'])
    counts=right.groupby('user').size()
    users=sorted(counts[counts>=2].index)[:athlete_limit]
    return right[right.user.isin(users)].groupby('user',sort=True).head(pitches_per_athlete).copy()


def heldout_mean(group):
    return group.user.map({u:float(group.loc[group.user.ne(u),'observed_kmh'].mean())
                           for u in group.user.unique()})


def summarize(data):
    rows=[]
    for cutoff,g in data.groupby('cutoff_hz'):
        g=g.sort_values(['user','pitch']).copy()
        if g.user.nunique()<2:
            raise ValueError('At least two athletes required')
        g['absolute_error']=abs(g.predicted_kmh-g.observed_kmh)
        g['mean_error']=abs(heldout_mean(g)-g.observed_kmh)
        changes=[]
        for user,a in g.groupby('user'):
            if len(a)<2: continue
            actual=a.observed_kmh.to_numpy(); pred=a.predicted_kmh.to_numpy()
            for j in range(1,len(a)):
                changes.append({'user':user,'delta_error':abs((pred[j]-pred[0])-(actual[j]-actual[0])),
                                'no_change_error':abs(actual[j]-actual[0])})
        c=pd.DataFrame(changes)
        if c.empty: raise ValueError('Repeated trials required')
        rows.append({'cutoff_hz':float(cutoff),'athletes':int(g.user.nunique()),'pitches':len(g),
                     'absolute_mae_kmh':float(g.groupby('user').absolute_error.mean().mean()),
                     'leave_athlete_out_mean_mae_kmh':float(g.groupby('user').mean_error.mean().mean()),
                     'change_comparisons':len(c),'change_athletes':int(c.user.nunique()),
                     'change_mae_kmh':float(c.groupby('user').delta_error.mean().mean()),
                     'no_change_mae_kmh':float(c.groupby('user').no_change_error.mean().mean())})
    return rows


def run(data,out,athlete_limit=10):
    data,out=Path(data),Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(data/'metadata.csv').merge(
        pd.read_csv(data/'poi/poi_metrics.csv',usecols=['session_pitch','p_throws']),
        on='session_pitch',validate='one_to_one')
    selected=select_trials(meta,athlete_limit)
    if selected.user.nunique()<2: raise ValueError('Insufficient selected athletes')
    # Freeze selection before executing predictions; no replacement after failures.
    selected[['user','session_pitch']].to_csv(out/'selection.csv',index=False)
    records=[];failures=[]
    for row in selected.itertuples():
        for cutoff in FILTERS:
            case=out/'cases'/str(row.session_pitch)/str(int(cutoff))
            try:
                prepare(data,row.session_pitch,case,cutoff_hz=cutoff,steps=(1/5760,))
                with np.load(case/'reference-0.npz') as r:
                    model=model_for(r['lengths'],float(r['dt']))
                    model.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
                    result,_=inverse_and_rollout(model,r['qpos'],float(r['dt']))
                records.append({'user':int(row.user),'pitch':row.session_pitch,'cutoff_hz':cutoff,
                                'predicted_kmh':result['conditional_release_speed_kmh'],
                                'observed_kmh':float(row.pitch_speed_mph)*1.609344,
                                'ball_point_rmse_mm':result['ball_point_rmse_mm']})
            except (ValueError,RuntimeError,IndexError,KeyError) as e:
                failures.append({'user':int(row.user),'pitch':row.session_pitch,'cutoff_hz':cutoff,
                                 'error_type':type(e).__name__,'reason':str(e)})
        pd.DataFrame(records).to_csv(out/'predictions.csv',index=False)
        (out/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    predictions=pd.DataFrame(records)
    if predictions.empty: raise ValueError('All trials failed')
    complete=predictions.groupby('pitch').cutoff_hz.nunique()
    paired=predictions[predictions.pitch.isin(complete[complete==len(FILTERS)].index)]
    report={'selection':'First numeric athlete IDs with >=2 right-handed pitches; first three lexicographic pitch IDs per athlete. No outcome-based selection or replacement.',
            'selected_athletes':int(selected.user.nunique()),'selected_pitches':len(selected),
            'successful_predictions':len(records),'failures':failures,
            'summary_all_available':summarize(predictions),'summary_common_trials':summarize(paired),
            'model':'Fixed 15cm grip, generic hand inertia, RK4, inverse loads recomputed for observed reference. Not equal-capacity motion optimization.',
            'processing':'All raw/10/20/30Hz hand+forearm marker conditions reported. Joint-center processing unchanged.',
            'change_reference':'First lexicographic pitch within athlete; not asserted chronological.',
            'mean_baseline':'Mean of observed speeds from other athletes among available trials in this evaluation subset; pitch-weighted training mean, athlete-balanced scoring.',
            'scope':'Development cohort already inspected in earlier analyses; not independent external or intervention validation.',
            'coaching_ready':False,'predicted_gain_kmh':None}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    p.add_argument('--out',type=Path,default=Path('artifacts/cohort-baseline'))
    p.add_argument('--athletes',type=int,default=10)
    a=p.parse_args();r=run(a.data,a.out,a.athletes)
    print(json.dumps({k:r[k] for k in ['selected_athletes','selected_pitches','successful_predictions','failures','summary_common_trials']},indent=2))
