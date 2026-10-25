"""Frozen 20 Hz expansion; calibration athletes are excluded from evaluation."""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
from cohort_baseline import select_trials
from coupled_payload import run as prepare, model_for, inverse_and_rollout

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'openbiomechanics/baseball_pitching/data'
OUT = ROOT / 'artifacts/expanded-speed'
TRAIN = ROOT.parent / 'cohort-baseline/artifacts/cohort-baseline/predictions.csv'

def scores(frame):
    result = {}
    for col in ['raw_kmh', 'bias_kmh', 'training_mean_kmh']:
        result[col + '_absolute_mae_kmh'] = float((frame[col]-frame.observed_kmh).abs().groupby(frame.user).mean().mean())
    differences=((frame.bias_kmh-frame.observed_kmh).abs()-(frame.training_mean_kmh-frame.observed_kmh).abs()).groupby(frame.user).mean().to_numpy()
    rng=np.random.default_rng(20261025)
    boot=rng.choice(differences,size=(10000,len(differences)),replace=True).mean(axis=1)
    result['bias_minus_mean_mae_athlete_bootstrap_95_percentile_kmh']=np.quantile(boot,[.025,.975]).tolist()
    result['bootstrap_scope']='Conditional on frozen calibration, athlete resampling, 10000 replicates; not intervention uncertainty.'
    changes = []
    for user, group in frame.groupby('user'):
        group = group.sort_values('pitch')
        if len(group) < 2:
            continue
        actual = group.observed_kmh.to_numpy()
        item = {'user': int(user), 'comparisons': len(group)-1,
                'nochange_mae_kmh': float(np.mean(abs(actual[1:]-actual[0])))}
        for col in ['raw_kmh','bias_kmh']:
            pred = group[col].to_numpy()
            item[col + '_change_mae_kmh'] = float(np.mean(abs((pred[1:]-pred[0])-(actual[1:]-actual[0]))))
        changes.append(item)
    if changes:
        cf = pd.DataFrame(changes)
        result.update({col: float(cf[col].mean()) for col in ['nochange_mae_kmh','raw_kmh_change_mae_kmh','bias_kmh_change_mae_kmh']})
    result['per_athlete_changes'] = changes
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(TRAIN)
    training = training[training.cutoff_hz.eq(20.)].copy()
    assert not training.duplicated(['user','pitch']).any()
    train_users = sorted(int(u) for u in training.user.unique())
    frozen_bias = float((training.observed_kmh-training.predicted_kmh).groupby(training.user).mean().mean())
    frozen_mean = float(training.groupby('user').observed_kmh.mean().mean())
    meta = pd.read_csv(DATA/'metadata.csv').merge(pd.read_csv(DATA/'poi/poi_metrics.csv',usecols=['session_pitch','p_throws']),on='session_pitch',validate='one_to_one')
    selected = select_trials(meta,20)
    selected = selected[~selected.user.isin(train_users)].copy()
    assert selected.user.nunique() == 10
    assert not selected.duplicated(['user','session_pitch']).any()
    assert set(selected.user).isdisjoint(train_users)
    selected[['user','session_pitch']].to_csv(OUT/'selection.csv',index=False)
    protocol = {'cutoff_hz':20.,'dt_requested_s':1/5760,'training_users':train_users,
                'frozen_bias_kmh':frozen_bias,'frozen_training_mean_kmh':frozen_mean,
                'selection':'Next numeric ten right-handed users after first ten development users, first three lexicographic trials each.',
                'scope':'Calibration-excluded expansion cohort; other earlier exposure unverified; same dataset, not external validation.',
                'training_source':str(TRAIN),'training_sha256':hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
                'selected_users':[int(u) for u in selected.user.unique()],
                'selected_trials':len(selected),'equal_capacity':False,'coaching_ready':False}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows, failures = [], []
    started = time.monotonic()
    for row in selected.itertuples():
        pitch = row.session_pitch
        try:
            case = OUT/'cases'/pitch/'20'
            prepare(DATA,pitch,case,cutoff_hz=20.,steps=(1/5760,))
            with np.load(case/'reference-0.npz') as ref:
                model = model_for(ref['lengths'],float(ref['dt']))
                model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
                outcome,_ = inverse_and_rollout(model,ref['qpos'],float(ref['dt']))
            raw = outcome['conditional_release_speed_kmh']
            rows.append({'user':int(row.user),'pitch':pitch,'observed_kmh':float(row.pitch_speed_mph)*1.609344,
                         'raw_kmh':raw,'bias_kmh':raw+frozen_bias,'training_mean_kmh':frozen_mean,
                         'trajectory_rmse_mm':outcome['ball_point_rmse_mm']})
        except (ValueError, RuntimeError, IndexError, KeyError) as exc:
            failures.append({'user':int(row.user),'pitch':pitch,'error_type':type(exc).__name__,'reason':str(exc)})
        pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
        (OUT/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
        print(json.dumps({'completed':len(rows)+len(failures),'total':len(selected),'successes':len(rows),'failures':len(failures),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    report = {**protocol,'successful_predictions':len(rows),'failures':failures,
              'scoring':'Athlete-balanced absolute error. Changes from first lexicographic successfully predicted trial, not chronology; failed trials not replaced.',
              'summary':scores(pd.DataFrame(rows)) if rows else None}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2))

if __name__ == '__main__':
    main()
