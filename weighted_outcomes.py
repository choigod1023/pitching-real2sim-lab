"""Descriptive athlete-balanced mass-condition contrasts; no causal claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=20261025


def paired_interval(values,draws=10000):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():raise ValueError('Finite paired athlete values required')
    rng=np.random.default_rng(SEED)
    means=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def analyze(raw):
    data=raw.copy()
    if data[['subject','ball_weight']].isna().any().any():raise ValueError('Missing group identifier')
    weights=data.ball_weight.astype(str).str.extract(r'^([3-7])oz$',expand=False)
    if weights.isna().any():raise ValueError('Unexpected mass label')
    data['weight_oz']=weights.astype(int)
    data['pitch_speed_kmh']=data.pitch_velocity*1.609344
    metrics=[c for c in raw.columns if c not in ('subject','ball_weight','pitch_velocity')]+['pitch_speed_kmh']
    if not np.isfinite(data[metrics].to_numpy(dtype=float)).all():raise ValueError('Missing or nonfinite outcome')
    counts=data.groupby(['subject','weight_oz']).size().rename('repeats')
    means=data.groupby(['subject','weight_oz'])[metrics].mean()
    condition=[];contrasts=[];consistency=[]
    for weight in range(3,8):
        g=data[data.weight_oz.eq(weight)];athlete=means.xs(weight,level='weight_oz')
        condition.append({'weight_oz':weight,'pitches':len(g),'athletes':len(athlete),
                          'athlete_balanced_means':{k:float(v) for k,v in athlete.mean().items()},
                          'pitch_weighted_speed_kmh':float(g.pitch_speed_kmh.mean())})
    for metric in metrics:
        wide=means[metric].unstack('weight_oz')
        for weight in (3,4,6,7):
            pair=wide[[5,weight]].dropna();delta=(pair[weight]-pair[5]).to_numpy()
            contrasts.append({'metric':metric,'weight_oz':weight,'reference_oz':5,'paired_athletes':len(pair),
                              'mean_difference':float(delta.mean()),'median_difference':float(np.median(delta)),
                              'sd_difference':float(np.std(delta,ddof=1)),
                              'positive_athletes':int(np.sum(delta>1e-10)),'negative_athletes':int(np.sum(delta< -1e-10)),
                              'bootstrap_percentile_95':paired_interval(delta)})
        grouped=data.groupby(['subject','weight_oz'])[metric]
        ranges=grouped.max()-grouped.min();repeat_groups=counts[counts>1].index
        consistency.append({'metric':metric,'repeat_groups':len(repeat_groups),
                            'exact_constant_groups':int((ranges.loc[repeat_groups]==0).sum())})
    return {'rows':len(data),'athletes':int(data.subject.nunique()),'condition':condition,'contrasts':contrasts,
            'repeat_counts':[{'subject':int(s),'weight_oz':int(w),'repeats':int(n)} for (s,w),n in counts.items()],
            'within_group_consistency':consistency,'seed':SEED,'bootstrap_draws':10000,
            'interpretation':'Paired subject-condition means. Pointwise exploratory intervals over athletes, not simultaneous intervals, causal effects, physical model predictions, safety evidence, or individual coaching gains. Fixed condition order confounds load with fatigue/order. No trial identity matching used.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('../weighted-manifest/artifacts/compiled/data_table.csv'))
    p.add_argument('--out',type=Path,default=Path('artifacts/weighted-outcomes'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    result=analyze(pd.read_csv(a.data));result['input_sha256']=hashlib.sha256(a.data.read_bytes()).hexdigest()
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'rows':result['rows'],'athletes':result['athletes'],'speed_contrasts':[r for r in result['contrasts'] if r['metric']=='pitch_speed_kmh']},indent=2))
