"""Exploratory held-athlete-out scalar correction, nominal BR only.

Fixed ridge penalty 1 on a slope without intercept, athlete-balanced training
loss. No timing, filtering, template, or penalty selection by held-out targets.
This cannot validate intervention effects or repair missing release physics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def paired(data):
    rows=[]
    for user,g in data.groupby('user'):
        g=g.sort_values('pitch')
        for j in range(1,len(g)):
            rows.append({'user':user,'pitch':g.iloc[j].pitch,
                         'x':g.iloc[j].predicted_kmh-g.iloc[0].predicted_kmh,
                         'y':g.iloc[j].observed_kmh-g.iloc[0].observed_kmh})
    return pd.DataFrame(rows)


def slope(training):
    weight=1/training.groupby('user').user.transform('count').to_numpy()
    x,y=training.x.to_numpy(),training.y.to_numpy()
    return float(np.sum(weight*x*y)/(np.sum(weight*x*x)+1.))


def evaluate(data):
    results=[]
    for method,g in data[data.offset_frames.eq(0)].groupby('method'):
        pairs=paired(g);folds=[]
        for user,test in pairs.groupby('user'):
            train=pairs[pairs.user.ne(user)]
            if train.user.nunique()<2:raise ValueError('Insufficient training athletes')
            beta=slope(train)
            folds.append({'user':int(user),'slope':beta,'pairs':len(test),
                          'mae_kmh':float(np.mean(abs(beta*test.x-test.y))),
                          'no_change_mae_kmh':float(np.mean(abs(test.y)))})
        results.append({'method':method,'folds':folds,
                        'mae_kmh':float(np.mean([f['mae_kmh'] for f in folds])),
                        'no_change_mae_kmh':float(np.mean([f['no_change_mae_kmh'] for f in folds]))})
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result=evaluate(pd.read_csv(a.directory/'predictions.csv'))
    (a.directory/'calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='folds'} for r in result],indent=2))
