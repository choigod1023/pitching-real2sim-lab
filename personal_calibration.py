"""Two observed calibration pitches; athlete-disjoint model selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paired_motion_baseline import FEATURES


def fit_population(groups, alpha):
    xs=[];ys=[];ws=[]
    for x,y in groups:
        xs.append(x-x.mean(axis=0));ys.append(y-y.mean());ws.extend([1/len(y)]*len(y))
    x=np.vstack(xs);y=np.concatenate(ys);w=np.asarray(ws)
    scale=np.sqrt(np.average(x*x,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
    z=x/scale
    beta=np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(x.shape[1]),z.T@(w*y))
    return scale,beta


def predict(model, calibration_x, calibration_y, evaluation_x, strength=None):
    scale,beta=model;cx=np.asarray(calibration_x);cy=np.asarray(calibration_y)
    center=cx.mean(axis=0);z=(cx-center)/scale;target=cy-cy.mean()
    b=beta.copy()
    if strength is not None:
        if strength<=0:raise ValueError('positive shrinkage required')
        b+=np.linalg.solve(z.T@z+strength*np.eye(len(b)),z.T@(target-z@b))
    return cy.mean()+(np.asarray(evaluation_x)-center)/scale@b


def choose(groups):
    keys=sorted(groups);scores=[]
    for alpha in (1.,10.,100.):
        errors={s:[] for s in (None,1.,10.,100.)}
        for k in range(5):
            held=set(keys[k::5]);model=fit_population([groups[u] for u in keys if u not in held],alpha)
            for u in held:
                x,y=groups[u]
                if len(y)<3:continue
                for strength in errors:
                    p=predict(model,x[:2],y[:2],x[2:],strength)
                    errors[strength].append(float(np.mean(abs(p-y[2:]))))
        scores.extend(dict(alpha=alpha,strength=s,mae=float(np.mean(e))) for s,e in errors.items())
    global_choice=min((r for r in scores if r['strength'] is None),key=lambda r:r['mae'])
    personal_choice=min((r for r in scores if r['strength'] is not None),key=lambda r:r['mae'])
    return global_choice,personal_choice


def run():
    root=Path(__file__).resolve().parent
    source=root.parent/'temporal-motion-gate/artifacts/temporal-motion/predictions.csv'
    frame=pd.read_csv(source);groups={};ids={}
    for u,g in frame.groupby('user'):
        g=g.sort_values('pitch');x=np.vstack([np.zeros(len(FEATURES)),g[FEATURES].to_numpy()]);y=np.r_[0.,g.actual.to_numpy()]
        if g.reference.nunique()!=1 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid source')
        groups[int(u)]=(x,y);ids[int(u)]=[g.reference.iloc[0]]+g.pitch.tolist()
    rows=[];folds=[];excluded=[]
    for u,(x,y) in groups.items():
        if len(y)<3:excluded.append(u);continue
        training={k:v for k,v in groups.items() if k!=u};gc,pc=choose(training)
        predictions={}
        for name,choice in [('global',gc),('personal',pc)]:
            model=fit_population(list(training.values()),choice['alpha'])
            predictions[name]=predict(model,x[:2],y[:2],x[2:],choice['strength'])
        for j in range(2,len(y)):
            rows.append(dict(user=u,pitch=ids[u][j],calibration=ids[u][:2],actual=float(y[j]),mean=float(y[:2].mean()),last=float(y[1]),global_prediction=float(predictions['global'][j-2]),personal=float(predictions['personal'][j-2])))
        folds.append(dict(user=u,global_choice=gc,personal_choice=pc))
    df=pd.DataFrame(rows);methods=['mean','last','global_prediction','personal'];per={};summary={}
    for method in methods:
        errors=df[method]-df.actual;per[method]=abs(errors).groupby(df.user).mean()
        summary[method]=dict(mae=float(per[method].mean()),rmse=float(np.sqrt((errors**2).groupby(df.user).mean().mean())))
    delta=(per['personal']-per['mean']).to_numpy();rng=np.random.default_rng(20261027)
    ci=np.quantile(rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1),[.025,.975]).tolist()
    report=dict(athletes=len(delta),evaluation_pitches=len(df),excluded_two_pitch_athletes=excluded,summary=summary,
        better_than_mean=int(sum(delta<0)),difference_ci95=ci,folds=folds,predictions=rows,
        scope='Previously inspected development data; two labelled calibration pitches required; not intervention or physical simulator accuracy.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((root/'PERSONAL_PROTOCOL.md').read_bytes()).hexdigest())
    report['candidate_retained']=summary['personal']['mae']<1 and ci[1]<0 and all(summary['personal']['mae']<summary[m]['mae'] for m in methods[:-1])
    (root/'personal_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['folds','predictions']}))
    return report


if __name__=='__main__':run()
