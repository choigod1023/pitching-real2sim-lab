"""Nested athlete-held-out statistical correction; leaves all physics unchanged."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES=['retargeted_wrist_kmh','rotational_component_norm_kmh','rfin_proxy_kmh']
ALPHAS=[np.inf,100.,10.,1.,.1,.01]  # strongest wins exact ties


def weights(users):
    _,inv,counts=np.unique(users,return_inverse=True,return_counts=True)
    w=1/counts[inv];return w/w.sum()


def balanced(values,users):
    return float(np.dot(weights(users),values))


def predict(x,y,users,test,alpha,intercept):
    w=weights(users)
    xm=np.sum(w[:,None]*x,axis=0) if intercept else np.zeros(x.shape[1])
    ym=float(w@y) if intercept else 0.
    scale=np.sqrt(np.sum(w[:,None]*(x-xm)**2,axis=0))
    scale=np.where(scale>1e-10,scale,1.)
    z=(x-xm)/scale
    beta=np.zeros(x.shape[1]) if np.isinf(alpha) else np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(x.shape[1]),z.T@(w*(y-ym)))
    return ym+(test-xm)/scale@beta


def nested_prediction(x,y,users,test,intercept):
    scores=[]
    for alpha in ALPHAS:
        errors=[];ids=[]
        for user in np.unique(users):
            train=users!=user;hold=~train
            pred=predict(x[train],y[train],users[train],x[hold],alpha,intercept)
            errors.extend(abs(pred-y[hold]));ids.extend(users[hold])
        scores.append(balanced(np.asarray(errors),np.asarray(ids)))
    best=int(np.argmin(scores));alpha=ALPHAS[best]
    return predict(x,y,users,test,alpha,intercept),('infinity' if np.isinf(alpha) else alpha),scores


def evaluate(group):
    group=group.sort_values(['user','pitch']).reset_index(drop=True)
    x=group[FEATURES].to_numpy();y=group.observed_kmh.to_numpy();users=group.user.to_numpy()
    raw=group.integrated_ball_kmh.to_numpy()
    dx=[];dy=[];du=[];dp=[];draw=[]
    for user,g in group.groupby('user'):
        idx=g.index.to_numpy()
        for i in idx[1:]:
            dx.append(x[i]-x[idx[0]]);dy.append(y[i]-y[idx[0]])
            du.append(user);dp.append(group.pitch.iloc[i]);draw.append(raw[i]-raw[idx[0]])
    dx,dy,du,draw=map(np.asarray,(dx,dy,du,draw))
    rows=[];folds=[]
    for user in np.unique(users):
        tr=users!=user;te=~tr
        preds,alpha,scores=nested_prediction(x[tr],y[tr],users[tr],x[te],True)
        mean=balanced(y[tr],users[tr]);bias=balanced(y[tr]-raw[tr],users[tr])
        methods={'raw_physics':raw[te],'train_mean':np.full(te.sum(),mean),
                 'legacy_pitch_weighted_mean':np.full(te.sum(),float(y[tr].mean())),
                 'train_bias':raw[te]+bias,'nested_components':preds}
        for method,values in methods.items():
            for pitch,a,p in zip(group.pitch[te],y[te],values):
                rows.append(dict(user=int(user),pitch=pitch,task='absolute',method=method,observed=float(a),predicted=float(p),error=abs(float(a-p))))
        trd=du!=user;ted=~trd
        pred_delta,da,ds=nested_prediction(dx[trd],dy[trd],du[trd],dx[ted],False)
        for method,values in {'raw_physics':draw[ted],'no_change':np.zeros(ted.sum()),'nested_components':pred_delta}.items():
            for pitch,a,p in zip(np.array(dp)[ted],dy[ted],values):
                rows.append(dict(user=int(user),pitch=pitch,task='change',method=method,observed=float(a),predicted=float(p),error=abs(float(a-p))))
        folds.append(dict(user=int(user),absolute_alpha=alpha,change_alpha=da,train_bias_kmh=bias,absolute_inner_scores=scores,change_inner_scores=ds))
    return pd.DataFrame(rows),folds


def run():
    source=Path(__file__).resolve().parent.parent/'speed-path/artifacts/speed-path/decomposition.csv'
    data=pd.read_csv(source)
    assert not data.duplicated(['user','pitch','cutoff_hz']).any()
    assert data.groupby(['user','pitch']).observed_kmh.nunique().eq(1).all()
    assert np.isfinite(data[FEATURES+['observed_kmh','integrated_ball_kmh']]).all().all()
    assert set(data.cutoff_hz)=={0.,10.,20.,30.}
    expected=set(map(tuple,data[data.cutoff_hz.eq(0)][['user','pitch']].to_numpy()))
    all_rows=[];folds=[];summary=[]
    for cutoff,g in data.groupby('cutoff_hz'):
        assert set(map(tuple,g[['user','pitch']].to_numpy()))==expected
        rows,f=evaluate(g);rows['cutoff_hz']=cutoff;all_rows.append(rows)
        folds.extend([dict(cutoff_hz=cutoff,**r) for r in f])
        for (task,method),r in rows.groupby(['task','method']):
            summary.append(dict(cutoff_hz=cutoff,task=task,method=method,athlete_balanced_mae_kmh=balanced(r.error.to_numpy(),r.user.to_numpy()),rows=len(r)))
    out=Path(__file__).resolve().parent/'artifacts/speed-correction';out.mkdir(parents=True,exist_ok=True)
    pd.concat(all_rows).to_csv(out/'predictions.csv',index=False)
    report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),features=FEATURES,summary=summary,folds=folds,
                scope='Already inspected development data. Statistical correction, not physics correction or intervention validation.',physics_changed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':run()
