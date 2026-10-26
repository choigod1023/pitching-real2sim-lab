"""Nested athlete-disjoint ridge for observed within-session velocity changes."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
FEATURES=['max_shoulder_external_rotation','elbow_flexion_fp','rotation_hip_shoulder_separation_fp','stride_length','lead_knee_extension_from_fp_to_br','max_pelvis_rotational_velo','max_torso_rotational_velo','timing_peak_torso_to_peak_pelvis_rot_velo']
ALPHAS=[.1,1.,10.,100.]

def weights(users):
    _,inv,n=np.unique(users,return_inverse=True,return_counts=True)
    return 1/n[inv]

def train(x,y,users,alpha):
    w=weights(users)
    scale=np.sqrt(np.average(x*x,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
    z=x/scale
    coef=np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(x.shape[1]),z.T@(w*y))
    return coef/scale

def balanced(errors,users):
    return float(np.average(errors,weights=weights(users)))

def outer_fold(x,y,users,heldout):
    mask=users!=heldout
    tx,ty,tu=x[mask],y[mask],users[mask]
    groups=np.unique(tu);scores=[]
    for alpha in ALPHAS:
        e=[];us=[]
        for k in range(5):
            test=np.isin(tu,groups[k::5]);training=~test
            coef=train(tx[training],ty[training],tu[training],alpha)
            e.extend(np.abs(tx[test]@coef-ty[test]));us.extend(tu[test])
        scores.append(balanced(np.array(e),np.array(us)))
    alpha=ALPHAS[int(np.argmin(scores))];coef=train(tx,ty,tu,alpha)
    return x[~mask]@coef,dict(alpha=alpha,inner_mae=scores,coef=coef.tolist(),training_mean=float(np.average(ty,weights=weights(tu))))

def run():
    data=ROOT.parent/'openbiomechanics/baseball_pitching/data'
    paths=[data/'metadata.csv',data/'poi/poi_metrics.csv']
    meta=pd.read_csv(paths[0]);poi=pd.read_csv(paths[1])
    merged=meta[['user','session','session_pitch']].merge(poi,on=['session','session_pitch'],validate='one_to_one')
    rows=[]
    for (user,session,kind),g in merged.groupby(['user','session','pitch_type']):
        if kind!='FF':continue
        g=g.sort_values('session_pitch');ref=g.iloc[0]
        for _,r in g.iloc[1:].iterrows():
            item=dict(user=user,session=session,reference=ref.session_pitch,pitch=r.session_pitch,p_throws=r.p_throws,actual=(r.pitch_speed_mph-ref.pitch_speed_mph)*1.609344)
            item.update({f:float(r[f]-ref[f]) for f in FEATURES});rows.append(item)
    pairs=pd.DataFrame(rows);x=pairs[FEATURES].to_numpy();y=pairs.actual.to_numpy();users=pairs.user.to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Missing data: do not silently exclude')
    pairs['ridge']=np.nan;pairs['training_mean']=np.nan;pairs['no_change']=0.;folds=[]
    for user in np.unique(users):
        pred,details=outer_fold(x,y,users,user)
        pairs.loc[users==user,'ridge']=pred;pairs.loc[users==user,'training_mean']=details['training_mean'];folds.append(dict(heldout=int(user),**details))
    summaries=[]
    for subgroup in ['all','R','L']:
        p=pairs if subgroup=='all' else pairs[pairs.p_throws==subgroup]
        for method in ['ridge','no_change','training_mean']:
            err=p[method]-p.actual;large=p.actual.abs()>=1
            summaries.append(dict(group=subgroup,method=method,athletes=int(p.user.nunique()),pairs=len(p),mae=balanced(np.abs(err),p.user.to_numpy()),rmse=float(np.sqrt(balanced(err**2,p.user.to_numpy()))),
                sign_accuracy=balanced((np.sign(p[method])==np.sign(p.actual)).astype(float),p.user.to_numpy()),large_pairs=int(large.sum()),large_sign_accuracy=balanced((np.sign(p.loc[large,method])==np.sign(p.loc[large,'actual'])).astype(float),p.loc[large,'user'].to_numpy())))
    per=pairs.assign(ridge_error=abs(pairs.ridge-pairs.actual),zero_error=abs(pairs.actual)).groupby('user')[['ridge_error','zero_error']].mean()
    report=dict(scope='Observed within-session development association, not intervention or unseen-dataset validation',features=FEATURES,rows=len(merged),athletes=int(pairs.user.nunique()),pairs=len(pairs),summary=summaries,better_athletes=int((per.ridge_error<per.zero_error).sum()),worse_athletes=int((per.ridge_error>per.zero_error).sum()),folds=folds,sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[ROOT/'PAIRED_MOTION_PROTOCOL.md']})
    out=ROOT/'artifacts/paired-motion';out.mkdir(parents=True,exist_ok=True);pairs.to_csv(out/'predictions.csv',index=False);per.to_csv(out/'athlete_errors.csv')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('folds','sources_sha256')}))

if __name__=='__main__':run()
