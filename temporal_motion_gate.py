import json,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd
from paired_motion_baseline import FEATURES,outer_fold,balanced

ROOT=Path(__file__).resolve().parent
JOINTS=['pelvis','torso','shoulder','elbow','wrist','lead_knee']
TEMP=[f'{j}_{k}' for j in JOINTS for k in ['p0','p25','p50','p75','p100','peak_phase']]+['duration_ms']

def extract(g):
    g=g.sort_values('time');t=g.time.to_numpy();fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0])
    if not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('nonmonotonic time')
    if not np.isfinite([fp,br]).all() or not t[0]<=fp<br<=t[-1]:raise ValueError('invalid events')
    if g.fp_10_time.nunique()!=1 or g.BR_time.nunique()!=1:raise ValueError('nonconstant events')
    window=(t>=fp)&(t<=br)
    if window.sum()<3:raise ValueError('short window')
    # Keep bracketing samples only, never fill a missing measured signal.
    lo=max(np.searchsorted(t,fp)-1,0);hi=min(np.searchsorted(t,br)+1,len(t)-1)
    ts=t[lo:hi+1];query=np.linspace(fp,br,5);out=[]
    for joint in JOINTS:
        xyz=g[[joint+'_velo_'+c for c in 'xyz']].to_numpy()[lo:hi+1]
        if not np.isfinite(xyz).all():raise ValueError('nonfinite '+joint)
        speed=np.linalg.norm(xyz,axis=1)
        sampled=np.array([np.interp(query,ts,xyz[:,k]) for k in range(3)]).T
        tw=(ts>=fp)&(ts<=br);peak=ts[tw][np.argmax(speed[tw])]
        out.extend([*np.linalg.norm(sampled,axis=1),(peak-fp)/(br-fp)])
    return np.r_[out,(br-fp)*1000]

def run():
    data=ROOT.parent/'openbiomechanics/baseball_pitching/data';out=ROOT/'artifacts/temporal-motion';out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(data/'metadata.csv');poi=pd.read_csv(data/'poi/poi_metrics.csv')
    joined=meta[['user','session','session_pitch']].merge(poi,on=['session','session_pitch'],validate='one_to_one')
    cols=['session_pitch','time','fp_10_time','BR_time']+[j+'_velo_'+c for j in JOINTS for c in 'xyz']
    signals=pd.read_csv(data/'full_sig/joint_velos.csv',usecols=cols);groups=dict(tuple(signals.groupby('session_pitch')))
    records=[];excluded=[]
    for r in joined.to_dict('records'):
        try:
            if r['pitch_type']!='FF':raise ValueError('not FF')
            if not np.isfinite([r['pitch_speed_mph'],*[r[f] for f in FEATURES]]).all():raise ValueError('nonfinite POI')
            r.update(dict(zip(TEMP,extract(groups[r['session_pitch']]))));records.append(r)
        except (ValueError,KeyError) as e:excluded.append(dict(pitch=r['session_pitch'],reason=str(e)))
    valid=pd.DataFrame(records);rows=[]
    for _,g in valid.groupby(['user','session','pitch_type']):
        g=g.sort_values('session_pitch');ref=g.iloc[0]
        for _,r in g.iloc[1:].iterrows():
            row=dict(user=int(r.user),reference=ref.session_pitch,pitch=r.session_pitch,actual=(r.pitch_speed_mph-ref.pitch_speed_mph)*1.609344)
            row.update({f:float(r[f]-ref[f]) for f in FEATURES+TEMP});rows.append(row)
    pairs=pd.DataFrame(rows);y=pairs.actual.to_numpy();u=pairs.user.to_numpy();folds=[]
    for name,features in [('poi',FEATURES),('temporal',TEMP)]:
        x=pairs[features].to_numpy();pairs[name]=np.nan
        for user in np.unique(u):
            pred,fit=outer_fold(x,y,u,user);pairs.loc[u==user,name]=pred;pairs.loc[u==user,'mean']=fit['training_mean'];folds.append(dict(model=name,user=int(user),**fit))
    pairs['zero']=0.;summaries={};large=abs(y)>=1
    for method in ['poi','temporal','mean','zero']:
        summaries[method]=dict(mae=balanced(abs(pairs[method]-y),u),rmse=float(np.sqrt(balanced((pairs[method]-y)**2,u))),large_sign=balanced((np.sign(pairs.loc[large,method])==np.sign(y[large])).astype(float),u[large]))
    per=pairs.assign(error=abs(pairs.temporal-y),zeroerror=abs(y)).groupby('user')[['error','zeroerror']].mean();better=int((per.error<per.zeroerror).sum())
    checks=dict(below_one=summaries['temporal']['mae']<1,beats_controls=all(summaries['temporal']['mae']<summaries[c]['mae'] for c in ['zero','mean','poi']),majority_60=better/len(per)>=.6,direction=summaries['temporal']['large_sign']>summaries['mean']['large_sign'])
    gate=dict(prediction_checks=checks,prediction_pass=all(checks.values()),anatomical_mapping_validated=False,individual_capacity_validated=False,optimization_executed=False)
    report=dict(valid_pitches=len(valid),excluded=excluded,athletes=int(pairs.user.nunique()),pairs=len(pairs),large_pairs=int(large.sum()),summary=summaries,better_athletes=better,gate=gate,folds=folds)
    paths=[data/'metadata.csv',data/'poi/poi_metrics.csv',data/'full_sig/joint_velos.csv',ROOT/'TEMPORAL_PROTOCOL.md'];report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False));(out/'gate.json').write_text(json.dumps(gate,indent=2));pairs.to_csv(out/'predictions.csv',index=False)
    # Preserve literal active/commented source fields, not inferred defaults.
    path=data.parent/'code/v3d/CMO.v3s';s=path.read_text();blocks=[b for b in s.split(';') if re.search(r'/RESULT_NAME=(RT|LT)_(SHOULDER|ELBOW|WRIST)_ANGLE\b',b)]
    audit=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),blocks=blocks,interpretation='Commented ! fields are not verified executable settings; no anatomical remapping adopted.')
    a=pd.read_csv(data/'full_sig/joint_angles.csv');audit['angle_columns']={c:dict(finite=int(np.isfinite(a[c]).sum()),nonzero=int((a[c].fillna(0).abs()>1e-9).sum()),minimum=float(a[c].min()),maximum=float(a[c].max())) for c in a if c.startswith(('shoulder_angle','elbow_angle','wrist_angle'))}
    (out/'coordinate_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('folds','source_sha256')}))

if __name__=='__main__':run()
