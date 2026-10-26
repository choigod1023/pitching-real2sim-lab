"""Fixed observational within-session fast/slow comparisons; no causal inference."""
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'openbiomechanics/baseball_pitching/data'
POI=['elbow_flexion_fp','elbow_flexion_mer','max_shoulder_external_rotation',
     'rotation_hip_shoulder_separation_fp','torso_anterior_tilt_br']
PEAKS={'pelvis_peak_ms':'pelvis_velo_z','torso_peak_ms':'torso_velo_z',
       'elbow_peak_ms':'elbow_velo_x','shoulder_peak_ms':'shoulder_velo_z'}
FEATURES=['fp_to_br_ms',*PEAKS,'wrist_peak_ms','torso_minus_pelvis_peak_ms',*POI]

def extract(g):
    g=g.sort_values('time');fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0])
    if not np.isfinite([fp,br]).all() or br<=fp:raise ValueError('invalid events')
    w=g[(g.time>=fp-.000051)&(g.time<=br+.000051)]
    if len(w)<3:raise ValueError('short event window')
    cols=['time',*PEAKS.values(),'wrist_velo_x','wrist_velo_y','wrist_velo_z']
    if not np.isfinite(w[cols].to_numpy(float)).all():raise ValueError('nonfinite signals')
    out={'fp_to_br_ms':1000*(br-fp),'fp_time_s':fp,'br_time_s':br}
    for key,col in PEAKS.items():out[key]=1000*(float(w.loc[w[col].abs().idxmax(),'time'])-br)
    wrist=np.linalg.norm(w[['wrist_velo_x','wrist_velo_y','wrist_velo_z']],axis=1)
    out['wrist_peak_ms']=1000*(float(w.time.iloc[int(np.argmax(wrist))])-br)
    out['torso_minus_pelvis_peak_ms']=out['torso_peak_ms']-out['pelvis_peak_ms']
    return out

def pair(a,b):
    slow,fast=sorted([a,b],key=lambda r:(r['pitch_speed_mph'],r['session_pitch']))
    if slow['pitch_speed_mph']==fast['pitch_speed_mph']:return None
    out={k:slow[k] for k in ['user','session','pitch_type','split']}
    out.update(slow_pitch=slow['session_pitch'],fast_pitch=fast['session_pitch'],
               slow_kmh=slow['pitch_speed_mph']*1.609344,fast_kmh=fast['pitch_speed_mph']*1.609344,
               delta_kmh=(fast['pitch_speed_mph']-slow['pitch_speed_mph'])*1.609344)
    out.update({f:float(fast[f]-slow[f]) for f in FEATURES})
    return out

def comparisons(frame):
    extrema=[];allpairs=[];excluded=[]
    for key,g in frame.groupby(['user','session','pitch_type'],sort=True):
        records=g.sort_values(['pitch_speed_mph','session_pitch']).to_dict('records')
        if len(records)<2:excluded.append({'group':list(key),'reason':'single pitch'});continue
        item=pair(records[0],records[-1])
        if item is None:excluded.append({'group':list(key),'reason':'identical recorded speeds'});continue
        extrema.append(item)
        allpairs.extend(p for a,b in combinations(records,2) if (p:=pair(a,b)) is not None)
    return pd.DataFrame(extrema),pd.DataFrame(allpairs),excluded

def interval(values):
    x=np.asarray(values,float)
    if not len(x):return [None,None]
    rng=np.random.default_rng(20261026)
    means=x[rng.integers(0,len(x),(2000,len(x)))].mean(axis=1)
    return np.percentile(means,[2.5,97.5]).tolist()

def summarize(extrema,allpairs):
    results=[]
    for feature in FEATURES:
        discovery=extrema[extrema.split=='discovery'].groupby('user')[feature].mean()
        mean=float(discovery.mean());direction=int(np.sign(mean));sd=float(discovery.std(ddof=1))
        row={'feature':feature,'discovery_direction':direction,
             'discovery_standardized_difference':abs(mean)/sd if sd>0 else 0.}
        for split in ['discovery','evaluation']:
            x=extrema[extrema.split==split].groupby('user')[feature].mean()
            p=allpairs[allpairs.split==split].copy()
            agreement=p.groupby('user')[feature].apply(lambda v:float((v*direction>1e-8).mean()))
            nonzero=x[np.abs(x)>1e-8]
            row[split]={'athletes':len(x),'mean_fast_minus_slow':float(x.mean()),
                'median_fast_minus_slow':float(x.median()),'descriptive_bootstrap_mean_95':interval(x),
                'same_discovery_direction_fraction_all':float((x*direction>1e-8).mean()),
                'same_direction_fraction_nonzero':float((nonzero*direction>0).mean()) if len(nonzero) else None,
                'zero_difference_athletes':int((np.abs(x)<=1e-8).sum()),
                'all_pairs_athlete_balanced_direction_fraction':float(agreement.mean())}
        results.append(row)
    return sorted(results,key=lambda r:(-r['discovery_standardized_difference'],r['feature']))

def main():
    out=ROOT/'artifacts/within-athlete';out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(DATA/'metadata.csv');poi=pd.read_csv(DATA/'poi/poi_metrics.csv')
    if meta.session_pitch.duplicated().any() or poi.session_pitch.duplicated().any():raise ValueError('duplicate ID')
    joined=meta.merge(poi[['session_pitch','session','pitch_type',*POI]],on=['session_pitch','session'],validate='one_to_one',how='left')
    joined['split']=np.where(joined.user.astype(int)%2==0,'discovery','evaluation')
    cols=['session_pitch','time','fp_10_time','BR_time',*PEAKS.values(),'wrist_velo_x','wrist_velo_y','wrist_velo_z']
    signals=pd.read_csv(DATA/'full_sig/joint_velos.csv',usecols=cols)
    groups={k:g for k,g in signals.groupby('session_pitch',sort=False)}
    records=[];excluded=[]
    for record in joined.to_dict('records'):
        try:
            if not np.isfinite([record['pitch_speed_mph'],*[record[f] for f in POI]]).all() or pd.isna(record['pitch_type']):raise ValueError('missing metadata/POI')
            record.update(extract(groups[record['session_pitch']]))
            records.append(record)
        except (ValueError,KeyError) as exc:excluded.append({'pitch':record['session_pitch'],'reason':str(exc)})
    frame=pd.DataFrame(records);extrema,pairs,group_excluded=comparisons(frame)
    for name,df in [('pitch_features',frame),('fast_slow_extrema',extrema),('all_distinct_speed_pairs',pairs)]:df.to_csv(out/f'{name}.csv',index=False)
    example=extrema[extrema.user==2].to_dict('records')
    report={'source':'https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching',
        'raw_metadata_pitches':len(meta),'valid_pitches':len(frame),'valid_athletes':int(frame.user.nunique()),
        'extrema_groups':len(extrema),'distinct_speed_pairs':len(pairs),
        'excluded_pitches':excluded,'excluded_groups':group_excluded,
        'split_athletes':frame.groupby('split').user.nunique().to_dict(),
        'ranked_features':summarize(extrema,pairs),'fixed_user2_example':example,
        'user2_pitch_features':frame[frame.user==2][['session_pitch','pitch_speed_mph',*FEATURES]].to_dict('records'),
        'causal_effect_identified':False,'previously_untouched_external_validation':False}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['valid_pitches','valid_athletes','extrema_groups','distinct_speed_pairs','excluded_groups','fixed_user2_example']}))
    print(json.dumps(report['ranked_features'][:4],indent=2))

if __name__=='__main__':main()
