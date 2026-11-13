"""Observed within-pitcher links; never substitute inputs for predicted gains."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ANGULAR={'arm_slot','spin_axis','upperarm','forearm'}
LINKS=[('knee',x) for x in ('release_x','release_z','extension','arm_slot','spin_axis','upperarm','forearm')]
LINKS += [('arm_slot','spin_axis'),('forearm','spin_axis'),('spin_axis','horizontal_cm'),('spin_axis','vertical_cm')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def wrap(x):
    return (np.asarray(x)+180)%360-180


def weights(games):
    counts=Counter(games);w=np.array([1/counts[g] for g in games],float)
    return w/w.sum()


def angular_mean(y,w):
    v=np.exp(1j*np.radians(y));z=w@v
    if abs(z)<1e-8:raise ValueError('Undefined circular mean')
    return float(np.degrees(np.angle(z)))


def fit(x,y,games,angular=False):
    x=np.asarray(x,float);y=np.asarray(y,float);w=weights(games)
    if angular:
        center=angular_mean(y,w);y=center+wrap(y-center)
    mx=w@x;my=float(w@y);xc=x-mx
    beta=np.linalg.solve(xc.T@(w[:,None]*xc)+np.eye(x.shape[1]),xc.T@(w*(y-my)))
    return beta,float(my-mx@beta),angular_mean(y,w) if angular else my


def circular_median(values):
    if not len(values):return None
    local=values[0]+wrap(values-values[0])
    return None if np.ptp(local)>=180 else float(wrap(np.median(local)))


def consecutive(mask):
    longest=run=0
    for value in mask:
        run=run+1 if value else 0;longest=max(longest,run)
    return longest


def direction_trace(points,scores,ids,size,nominal,excluded):
    good=np.isfinite(points[:,ids]).all((1,2))&np.isfinite(scores[:,ids]).all(1)&(scores[:,ids]>=.5).all(1)
    p=points[:,ids]
    good&=(p[:,:,0]>=0).all(1)&(p[:,:,0]<size[0]).all(1)&(p[:,:,1]>=0).all(1)&(p[:,:,1]<size[1]).all(1)
    vector=p[:,1]-p[:,0];length=np.linalg.norm(vector,axis=1)
    if good.any():
        typical=float(np.median(length[good]));good&=(length>=max(8,.35*typical))&(length<=2*typical)
    good&=~np.isin(nominal,excluded)
    angle=np.degrees(np.arctan2(vector[:,0],-vector[:,1]))
    return np.where(good,angle,np.nan),good


def window_features(values,times,first):
    output=[]
    for shift in (-1,0,1):
        index=first+shift
        if index<0 or index>=len(times):output.append(None);continue
        relative=times-times[index]
        if relative[0]>-70 or relative[-1]<-15:output.append(None);continue
        selected=values[(relative>=-70)&(relative<=-15)];valid=np.isfinite(selected)
        if len(selected)<4 or valid.sum()<3 or consecutive(valid)<3:
            output.append(None);continue
        output.append(circular_median(selected[valid]))
    return output


def paired(a,b,angular=False):
    return [None if x is None or y is None else float(wrap(y-x) if angular else y-x) for x in a for y in b]


def source_records(lab,previous):
    path=lab/'skenes-cross-game-motion-20261106/artifacts/skenes-cross-game/skenes-2024.private.csv'
    data={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            key=tuple(int(float(row[k])) for k in ('game_pk','at_bat_number','pitch_number'))
            if key in data:raise ValueError('Nonunique pitch row')
            data[key]=row
    output={}
    fields={'release_x':'release_pos_x','release_y':'release_pos_y','release_z':'release_pos_z',
            'extension':'release_extension','arm_slot':'arm_angle','spin_axis':'spin_axis'}
    for p in previous['rows']:
        for key,ctx in zip(p['keys'],p['context']):
            row=data[tuple(ctx['identity'])]
            if int(row['pitcher'])!=694973 or row['pitch_type']!='FF':raise ValueError('Wrong pitch')
            record={k:float(row[v])*(30.48 if k not in ANGULAR else 1) for k,v in fields.items()}
            record.update({'speed_kmh':float(row['release_speed'])*1.609344,'rpm':float(row['release_spin_rate']),
                           'horizontal_cm':float(row['pfx_x'])*30.48,'vertical_cm':float(row['pfx_z'])*30.48})
            if not all(np.isfinite(x) for x in record.values()):raise ValueError('Missing record')
            old=previous['source_clips'][key]['record']
            if any(abs(record[k]-old[k])>1e-7 for k in old):raise ValueError('Same-pitch record drift')
            output[key]=record
    return output,digest(path)


def design(p,input_name,output_name,scenario,mode):
    x=p['measurements'][input_name][scenario];y=p['measurements'][output_name][scenario]
    if x is None or y is None:return None
    controls=[p['delta']['speed_kmh'],p['delta']['rpm']/100]
    if mode=='full':controls += [p['count_delta']['balls'],p['count_delta']['strikes']]
    elif mode!='speed_rpm':raise ValueError('Unknown control set')
    return np.array([*controls,x/5],float),float(y)


def evaluate(pairs,input_name,output_name,scenario=4,mode='full'):
    folds=[];angular=output_name in ANGULAR
    for game in sorted({p['game'] for p in pairs}):
        train=[p for p in pairs if p['game']!=game and design(p,input_name,output_name,scenario,mode) is not None]
        test=[p for p in pairs if p['game']==game and design(p,input_name,output_name,scenario,mode) is not None]
        f={'game':game,'train_games':sorted({p['game'] for p in train}),'train_pairs':len(train),'predictions':[]}
        if len(train)<3 or len(f['train_games'])<2 or not test:
            f['reason']='insufficient_support';folds.append(f);continue
        values=[design(p,input_name,output_name,scenario,mode) for p in train];x=np.array([v[0] for v in values]);y=np.array([v[1] for v in values]);g=[p['game'] for p in train]
        b0,i0,mean=fit(x[:,:-1],y,g,angular);b1,i1,_=fit(x,y,g,angular);w=weights(g)
        nuisance=np.column_stack([np.ones(len(x)),x[:,:-1]])
        residual=x[:,-1]-nuisance@np.linalg.lstsq(nuisance*np.sqrt(w[:,None]),x[:,-1]*np.sqrt(w),rcond=None)[0]
        f.update({'coefficient_per_unit':float(b1[-1]/5),'m0_beta':b0.tolist(),'m0_intercept':i0,'m1_beta':b1.tolist(),'m1_intercept':i1,
                  'rank':int(np.linalg.matrix_rank(np.column_stack([np.ones(len(x)),x]))),'columns':x.shape[1]+1,
                  'input_residual_rms':float(np.sqrt(w@(residual**2))*5)})
        for p in test:
            z,y=design(p,input_name,output_name,scenario,mode)
            estimates={'m0':float(i0+z[:-1]@b0),'m1':float(i1+z@b1),'mean':mean,'zero':0.}
            if angular:estimates={k:float(wrap(v)) for k,v in estimates.items()}
            errors={k:float(abs(wrap(v-y) if angular else v-y)) for k,v in estimates.items()}
            f['predictions'].append({'pair':p['id'],'observed':y,'estimates':estimates,'errors':errors})
        f['mae']={k:float(np.mean([p['errors'][k] for p in f['predictions']])) for k in ('m0','m1','mean','zero')};folds.append(f)
    valid=[f for f in folds if f['predictions']];n=sum(len(f['predictions']) for f in valid)
    mae={k:float(np.mean([f['mae'][k] for f in valid])) if valid else None for k in ('m0','m1','mean','zero')}
    slopes=[f['coefficient_per_unit'] for f in valid];sign=bool(slopes) and (min(slopes)>0 or max(slopes)<0)
    passes=len(valid)>=3 and n==len(pairs) and len(valid)==len(folds) and sign and all(f['mae']['m1']<min(f['mae'][k] for k in ('m0','mean','zero'))-1e-9 for f in valid)
    return {'input':input_name,'output':output_name,'mode':mode,'scenario':scenario,'units':'deg' if angular else 'cm',
            'candidate_pairs':len(pairs),'evaluated_pairs':n,'evaluated_games':len(valid),'mae':mae,'folds':folds,'repeat_pass':bool(passes)}


def simulation_gate(statistical_body_link,confirmed_camera,same_pitch_3d,force_work,hand_ball_spin,baseline_validation):
    conditions={'statistical_body_link':bool(statistical_body_link),'confirmed_crossgame_camera':bool(confirmed_camera),
                'same_pitch_3d':bool(same_pitch_3d),'force_work_budget':bool(force_work),
                'hand_ball_spin_generation':bool(hand_ball_spin),'baseline_validation':bool(baseline_validation)}
    return {'conditions':conditions,'allow_body_intervention':all(conditions.values()),
            'unmet':[k for k,v in conditions.items() if not v],
            'predicted_gain':{'speed_kmh':None,'rpm':None,'horizontal_cm':None,'vertical_cm':None}}


def run(lab,out):
    out.mkdir(parents=True,exist_ok=True)
    parent=lab/'matched-knee-movement-20261113/artifacts/matched-knee-movement-001/results.private.json'
    cache=lab/'motion-outcome-replication-20261113/artifacts/motion-outcome-replication-001/mapped-tracks.private.npz'
    previous=read(parent);records,csvhash=source_records(lab,previous);z=np.load(cache,allow_pickle=False)
    features={};traces={};total=0;finger_finite=0
    for key,clip in previous['source_clips'].items():
        if digest(clip['video'])!=clip['source_sha256']:raise ValueError('Changed source video')
        points=z[key+'_points'];scores=z[key+'_scores'];frames=z[key+'_frames'];times=z[key+'_times'];total+=len(points)
        finger_finite+=int(np.isfinite(points[:,[18,20,22]]).all(-1).sum())
        if len(clip['frame_offsets'])!=1:raise ValueError('Ambiguous source-frame mapping')
        nominal=frames-clip['frame_offsets'][0];features[key]={}
        for name,ids in (('upperarm',[12,14]),('forearm',[14,16])):
            values,mask=direction_trace(points,scores,ids,clip['source_size'],nominal,clip['exclude'].get('arm',[]))
            features[key][name]=window_features(values,times,clip['first_index'])
            traces[key+'_'+name]=values;traces[key+'_'+name+'_mask']=mask
    rows=[]
    for p in previous['rows']:
        a,b=p['keys'];m={'knee':p['features']['lead_knee_change']['grid']}
        for key in records[a]:
            m[key]=paired([records[a][key]]*3,[records[b][key]]*3,key in ANGULAR)
        for name in ('upperarm','forearm'):m[name]=paired(features[a][name],features[b][name],True)
        rows.append({**p,'measurements':m})
    analyses={}
    for cohort in sorted({p['cohort'] for p in rows}):
        chosen=[p for p in rows if p['cohort']==cohort and p['context_match']['strict'] and p['pair_camera']['pass']]
        links=[]
        for input_name,output_name in LINKS:
            modes={}
            for mode in ('full','speed_rpm'):
                s=[evaluate(chosen,input_name,output_name,i,mode) for i in range(9)]
                modes[mode]={'nominal':s[4],'scenarios':s,'all_timing_pass':all(v['repeat_pass'] for v in s)}
            links.append({'input':input_name,'output':output_name,'modes':modes})
        analyses[cohort]={'n_pairs':len(chosen),'n_games':len({p['game'] for p in chosen}),'links':links}
    body_pass=any(link['input']=='knee' and link['modes']['full']['all_timing_pass'] for g in analyses.values() for link in g['links'])
    result={'protocol_sha256':digest(Path(__file__).with_name('PROTOCOL.md')),'parent_sha256':digest(parent),'cache_sha256':digest(cache),'csv_sha256':csvhash,
            'records':records,'video_features':features,'source_clips':previous['source_clips'],'rows':rows,'analyses':analyses,
            'observation_audit':{'clips':len(records),'frames':total,'finite_finger_point_frames':finger_finite,'hand_direction':'structurally_unavailable_body_only_detector',
                                 'spin_axis':'recorded_2d_xz_angle_not_3d_vector'},
            'real2sim':simulation_gate(body_pass,False,False,False,False,False)}
    (out/'results.private.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'direction-traces.private.npz',**traces)
    print('observation',result['observation_audit'],flush=True)
    for cohort,g in analyses.items():
        for link in g['links']:
            m=link['modes']['full'];n=m['nominal']
            print(cohort,link['input'],link['output'],n['evaluated_pairs'],n['mae'],'repeat',m['all_timing_pass'],flush=True)
    print('real2sim',result['real2sim'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lab',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.lab.resolve(),a.out)
