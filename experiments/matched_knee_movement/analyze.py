"""Fixed matched observational analysis, not an intervention or causal model."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SEED = 20261113
FEATURES = ('lead_knee_change', 'lead_knee_level')
CONTROL_NAMES = ('speed_kmh', 'rpm', 'balls', 'strikes')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def context(row):
    return {**{k: int(float(row[k])) for k in ('inning', 'balls', 'strikes', 'outs_when_up', 'batter')},
            **{k: row[k] for k in ('stand', 'home_team', 'inning_topbot')},
            'bases': ''.join('0' if not row[k] or row[k].lower()=='nan' else '1' for k in ('on_1b', 'on_2b', 'on_3b'))}


def same_context(a, b):
    basic = all(a[k]==b[k] for k in ('inning', 'stand', 'home_team', 'inning_topbot', 'bases'))
    strict = basic and a['outs_when_up']==b['outs_when_up'] and a['batter']==b['batter']
    return {'basic': basic, 'strict': strict,
            'full_count': strict and a['balls']==b['balls'] and a['strikes']==b['strikes']}


def source_context(lab, result):
    speed = lab/'skenes-cross-game-motion-20261106/artifacts/skenes-cross-game'
    metadata = read(speed/'selection.private.json')['pitches']
    cohort = read(lab/'skenes-ff-motion-cohort-20261110/cohort.json')
    identities = {k: (p['game'], p['at_bat'], p['pitch_number']) for k,p in metadata.items()}
    for game in cohort['games']:
        for pair in game['chosen']:
            for side in ('low', 'high'):
                meta=pair[side];row=meta['row']
                identities[meta['play_id'][:8]] = tuple(int(float(row[k])) for k in ('game_pk','at_bat_number','pitch_number'))
    csvpath = speed/'skenes-2024.private.csv'
    rows={}
    with csvpath.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            identity=tuple(int(float(row[k])) for k in ('game_pk','at_bat_number','pitch_number'))
            if identity in rows: raise ValueError('Nonunique pitch identity')
            rows[identity]=row
    contexts={}
    for pair in result['pairs']:
        for key in pair['keys']:
            row=rows[identities[key]];clip=result['clips'][key]
            if row['pitch_type']!='FF' or int(row['pitcher'])!=694973:
                raise ValueError('Wrong pitcher or pitch type')
            expected={'speed_kmh':float(row['release_speed'])*1.609344,'rpm':float(row['release_spin_rate']),
                      'horizontal_cm':float(row['pfx_x'])*30.48,'vertical_cm':float(row['pfx_z'])*30.48}
            if any(abs(expected[k]-clip['record'][k])>1e-7 for k in expected):
                raise ValueError('Source outcome mismatch')
            contexts[key]={'identity':identities[key],**context(row)}
    return contexts, digest(csvpath)


def background_frame(clip):
    cap=cv2.VideoCapture(clip['video']);fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: raise ValueError('Missing source frame rate')
    target=int(clip['first_source_frame']-round(.5*fps))
    if target<0: raise ValueError('Background time before source starts')
    for _ in range(target+1):
        ok,image=cap.read()
        if not ok: raise ValueError('Background frame unavailable')
    pts=cap.get(cv2.CAP_PROP_POS_MSEC);cap.release()
    height=round(image.shape[0]*1280/image.shape[1])
    image=cv2.resize(image,(1280,height),interpolation=cv2.INTER_AREA)
    return image, {'frame':target,'pts_ms':float(pts),'fps':float(fps)}


def background_features(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);h,w=gray.shape
    mask=np.zeros_like(gray);mask[int(.18*h):int(.90*h),:int(.26*w)]=255
    mask[int(.18*h):int(.90*h),int(.74*w):]=255
    keypoints,descriptors=cv2.SIFT_create(nfeatures=3500).detectAndCompute(gray,mask)
    points=np.array([k.pt for k in keypoints],float).reshape(-1,2)
    # SIFT has subpixel centers; the raster mask alone can admit boundary centers.
    keep=((points[:,0]<.26*w)|(points[:,0]>=.74*w))&(points[:,1]>=.18*h)&(points[:,1]<=.90*h)
    return points[keep],descriptors[keep] if descriptors is not None else None,(w,h)


def camera_gate(metrics):
    return bool(metrics['matches']>=20 and metrics['inliers']>=20 and metrics['inlier_fraction']>=.35
                and min(metrics['side_inliers'])>=8 and metrics['span_x']>=.5 and metrics['span_y']>=.25
                and metrics['median_px']<=1 and metrics['p90_px']<=2.5
                and .90<=metrics['scale']<=1.10 and abs(metrics['rotation_deg'])<=1)


def compare_background(first, second):
    ap,ad,(w,h)=first;bp,bd,_=second
    empty={'matches':0,'inliers':0,'pass':False,'reason':'insufficient_background_matches'}
    if ad is None or bd is None or len(bd)<2:return empty
    matches=cv2.BFMatcher().knnMatch(ad,bd,k=2)
    good=[m for m,n in matches if m.distance<.7*n.distance]
    if len(good)<3:return {**empty,'matches':len(good)}
    a=np.array([ap[m.queryIdx] for m in good],float);b=np.array([bp[m.trainIdx] for m in good],float)
    cv2.setRNGSeed(SEED)
    matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=5000,confidence=.999)
    if matrix is None:return {**empty,'matches':len(good)}
    inside=inliers.ravel().astype(bool);aa=a[inside];error=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1)[inside]
    m={'matches':len(good),'inliers':int(inside.sum()),'inlier_fraction':float(inside.mean()),
       'side_inliers':[int((aa[:,0]<.26*w).sum()),int((aa[:,0]>.74*w).sum())],
       'span_x':float(np.ptp(aa[:,0])/w),'span_y':float(np.ptp(aa[:,1])/h),
       'median_px':float(np.median(error)),'p90_px':float(np.percentile(error,90)),
       'scale':float(np.hypot(matrix[0,0],matrix[1,0])),
       'rotation_deg':float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))),
       'center_shift_px':float(np.linalg.norm(matrix[:,:2]@np.array([w/2,h/2])+matrix[:,2]-np.array([w/2,h/2]))),
       'matrix':matrix.tolist()}
    m['pass']=camera_gate(m)
    return m


def weights(games):
    counts=Counter(games);w=np.array([1/counts[g] for g in games],float)
    return w/w.sum()


def fit(x,y,games):
    x,y=np.asarray(x,float),np.asarray(y,float);w=weights(games)
    xm=w@x;ym=float(w@y);xc=x-xm;yc=y-ym
    beta=np.linalg.solve(xc.T@(w[:,None]*xc)+np.eye(x.shape[1]),xc.T@(w*yc))
    return beta,float(ym-xm@beta)


def design(pair, feature, scenario, broad, control_mode='full'):
    value=pair['features'][feature]['grid'][scenario]
    if value is None:return None
    controls=[pair['delta']['speed_kmh'],pair['delta']['rpm']/100,pair['count_delta']['balls'],pair['count_delta']['strikes']]
    if control_mode=='speed_rpm':controls=controls[:2]
    elif control_mode!='full':raise ValueError('Unknown control mode')
    if broad and control_mode=='full':controls.append(pair['count_delta']['outs_when_up'])
    return np.array([*controls,value/5],float)


def model_gate(folds,total_pairs):
    predictions=[p for f in folds for p in f.get('predictions',[])]
    slopes=[f['knee_cm_per_degree'] for f in folds if 'knee_cm_per_degree' in f]
    signs=bool(slopes) and (all(s>0 for s in slopes) or all(s<0 for s in slopes))
    return (len(folds)>=3 and len(predictions)==total_pairs and len(slopes)==len(folds) and signs
            and all(f['mae']['m1']<min(f['mae'][k] for k in ('m0','mean','zero'))-1e-9 for f in folds))


def evaluate(pairs,feature,scenario=4,broad=False,control_mode='full'):
    folds=[]
    for game in sorted({p['game'] for p in pairs}):
        train=[p for p in pairs if p['game']!=game and design(p,feature,scenario,broad,control_mode) is not None]
        targets=[p for p in pairs if p['game']==game]
        fold={'game':game,'train_pairs':len(train),'train_games':sorted({p['game'] for p in train}),
              'candidate_test_pairs':len(targets),'predictions':[]}
        valid=[p for p in targets if design(p,feature,scenario,broad,control_mode) is not None]
        if len(train)<3 or len(fold['train_games'])<2 or not valid:
            fold['reason']='insufficient_training_or_test_support';folds.append(fold);continue
        x=np.array([design(p,feature,scenario,broad,control_mode) for p in train]);y=np.array([p['delta']['horizontal_cm'] for p in train])
        games=[p['game'] for p in train];w=weights(games)
        b0,i0=fit(x[:,:-1],y,games);b1,i1=fit(x,y,games)
        mean=float(w@y)
        nuisance=np.column_stack([np.ones(len(x)),x[:,:-1]])
        knee_fit=np.linalg.lstsq(nuisance*np.sqrt(w)[:,None],x[:,-1]*np.sqrt(w),rcond=None)[0]
        residual=x[:,-1]-nuisance@knee_fit
        fold.update({'knee_cm_per_degree':float(b1[-1]/5),'m0_coefficients':b0.tolist(),'m0_intercept':i0,
                     'm1_coefficients':b1.tolist(),'m1_intercept':i1,'constant_columns':np.flatnonzero(np.ptp(x,axis=0)<1e-12).tolist(),
                     'design_rank':int(np.linalg.matrix_rank(np.column_stack([np.ones(len(x)),x]))),
                     'design_columns_with_intercept':x.shape[1]+1,
                     'knee_residual_rms_deg':float(np.sqrt(w@(residual**2))*5)})
        for p in valid:
            z=design(p,feature,scenario,broad,control_mode);actual=p['delta']['horizontal_cm']
            estimates={'m0':float(i0+z[:-1]@b0),'m1':float(i1+z@b1),'mean':mean,'zero':0.}
            fold['predictions'].append({'pair':p['id'],'observed':actual,'estimates':estimates,
                                        'error':{k:abs(v-actual) for k,v in estimates.items()}})
        fold['mae']={k:float(np.mean([p['error'][k] for p in fold['predictions']])) for k in ('m0','m1','mean','zero')}
        folds.append(fold)
    usable=[f for f in folds if f.get('predictions')]
    predictions=[p for f in usable for p in f['predictions']]
    mae={k:float(np.mean([f['mae'][k] for f in usable])) if usable else None for k in ('m0','m1','mean','zero')}
    direction={k:{'correct':int(sum(np.sign(p['estimates'][k])==np.sign(p['observed']) for p in predictions if p['observed']!=0)),
                  'total':sum(p['observed']!=0 for p in predictions)} for k in ('m0','m1')}
    return {'feature':feature,'scenario':scenario,'candidate_pairs':len(pairs),'candidate_games':len(folds),
            'evaluated_pairs':len(predictions),'evaluated_games':len(usable),'mae_cm':mae,'direction':direction,
            'folds':folds,'pass':bool(model_gate(folds,len(pairs))) if pairs else False}


def run(lab,out):
    cv2.setNumThreads(2);out.mkdir(parents=True,exist_ok=True)
    previous=lab/'motion-outcome-replication-20261113/artifacts/motion-outcome-replication-001/results.private.json'
    prior=read(previous);contexts,csvhash=source_context(lab,prior)
    ordered=sorted(prior['pairs'],key=lambda p:(p['date'],p['id']))
    reference=ordered[0]['keys'][0];keys=list(dict.fromkeys(k for p in ordered for k in p['keys']))
    images={};camera_features={};clock={}
    for key in keys:
        c=prior['clips'][key]
        if digest(c['video'])!=c['source_sha256']:raise ValueError('Source hash changed')
        images[key],clock[key]=background_frame(c);camera_features[key]=background_features(images[key])
    camera={key:({'pass':True,'identity_reference':True} if key==reference else compare_background(camera_features[reference],camera_features[key])) for key in keys}
    rows=[]
    for p in ordered:
        a,b=p['keys'];cx=same_context(contexts[a],contexts[b]);bg=compare_background(camera_features[a],camera_features[b])
        rows.append({**p,'context_match':cx,'context':[contexts[a],contexts[b]],'pair_camera':bg,
                     'shared_camera':bool(bg['pass'] and camera[a]['pass'] and camera[b]['pass']),
                     'count_delta':{k:contexts[b][k]-contexts[a][k] for k in ('balls','strikes','outs_when_up')}})
    analyses={}
    for scope in ('primary','basic_context','pair_camera_only'):
        for cohort in sorted({p['cohort'] for p in rows}):
            chosen=[p for p in rows if p['cohort']==cohort and p['context_match']['basic']
                    and (p['shared_camera'] if scope!='pair_camera_only' else p['pair_camera']['pass'])
                    and (p['context_match']['strict'] if scope!='basic_context' else True)]
            records=[]
            for feature in FEATURES:
                scenarios=[evaluate(chosen,feature,s,scope=='basic_context') for s in range(9)]
                records.append({'feature':feature,'nominal':scenarios[4],'scenarios':scenarios,
                                'all_timing_pass':all(s['pass'] for s in scenarios)})
            analyses[scope+'/'+cohort]={'n_pairs':len(chosen),'n_games':len({p['game'] for p in chosen}),'models':records}
    result={'protocol_sha256':digest(Path(__file__).with_name('PROTOCOL.md')),'previous_result_sha256':digest(previous),
            'statcast_sha256':csvhash,'camera_reference':reference,'background_clock':clock,'camera_to_reference':camera,
            'rows':rows,'analyses':analyses,'source_clips':{k:prior['clips'][k] for k in keys},
            'interpretation':'selected retrospective 2D conditional association, not causal or Real2Sim intervention'}
    (out/'results.private.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Preserve a compact outcome-blind view of the frames used for camera matching.
    tiles=[]
    for p in rows:
        tile=np.hstack([cv2.resize(images[k],(640,360)) for k in p['keys']])
        cv2.putText(tile,f"{p['date']} {p['id']} view={p['shared_camera']} context={p['context_match']['strict']}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        tiles.append(tile)
    for i in range(0,len(tiles),4):cv2.imwrite(str(out/f'background-sheet-{i//4+1}.jpg'),np.vstack(tiles[i:i+4]))
    print('camera',sum(p['shared_camera'] for p in rows),'of',len(rows),'strict_context',sum(p['context_match']['strict'] for p in rows),flush=True)
    for name,a in analyses.items():
        for m in a['models']:print(name,m['feature'],a['n_pairs'],m['nominal']['evaluated_pairs'],m['nominal']['mae_cm'],'pass',m['all_timing_pass'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lab',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.lab.resolve(),args.out)
