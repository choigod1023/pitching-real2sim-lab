"""Target-free rigid hand-cluster audit and orientation smoothing pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import ezc3d
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter,sosfiltfilt
from scipy.spatial.transform import Rotation,RotationSpline
from ball_release import hand_frame
from pilot import regular_time,vector
from speed_path import retarget_velocity,rotation_velocity
from cohort_baseline import summarize


def proper_rotation(matrices):
    u,s,vt=np.linalg.svd(matrices)
    correction=np.tile(np.eye(3),(len(u),1,1))
    correction[:,-1,-1]=np.linalg.det(u@vt)
    return u@correction@vt


def fit_cluster(template,points):
    """NxKx3 observations, K>=3 non-collinear template points."""
    template,points=np.asarray(template),np.asarray(points)
    if template.ndim!=2 or template.shape[1]!=3 or points.ndim!=3 or points.shape[1:]!=template.shape:
        raise ValueError('Invalid marker dimensions')
    centered=template-template.mean(axis=0)
    if not np.isfinite(points).all() or not np.isfinite(template).all() or np.linalg.matrix_rank(centered)<2:
        raise ValueError('Nonfinite or collinear markers')
    world_centered=points-points.mean(axis=1,keepdims=True)
    if np.any(np.linalg.svd(world_centered,compute_uv=False)[:,1]<1e-8):
        raise ValueError('Observed cluster is collinear')
    rotations=proper_rotation(np.einsum('nki,kj->nij',world_centered,centered))
    origins=points.mean(axis=1)-np.einsum('nij,j->ni',rotations,template.mean(axis=0))
    reconstructed=origins[:,None,:]+np.einsum('nij,kj->nki',rotations,template)
    return origins,rotations,reconstructed


def smooth_cluster(template,origins,rotations,rate,sigma_s=1/(2*np.pi*20)):
    if not np.isfinite([rate,sigma_s]).all() or min(rate,sigma_s)<=0:
        raise ValueError('Positive finite rate and smoothing width required')
    # This is matrix-Gaussian smoothing + SO(3) projection, not Butterworth20.
    smooth_r=proper_rotation(gaussian_filter1d(rotations,sigma_s*rate,axis=0,mode='nearest'))
    sos=butter(4,20,fs=rate,output='sos')
    smooth_o=sosfiltfilt(sos,origins,axis=0)
    points=smooth_o[:,None,:]+np.einsum('nij,kj->nki',smooth_r,template)
    return smooth_o,smooth_r,points


def distances(points):
    return np.stack([np.linalg.norm(points[:,a]-points[:,b],axis=1) for a,b in [(0,1),(0,2),(1,2)]],axis=1)


def static_template(c):
    labels=[s.strip() for s in c['parameters']['POINT']['LABELS']['value']]
    required=['RFIN','RWRA','RWRB']
    # Accept one consistent skeleton namespace only; never mix ambiguous bodies.
    namespaces=[prefix for prefix in ('','Skeleton_001_')
                if all(labels.count(prefix+n)==1 for n in required)]
    if len(namespaces)!=1:raise ValueError('Missing or ambiguous static hand namespace')
    indices=[labels.index(namespaces[0]+n) for n in required]
    points=c['data']['points'][:,indices,:]
    scale={'m':1.,'mm':.001,'cm':.01}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
    raw=points[:3].transpose(2,1,0)*scale
    valid=np.isfinite(raw).all(axis=(1,2)) & (points[3]>=0).all(axis=0)
    valid &= np.linalg.norm(np.cross(raw[:,1]-raw[:,0],raw[:,2]-raw[:,0]),axis=1)>1e-8
    if not valid.any():raise ValueError('No valid static hand frame')
    raw=raw[valid];origin=(raw[:,1]+raw[:,2])/2
    _,frame=hand_frame(raw[:,0],raw[:,1],raw[:,2])
    return np.median(np.einsum('nji,nkj->nki',frame,raw-origin[:,None,:]),axis=0)


def release_times(end,rate,t,offsets):
    if not np.isfinite(rate) or rate<=0:raise ValueError('Invalid sampling rate')
    offsets=np.asarray(offsets,dtype=float)
    if offsets.ndim!=1 or not len(offsets) or not np.isfinite(offsets).all():raise ValueError('Invalid offsets')
    if np.any(offsets!=np.round(offsets)) or len(np.unique(offsets))!=len(offsets):raise ValueError('Unique integer frame offsets required')
    times=end+offsets/rate
    if np.any(times<t[1]) or np.any(times>t[-2]):raise ValueError('Release outside supported interval')
    return times


def run(data,cohort,out,template_source='trial',offsets=(0,)):
    if template_source not in ('trial','static'):raise ValueError('Unknown template source')
    data,cohort,out=Path(data),Path(cohort),Path(out);out.mkdir(parents=True,exist_ok=True)
    selection=pd.read_csv(cohort/'selection.csv');meta=pd.read_csv(data/'metadata.csv').set_index('session_pitch')
    landmarks=pd.read_csv(data/'full_sig/landmarks.csv');geometry=[];predictions=[];sources=[]
    for item in selection.itertuples():
        row=meta.loc[item.session_pitch];paths=list((data/'c3d').rglob(row.filename_new))
        if len(paths)!=1:raise ValueError('Ambiguous C3D')
        c=ezc3d.c3d(str(paths[0]));labels=[s.strip() for s in c['parameters']['POINT']['LABELS']['value']]
        sources.append({'pitch':item.session_pitch,'c3d_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest()})
        rate=float(c['parameters']['POINT']['RATE']['value'][0]);scale={'m':1.,'cm':.01,'mm':.001}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
        markers=[]
        for name in ['RFIN','RWRA','RWRB']:
            p=c['data']['points'][:,labels.index(name),:].T
            if not np.isfinite(p).all() or np.any(p[:,3]<0):raise ValueError('Missing marker')
            markers.append(p[:,:3]*scale)
        raw=np.stack(markers,axis=1)
        g=landmarks[landmarks.session_pitch.eq(item.session_pitch)].sort_values('time');t=regular_time(g)
        if len(t)!=len(raw) or not np.allclose(t,np.arange(len(t))/rate,atol=.000051,rtol=0):raise ValueError('Unverified alignment')
        start,end=float(g.fp_10_time.iloc[0]),float(g.BR_time.iloc[0]);mask=(t>=start)&(t<=end)
        if mask.sum()<5 or not t[1]<start<end<t[-2]:raise ValueError('Invalid analysis interval')
        origin=(raw[:,1]+raw[:,2])/2
        _,frame=hand_frame(*markers)
        local=np.einsum('nji,nkj->nki',frame,raw-origin[:,None,:])
        template=np.median(local[mask],axis=0)
        if template_source=='static':
            static_paths=list((data/'c3d').rglob(row.modelname_new))
            if len(static_paths)!=1:raise ValueError('Ambiguous static C3D')
            template=static_template(ezc3d.c3d(str(static_paths[0])))
            sources[-1]['static_sha256']=hashlib.sha256(static_paths[0].read_bytes()).hexdigest()
        fit_o,fit_r,fit=fit_cluster(template,raw)
        smooth_o,smooth_r,smooth=smooth_cluster(template,fit_o,fit_r,rate)
        cart=sosfiltfilt(butter(4,20,fs=rate,output='sos'),raw,axis=0)
        _,cart_r=hand_frame(cart[:,0],cart[:,1],cart[:,2])
        gaussian_cart=gaussian_filter1d(raw,rate/(2*np.pi*20),axis=0,mode='nearest')
        _,gaussian_cart_r=hand_frame(gaussian_cart[:,0],gaussian_cart[:,1],gaussian_cart[:,2])
        fit_residual=np.sqrt(np.mean(np.sum((fit[mask]-raw[mask])**2,axis=2)))
        geometry.append({'user':int(item.user),'pitch':item.session_pitch,
                         'raw_pair_distance_range_max_mm':float(np.ptp(distances(raw)[mask],axis=0).max()*1000),
                         'cartesian20_pair_distance_range_max_mm':float(np.ptp(distances(cart)[mask],axis=0).max()*1000),
                         'cartesian_gaussian_pair_distance_range_max_mm':float(np.ptp(distances(gaussian_cart)[mask],axis=0).max()*1000),
                         'smoothed_rigid_pair_distance_range_max_mm':float(np.ptp(distances(smooth)[mask],axis=0).max()*1000),
                         'rigid_fit_marker_rmse_mm':float(fit_residual*1000),
                         'smoothed_to_raw_marker_rmse_mm':float(np.sqrt(np.mean(np.sum((smooth[mask]-raw[mask])**2,axis=2)))*1000)})
        curve=CubicSpline(t,np.stack([vector(g,n+'_jc') for n in ['shoulder','elbow','wrist']],axis=1))
        with np.load(cohort/'cases'/item.session_pitch/'20/reference-0.npz') as cache:
            lengths=cache['lengths'].copy()
        eval_times=release_times(end,rate,t,offsets)
        for method,r in [('raw_frame',frame),('cartesian20',cart_r),('cartesian_gaussian',gaussian_cart_r),('rigid_fit',fit_r),('rigid_matrix_smooth',smooth_r)]:
            rotation_curve=RotationSpline(t,Rotation.from_matrix(r))
            for shift,eval_time in zip(offsets,eval_times):
                wrist_v=retarget_velocity(curve(eval_time),curve(eval_time,1),lengths)
                v=wrist_v+rotation_velocity(rotation_curve,eval_time,np.array([.15,0.,0.]))
                predictions.append({'user':int(item.user),'pitch':item.session_pitch,'method':method,
                                    'offset_frames':int(shift),'offset_ms':float(shift/rate*1000),
                                    'predicted_kmh':float(np.linalg.norm(v)*3.6),'observed_kmh':float(row.pitch_speed_mph)*1.609344})
    geo=pd.DataFrame(geometry);pred=pd.DataFrame(predictions);scores=[]
    for (method,shift),g in pred.groupby(['method','offset_frames']):
        # Reuse scoring only; this is an analytic point velocity, not rollout.
        score=summarize(g.assign(cutoff_hz=0.))[0];score.pop('cutoff_hz');scores.append({'method':method,'offset_frames':int(shift),**score})
    report={'pitches':len(geo),'athletes':int(geo.user.nunique()),'scores':scores,'sources':sources,
            'selection_sha256':hashlib.sha256((cohort/'selection.csv').read_bytes()).hexdigest(),
            'geometry_pitch_weighted_mean':{c:float(geo[c].mean()) for c in geo.columns if c.endswith('_mm')},
            'geometry_pitch_max':{c:float(geo[c].max()) for c in geo.columns if c.endswith('_mm')},
            'template':('Per-trial median local marker positions during foot-contact to BR; no speed targets. Offline fitting, not a static anatomical calibration.' if template_source=='trial' else 'Session-matched static C3D median local marker positions; no speed targets. Marker geometry only, not anatomical axes or ball grip.'),
            'smoothing':'Gaussian rotation-matrix smoothing sigma=1/(2*pi*20) seconds then proper SO(3) projection; centroid origin uses Butterworth20. Not frequency-equivalent to Cartesian Butterworth20.',
            'evaluation':'All five methods on the same inspected development cohort; same retargeted wrist velocity and 15cm point offset. Cartesian Gaussian control uses the same sigma as matrix smoothing. Analytic kinematics only, not newly validated dynamics.',
            'limits':['Preserving distances is imposed, not proof that raw markers are truly rigid.',
                      'Three markers determine an observable frame but skin/finger motion and noise remain.',
                      'No actual ball geometry, finger slip, independent test cohort or intervention validation.'],
            'predicted_gain_kmh':None,'coaching_ready':False}
    geo.to_csv(out/'geometry.csv',index=False);pred.to_csv(out/'predictions.csv',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('artifacts/hand-geometry'))
    p.add_argument('--template-source',choices=['trial','static'],default='trial')
    p.add_argument('--timing-audit',action='store_true')
    a=p.parse_args();r=run(a.data,a.cohort,a.out,a.template_source,(-2,-1,0,1,2) if a.timing_audit else (0,));print(json.dumps(r,indent=2))
