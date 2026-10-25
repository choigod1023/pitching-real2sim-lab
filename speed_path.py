"""Offline vector-by-vector release speed decomposition, without target fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import ezc3d
import mujoco
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, RotationSpline
from ball_release import hand_frame
from coupled_payload import model_for
from torque_arm import derivatives
from pilot import regular_time, vector


def direction_derivative(segment, velocity):
    length=np.linalg.norm(segment)
    if not np.isfinite([*segment,*velocity]).all() or length<1e-8:
        raise ValueError('Degenerate segment')
    u=segment/length
    return (velocity-u*np.dot(u,velocity))/length


def retarget_velocity(points, velocity, lengths):
    s,e,w=points;sd,ed,wd=velocity
    return sd+lengths[0]*direction_derivative(e-s,ed-sd)+lengths[1]*direction_derivative(w-e,wd-ed)


def rotation_velocity(rotation_curve, time, offset, epsilon=1e-5):
    return (rotation_curve(time+epsilon).as_matrix()@offset-rotation_curve(time-epsilon).as_matrix()@offset)/(2*epsilon)


def run(data, cohort, out):
    data,cohort,out=Path(data),Path(cohort),Path(out);out.mkdir(parents=True,exist_ok=True)
    predictions=pd.read_csv(cohort/'predictions.csv');selection=pd.read_csv(cohort/'selection.csv')
    meta=pd.read_csv(data/'metadata.csv').set_index('session_pitch')
    landmarks=pd.read_csv(data/'full_sig/landmarks.csv')
    records=[];audits=[]
    for item in selection.itertuples():
        pitch=item.session_pitch;row=meta.loc[pitch]
        paths=list((data/'c3d').rglob(row.filename_new))
        if len(paths)!=1:raise ValueError('Ambiguous C3D')
        c=ezc3d.c3d(str(paths[0]));labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
        rate=float(c['parameters']['POINT']['RATE']['value'][0])
        scale={'m':1.,'cm':.01,'mm':.001}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
        raw=[]
        for marker in ['RFIN','RWRA','RWRB']:
            p=c['data']['points'][:,labels.index(marker),:].T
            if not np.isfinite(p).all() or np.any(p[:,3]<0):raise ValueError('Missing markers')
            raw.append(p[:,:3]*scale)
        g=landmarks[landmarks.session_pitch.eq(pitch)].sort_values('time');t=regular_time(g)
        if len(t)!=len(raw[0]) or not np.allclose(t,np.arange(len(t))/rate,atol=.000051,rtol=0):
            raise ValueError('Unverified timing')
        release=float(g.BR_time.iloc[0])
        if not t[1]<release<t[-2]:raise ValueError('Release outside derivative support')
        points_curve=CubicSpline(t,np.stack([vector(g,n+'_jc') for n in ['shoulder','elbow','wrist']],axis=1))
        points=points_curve(release);vel=points_curve(release,1)
        for cutoff in [0.,10.,20.,30.]:
            case=cohort/'cases'/pitch/str(int(cutoff))
            with np.load(case/'reference-0.npz') as r:
                lengths=r['lengths'].copy();q=r['qpos'].copy();dt=float(r['dt'])
            m=model_for(lengths,dt);d=mujoco.MjData(m)
            qvel,_=derivatives(m,q,dt);d.qpos[:]=q[-1];d.qvel[:]=qvel[-1];mujoco.mj_forward(m,d)
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
            mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
            target_endpoint=jp@d.qvel
            if cutoff:
                sos=butter(4,cutoff,fs=rate,output='sos');marks=[sosfiltfilt(sos,p,axis=0) for p in raw]
            else:marks=raw
            _,frames=hand_frame(*marks);rot=RotationSpline(t,Rotation.from_matrix(frames))
            mid=(marks[1]+marks[2])/2
            marker_wrist=CubicSpline(t,mid)(release,1)
            fin=CubicSpline(t,marks[0])(release,1)
            model_wrist=retarget_velocity(points,vel,lengths)
            rotational=rotation_velocity(rot,release,np.array([.15,0.,0.]))
            exact_center=model_wrist+rotational
            matched=predictions[predictions.pitch.eq(pitch)&predictions.cutoff_hz.eq(cutoff)]
            if len(matched)!=1:raise ValueError('Expected matching successful cohort prediction')
            speed=lambda v:float(np.linalg.norm(v)*3.6)
            records.append({'user':int(item.user),'pitch':pitch,'cutoff_hz':cutoff,
                            'observed_kmh':float(row.pitch_speed_mph)*1.609344,
                            'rfin_proxy_kmh':speed(fin),'marker_wrist_kmh':speed(marker_wrist),
                            'processed_wrist_kmh':speed(vel[2]),'retargeted_wrist_kmh':speed(model_wrist),
                            'rotational_component_norm_kmh':speed(rotational),
                            'exact_assumed_ball_kmh':speed(exact_center),'endpoint_reference_kmh':speed(target_endpoint),
                            'integrated_ball_kmh':float(matched.predicted_kmh.iloc[0]),
                            'retargeting_speed_difference_kmh':speed(model_wrist)-speed(vel[2]),
                            'rotation_speed_increment_kmh':speed(exact_center)-speed(model_wrist),
                            'endpoint_derivative_difference_kmh':speed(target_endpoint)-speed(exact_center),
                            'integration_speed_difference_kmh':float(matched.predicted_kmh.iloc[0])-speed(target_endpoint),
                            'marker_to_processed_wrist_distance_mm':float(np.linalg.norm(CubicSpline(t,mid)(release)-points[2])*1000),
                            'rfin_distance_from_marker_wrist_mm':float(np.linalg.norm(CubicSpline(t,marks[0]-mid)(release))*1000),
                            **{f'{name}_{axis}_mps':float(v[k]) for name,v in [('model_wrist',model_wrist),('rotation',rotational),('ball',exact_center)] for k,axis in enumerate('xyz')}})
        audits.append({'pitch':pitch,'rate_hz':rate,'release_time_s':release,'sample_count':len(t),
                       'source_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest()})
    frame=pd.DataFrame(records);frame.to_csv(out/'decomposition.csv',index=False)
    summary=[]
    columns=['rfin_proxy_kmh','marker_wrist_kmh','processed_wrist_kmh','retargeted_wrist_kmh',
             'exact_assumed_ball_kmh','endpoint_reference_kmh','integrated_ball_kmh',
             'retargeting_speed_difference_kmh','rotation_speed_increment_kmh',
             'endpoint_derivative_difference_kmh','integration_speed_difference_kmh',
             'marker_to_processed_wrist_distance_mm','rfin_distance_from_marker_wrist_mm']
    for cutoff,g in frame.groupby('cutoff_hz'):
        summary.append({'cutoff_hz':cutoff,'pitch_weighted_means':{c:float(g[c].mean()) for c in columns},
                        'absolute_difference_mean':{c:float(g[c].abs().mean()) for c in columns if 'difference' in c},
                        'absolute_difference_max':{c:float(g[c].abs().max()) for c in columns if 'difference' in c},
                        'observed_mean_kmh':float(g.observed_kmh.mean())})
    report={'rows':len(frame),'pitches':int(frame.pitch.nunique()),'summary':summary,'input_audit':audits,
            'selection_sha256':hashlib.sha256((cohort/'selection.csv').read_bytes()).hexdigest(),
            'scope':'Vector decomposition on the same previously inspected development trials. No parameter is fitted to observed ball speed.',
            'derivative_note':'Exact assumed point means continuous spline/normalized-direction derivative at recorded event. Rotation uses symmetric 10 microsecond difference, not exact closed form.',
            'limits':['RFIN and wrist speeds are proxies, not measured ball speeds.',
                      'Norms of vector components do not add; speed increments are successive norm differences.',
                      'Processed joint centers and marker midpoints are different anatomical definitions and may have different filtering.',
                      'Event alignment confirms sample indexing, not true physical release timing or radar measurement location.'],
            'predicted_gain_kmh':None}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('artifacts/speed-path'))
    a=p.parse_args();r=run(a.data,a.cohort,a.out);print(json.dumps(r['summary'],indent=2))
