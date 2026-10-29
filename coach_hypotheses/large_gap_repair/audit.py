"""Separate hand-axis definitions and wrist transfer on the original five trials."""
import argparse
import hashlib
import json
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt
from scipy.spatial.transform import Rotation,RotationSpline

from ball_release import hand_frame
from pilot import regular_time
from speed_path import retarget_velocity,rotation_velocity
from ..stage2.uncertainty import prepare_case,PITCHES


def center_axis_frames(longitudinal, transverse):
    longitudinal,transverse=map(lambda a:np.asarray(a,float),(longitudinal,transverse))
    if longitudinal.ndim!=2 or longitudinal.shape[1]!=3 or longitudinal.shape!=transverse.shape or not np.isfinite([longitudinal,transverse]).all():
        raise ValueError('invalid_axis_arrays')
    length=np.linalg.norm(longitudinal,axis=1)
    if np.any(length<1e-8):raise ValueError('degenerate_long_axis')
    x=longitudinal/length[:,None]
    y=transverse-(transverse*x).sum(axis=1)[:,None]*x
    width=np.linalg.norm(y,axis=1)
    if np.any(width<1e-8):raise ValueError('degenerate_transverse_axis')
    y=y/width[:,None]
    return np.stack([x,y,np.cross(x,y)],axis=-1)


def fixed_wrist_position(points, lengths):
    points=np.asarray(points,float);segments=np.diff(points,axis=0)
    norms=np.linalg.norm(segments,axis=1)
    if points.shape!=(3,3) or np.any(norms<1e-8):raise ValueError('invalid_arm_points')
    return points[0]+np.sum(np.asarray(lengths)[:,None]*segments/norms[:,None],axis=0)


def load_source(data, pitch):
    data=Path(data)
    metadata=pd.read_csv(data/'metadata.csv',usecols=['session_pitch','filename_new'])
    row=metadata[metadata.session_pitch.eq(pitch)]
    if len(row)!=1:raise ValueError('nonunique_pitch')
    paths=list((data/'c3d').rglob(row.iloc[0].filename_new))
    if len(paths)!=1:raise ValueError('nonunique_c3d')
    source=ezc3d.c3d(str(paths[0]));point=source['parameters']['POINT'];labels=[s.strip() for s in point['LABELS']['value']]
    rate=float(point['RATE']['value'][0]);scale={'m':1.,'cm':.01,'mm':.001}[point['UNITS']['value'][0].strip().lower()]
    if rate!=360.:raise ValueError('unexpected_rate')
    markers=[]
    for label in ('RFIN','RWRA','RWRB'):
        values=source['data']['points'][:,labels.index(label),:].T
        if not np.isfinite(values).all() or np.any(values[:,3]<0):raise ValueError('invalid_hand_marker')
        markers.append(sosfiltfilt(butter(4,20,fs=rate,output='sos'),values[:,:3]*scale,axis=0))
    columns=[f'{name}_jc_{axis}' for name in ['shoulder','elbow','wrist','hand'] for axis in 'xyz']
    table=pd.read_csv(data/'full_sig/landmarks.csv',usecols=['session_pitch','time','fp_10_time','BR_time',*columns])
    frame=table[table.session_pitch.eq(pitch)].sort_values('time');time=regular_time(frame)
    centers=frame[columns].to_numpy(float).reshape(len(frame),4,3)
    if len(time)!=len(markers[0]) or not np.allclose(time,np.arange(len(time))/rate,atol=.000051,rtol=0) or not np.isfinite(centers).all():
        raise ValueError('source_alignment_or_centers')
    if frame.BR_time.nunique()!=1 or frame.fp_10_time.nunique()!=1:raise ValueError('nonconstant_event')
    curve=CubicSpline(time,centers);br=float(frame.BR_time.iloc[0]);fp=float(frame.fp_10_time.iloc[0])
    lengths=np.median(np.linalg.norm(np.diff(curve(np.linspace(fp,br,100))[:,:3],axis=1),axis=2),axis=0)
    _,surface=hand_frame(*markers)
    anatomical=center_axis_frames(centers[:,3]-centers[:,2],markers[1]-markers[2])
    return {'pitch':pitch,'time':time,'curve':curve,'br':br,'lengths':lengths,
            'mid':CubicSpline(time,(markers[1]+markers[2])/2),
            'fin':CubicSpline(time,markers[0]),
            'surface':RotationSpline(time,Rotation.from_matrix(surface)),
            'center_axis':RotationSpline(time,Rotation.from_matrix(anatomical)),
            'c3d_path':str(paths[0].resolve()),'c3d_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest()}


def evaluate_source(source):
    t=source['br'];curve=source['curve'];points=curve(t)[:3];velocity=curve(t,1)[:3]
    fixed=retarget_velocity(points,velocity,source['lengths']);processed=velocity[2];marker=source['mid'](t,1)
    rows=[]
    for frame_name in ('surface','center_axis'):
        rotation=source[frame_name];offset=np.array([.15,0.,0.]);component=rotation_velocity(rotation,t,offset)
        for origin,base in [('fixed',fixed),('processed',processed),('marker',marker)]:
            vector=base+component
            differences=[]
            for epsilon in (1e-4,5e-5,2.5e-5):
                values=[]
                for moment in (t-epsilon,t+epsilon):
                    if origin=='fixed':w=fixed_wrist_position(curve(moment)[:3],source['lengths'])
                    elif origin=='processed':w=curve(moment)[2]
                    else:w=source['mid'](moment)
                    values.append(w+rotation(moment).as_matrix()@offset)
                differences.append(float(np.linalg.norm((values[1]-values[0])/(2*epsilon)-vector)))
            rows.append({'pitch':source['pitch'],'frame':frame_name,'wrist_origin':origin,
                         'assumed_ball_kmh':float(np.linalg.norm(vector)*3.6),
                         'wrist_speed_kmh':float(np.linalg.norm(base)*3.6),
                         'rotation_vector_norm_kmh':float(np.linalg.norm(component)*3.6),
                         'vector_mps':vector.tolist(),'position_derivative_difference_mps':differences,
                         'body_pose_changed':False,'grip_m':.15,'br_s':t})
    dot=float(source['surface'](t).as_matrix()[:,0]@source['center_axis'](t).as_matrix()[:,0])
    return rows,{'pitch':source['pitch'],'axis_angle_difference_deg':float(np.degrees(np.arccos(np.clip(dot,-1,1)))),
                 'hand_center_from_wrist_mm':float(np.linalg.norm(curve(t)[3]-curve(t)[2])*1000),
                 'fin_from_marker_wrist_mm':float(np.linalg.norm(source['fin'](t)-source['mid'](t))*1000),
                 'processed_to_marker_wrist_mm':float(np.linalg.norm(curve(t)[2]-source['mid'](t))*1000)}


def run(data, output):
    data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];geometry=[];sources=[];regression=[]
    for pitch in PITCHES:
        source=load_source(data,pitch);results,axis=evaluate_source(source);rows.extend(results);geometry.append(axis)
        legacy=prepare_case(data,pitch)['legacy_nominal_kmh'][1]
        baseline=next(r for r in results if r['frame']=='surface' and r['wrist_origin']=='fixed')
        error=abs(baseline['assumed_ball_kmh']-legacy)
        if error>1e-7:raise AssertionError('legacy_not_reproduced')
        regression.append({'pitch':pitch,'speed_difference_kmh':error});sources.append({'pitch':pitch,'path':source['c3d_path'],'sha256':source['c3d_sha256']})
    # Outcome is read only after geometry/velocity computations are frozen in memory.
    metadata=pd.read_csv(data/'metadata.csv');poi=pd.read_csv(data/'poi/poi_metrics.csv',usecols=['session_pitch','pitch_speed_mph'])
    match=metadata[metadata.session_pitch.isin(PITCHES)].merge(poi,on='session_pitch',validate='one_to_one',suffixes=('','_poi'))
    if len(match)!=5 or not np.allclose(match.pitch_speed_mph,match.pitch_speed_mph_poi,atol=1e-8,rtol=0):raise ValueError('radar_key_mismatch')
    for item in match.itertuples():
        token=Path(item.filename_new).stem.split('_')[-1]
        if not np.isclose(float(token)/10,item.pitch_speed_mph,atol=1e-8):raise ValueError('filename_speed_conflict')
    truth=dict(zip(match.session_pitch,match.pitch_speed_mph*1.609344))
    for row in rows:row['recorded_kmh']=float(truth[row['pitch']]);row['signed_difference_kmh']=row['assumed_ball_kmh']-row['recorded_kmh']
    frame=pd.DataFrame(rows);frame.to_csv(output/'paths.csv',index=False);pd.DataFrame(geometry).to_csv(output/'geometry.csv',index=False)
    summaries=[]
    for (definition,origin),group in frame.groupby(['frame','wrist_origin']):
        centered_pred=group.assumed_ball_kmh-group.assumed_ball_kmh.mean();centered_true=group.recorded_kmh-group.recorded_kmh.mean()
        summaries.append({'frame':definition,'wrist_origin':origin,'absolute_mae_kmh':float(abs(group.signed_difference_kmh).mean()),
                          'within_centered_mae_kmh':float(abs(centered_pred-centered_true).mean()),
                          'mean_signed_difference_kmh':float(group.signed_difference_kmh.mean())})
    report={'summary':summaries,'regression':regression,'sources':sources,
            'actual_ball_center_observed':False,'grip_release_force_fit_to_radar':False,
            'new_core_adopted':False,'same_ability_improvement_tested':False}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    package=Path(__file__).parent
    inputs=[data/'metadata.csv',data/'poi/poi_metrics.csv',data/'full_sig/landmarks.csv',data.parent/'code/v3d/model/v6_model_hybrid_lm.mdh',package/'PROTOCOL.md',Path(__file__)]
    (output/'manifest.json').write_text(json.dumps({'protocol_commit':'d565bfa','sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}},indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description='Original3034 large release-speed gap direct check')
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/results')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):parser.error('output must stay inside large_gap_repair')
    print(json.dumps(run(args.data,args.output)['summary'],indent=2))


if __name__=='__main__':main()
