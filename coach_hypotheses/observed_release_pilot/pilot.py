"""Same-session virtual wrist and held-frame hand/Ball surface-marker checks."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import ezc3d
import numpy as np
import pandas as pd

CASES = [('1w2d/S1_DP_05_O.c3d','1w2d/S1_static_01_D.c3d','R'),
         ('1w2d/S1_DP_06_O.c3d','1w2d/S1_static_01_D.c3d','R'),
         ('1w1d/S1_NDP_07_O.c3d','1w1d/S1_static_03_N.c3d','L')]


def load_c3d(path):
    """Preserve source frame numbers; ezc3d's frame convention differs by one here."""
    with Path(path).open('rb') as handle:
        header = handle.read(512)
        handle.seek((header[0]-1)*512+3)
        processor = handle.read(1)[0]
    if header[1] != 80 or processor != 84:
        raise ValueError('only_intel_c3d_supported')
    first, last = struct.unpack_from('<HH', header, 6)
    source = ezc3d.c3d(str(path))
    parameter = source['parameters']['POINT']
    rate = float(parameter['RATE']['value'][0])
    units = parameter['UNITS']['value'][0].strip().lower()
    scale = {'mm':.001,'cm':.01,'m':1.}.get(units)
    if scale is None or rate != 200.:
        raise ValueError('unexpected_units_or_rate')
    points = source['data']['points']
    if points.shape[2] != last-first+1:
        raise ValueError('source_frame_count_mismatch')
    markers, validity = {}, {}
    for i,name in enumerate(parameter['LABELS']['value']):
        values = points[:,i,:].T
        markers[name.strip()] = values[:,:3]*scale
        validity[name.strip()] = np.isfinite(values).all(axis=1)&(values[:,3]>=0)
    events = []
    if 'EVENT' in source['parameters']:
        event = source['parameters']['EVENT']
        for i,label in enumerate(event['LABELS']['value']):
            seconds = float(60*event['TIMES']['value'][0,i]+event['TIMES']['value'][1,i])
            events.append({'label':label.strip(),'time':seconds,'source_frame':int(round(seconds*rate)),
                           'array_index':int(round(seconds*rate))-first})
    return {'markers':markers,'valid':validity,'rate':rate,'first':first,'last':last,
            'time':(first+np.arange(points.shape[2]))/rate,'events':events,
            'ezc3d_first':source['header']['points']['first_frame']}


def rigid_transform(template, observed):
    template,observed = np.asarray(template,float),np.asarray(observed,float)
    if template.shape != observed.shape or template.ndim!=2 or template.shape[1] != 3 or not np.isfinite(observed).all() or not np.isfinite(template).all():
        raise ValueError('invalid_cluster_points')
    a,b = template.mean(axis=0),observed.mean(axis=0)
    if np.linalg.matrix_rank(template-a) < 2 or np.linalg.matrix_rank(observed-b) < 2:
        raise ValueError('degenerate_cluster')
    u,_,vt = np.linalg.svd((template-a).T@(observed-b))
    correction = np.eye(3);correction[-1,-1] = np.linalg.det(vt.T@u.T)
    rotation = vt.T@correction@u.T
    translation = b-rotation@a
    residual = np.sqrt(np.mean(np.sum(((rotation@template.T).T+translation-observed)**2,axis=1)))
    return rotation,translation,float(residual)


def calibrate_wrist(static, side):
    cluster = [side+f'FR{i}' for i in range(1,5)]
    styloids = [side+'WRL',side+'WRM']
    markers = static['markers'];n = len(static['time']);split = n//2
    if split < 10 or not all(static['valid'][label].all() for label in cluster+styloids):
        raise ValueError('incomplete_static_calibration')
    points = np.stack([markers[label] for label in cluster],axis=1)
    wrist = (markers[styloids[0]]+markers[styloids[1]])/2
    template = points[:split].mean(axis=0)
    local = []
    for i in range(split):
        rotation,translation,_ = rigid_transform(template,points[i])
        local.append(rotation.T@(wrist[i]-translation))
    offset = np.mean(local,axis=0)
    errors = []
    for i in range(split,n):
        rotation,translation,_ = rigid_transform(template,points[i])
        errors.append(np.linalg.norm(rotation@offset+translation-wrist[i]))
    return {'template':template,'wrist_local':offset,'fit_frames':split,'held_frames':n-split,
            'held_wrist_rmse_mm':float(np.sqrt(np.mean(np.square(errors)))*1000)}


def dynamic_hand(source, calibration, side):
    markers = source['markers'];n = len(source['time'])
    required = [side+f'FR{i}' for i in range(1,5)]+[side+'3MC',side+'5MC']
    valid = np.logical_and.reduce([source['valid'][label] for label in required])
    wrist = np.full((n,3),np.nan);rotation = np.full((n,3,3),np.nan);residual = np.full(n,np.nan)
    for i in np.flatnonzero(valid):
        observed = np.array([markers[label][i] for label in required[:4]])
        r,t,residual[i] = rigid_transform(calibration['template'],observed)
        wrist[i] = r@calibration['wrist_local']+t
        x = markers[side+'3MC'][i]-wrist[i]
        transverse = markers[side+'5MC'][i]-markers[side+'3MC'][i]
        if np.linalg.norm(x)<1e-8 or np.linalg.norm(np.cross(x,transverse))<1e-10:
            valid[i]=False;continue
        x = x/np.linalg.norm(x)
        z = np.cross(x,transverse);z = z/np.linalg.norm(z)
        y = np.cross(z,x)
        rotation[i] = np.column_stack([x,y,z])
    return wrist,rotation,valid,residual


def fit_offset(time, wrist, rotations, ball, indices, linear):
    selected = np.asarray(indices,int)
    relative = np.einsum('nji,nj->ni',rotations[selected],ball[selected]-wrist[selected])
    design = np.column_stack([np.ones(len(selected)),time[selected]]) if linear else np.ones((len(selected),1))
    coefficients = np.linalg.lstsq(design,relative,rcond=None)[0]
    return coefficients if linear else np.vstack([coefficients,np.zeros((1,3))])


def predict_marker(time, wrist, rotations, coefficients):
    relative = coefficients[0]+np.asarray(time)[:,None]*coefficients[1]
    return wrist+np.einsum('nij,nj->ni',rotations,relative)


def evaluate(source, calibration, side, event_shift=0):
    events = [event for event in source['events'] if event['label']=='Event']
    if len(events)!=1:
        raise ValueError('ambiguous_candidate_event')
    event = events[0];rate = source['rate']
    time = source['time']-event['time']-event_shift/rate
    train = np.flatnonzero((time>=-.09-1e-6)&(time<=-.045+1e-6))
    test = np.flatnonzero((time>=-.04-1e-6)&(time<=-.005+1e-6))
    if len(train)!=10 or len(test)!=8:
        raise ValueError('insufficient_event_window')
    if test[0]<1 or test[-1]+1>=len(time):
        raise ValueError('derivative_outside_support')
    support = np.arange(train[0],test[-1]+2)
    wrist,rotations,valid,residual = dynamic_hand(source,calibration,side)
    ball = source['markers']['Ball']
    if not (valid[support]&source['valid']['Ball'][support]).all():
        raise ValueError('missing_observations_no_imputation')
    observed_velocity = (ball[test+1]-ball[test-1])*rate/2
    results=[];prediction_rows=[]
    for linear,label in [(False,'rigid_offset'),(True,'linear_relative_offset')]:
        coefficient = fit_offset(time,wrist,rotations,ball,train,linear)
        predicted = predict_marker(time[support],wrist[support],rotations[support],coefficient)
        lookup = {index:point for index,point in zip(support,predicted)}
        p = np.array([lookup[i] for i in test])
        velocity = np.array([(lookup[i+1]-lookup[i-1])*rate/2 for i in test])
        error = p-ball[test]
        results.append({'event_shift_frames':event_shift,'model':label,'train_samples':len(train),'test_samples':len(test),
                        'marker_position_rmse_mm':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))*1000),
                        'marker_speed_mae_kmh':float(np.mean(np.abs(np.linalg.norm(velocity,axis=1)-np.linalg.norm(observed_velocity,axis=1)))*3.6),
                        'marker_velocity_vector_rmse_kmh':float(np.sqrt(np.mean(np.sum((velocity-observed_velocity)**2,axis=1)))*3.6),
                        'candidate_event_array_index':event['array_index'],
                        'local_offset_at_event_mm':(coefficient[0]*1000).tolist(),
                        'fitted_local_relative_velocity_mps':coefficient[1].tolist(),
                        'held_forearm_marker_fit_rmse_mm':float(np.sqrt(np.mean(residual[test]**2))*1000),
                        'held_static_wrist_rmse_mm':calibration['held_wrist_rmse_mm']})
        for index,point,truth in zip(test,p,ball[test]):
            prediction_rows.append({'array_index':int(index),'relative_time_ms':float(time[index]*1000),
                                    'model':label,'event_shift_frames':event_shift,
                                    **{f'predicted_{a}_m':float(v) for a,v in zip('xyz',point)},
                                    **{f'observed_{a}_m':float(v) for a,v in zip('xyz',truth)}})
    return results,prediction_rows


def run(data, output):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    rows,predictions,exclusions,inputs=[],[],[],{}
    for dynamic,static,side in CASES:
        for name in (dynamic,static):
            inputs[name]=hashlib.sha256((data/name).read_bytes()).hexdigest()
        source=load_c3d(data/dynamic)
        try:
            calibration=calibrate_wrist(load_c3d(data/static),side)
            for shift in (0,1):
                metrics,points=evaluate(source,calibration,side,shift)
                rows.extend(dict(row,trial=dynamic,side=side) for row in metrics)
                predictions.extend(dict(row,trial=dynamic) for row in points)
        except ValueError as error:
            exclusions.append({'trial':dynamic,'reason':str(error)})
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
    summary={'inputs_sha256':inputs,'exclusions':exclusions,'rows':rows,
             'ball_marker_is_confirmed_center':False,'actual_release_time_validated':False,
             'radar_comparison_performed':False,'athlete_performance_effect_tested':False}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    package=Path(__file__).parent
    (output/'manifest.json').write_text(json.dumps({'protocol_commit':'6a3c38d',
        'protocol_sha256':hashlib.sha256((package/'PROTOCOL.md').read_bytes()).hexdigest(),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')}},indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description='Independent hand/Ball surface-marker pilot')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/results')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must stay inside observed_release_pilot')
    summary=run(args.data,args.output)
    print(json.dumps({'model_rows':len(summary['rows']),'exclusions':summary['exclusions']},ensure_ascii=False))


if __name__=='__main__':main()
