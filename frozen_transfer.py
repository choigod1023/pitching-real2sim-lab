"""Train-only hand/relative-motion launch derivative and held-out free flight."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from physical_release_comparison import launch_states,free_flight
from audit_prime_vectors import compare_vectors
from pixel_launch_validation import project


def fit_chain(times,ball,origin,rotation):
    times,ball,origin,rotation=map(lambda x:np.asarray(x,float),(times,ball,origin,rotation))
    n=len(times)
    if n<3 or ball.shape!=(n,3) or origin.shape!=(n,3) or rotation.shape!=(n,3,3):
        raise ValueError('At least three aligned training frames required')
    if not all(np.isfinite(x).all() for x in (times,ball,origin,rotation)) or np.any(np.diff(times)<=0):
        raise ValueError('Finite increasing training times required')
    if not np.allclose(rotation.transpose(0,2,1)@rotation,np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(rotation),1):
        raise ValueError('Proper rotations required')
    t=times-times[0];design=np.c_[np.ones(n),t]
    r0=Rotation.from_matrix(rotation[0]);log=(r0.inv()*Rotation.from_matrix(rotation)).as_rotvec()
    # Angular motion anchored at first observed orientation; no future samples.
    omega=np.sum(t[:,None]*log,axis=0)/np.dot(t,t)
    o=np.linalg.lstsq(design,origin,rcond=None)[0]
    relative=np.einsum('nji,nj->ni',rotation,ball-origin)
    r=np.linalg.lstsq(design,relative,rcond=None)[0]
    return dict(t0=float(times[0]),r0=rotation[0].copy(),omega=omega,origin=o,relative=r)


def chain_state(model,time):
    t=time-model['t0'];rot=model['r0']@Rotation.from_rotvec(model['omega']*t).as_matrix()
    r=model['relative'][0]+t*model['relative'][1]
    p=model['origin'][0]+t*model['origin'][1]+rot@r
    v=model['origin'][1]+rot@(np.cross(model['omega'],r)+model['relative'][1])
    return p,v


def run():
    import ezc3d
    root=Path(__file__).resolve().parent;lab=root.parent
    observation=lab/'hand-ball-link/artifacts/hand-ball/observation.json'
    later_path=lab/'ball-trajectory-fit/artifacts/trajectory-fit/trajectory_fit.json'
    observed=json.loads(observation.read_text());later=json.loads(later_path.read_text())
    body=Path(observed['source_sha256']['body']['path']);alignment_path=Path(observed['source_sha256']['alignment']['path'])
    alignment=json.loads(alignment_path.read_text());q=np.array(alignment['rotation']);gravity=q@np.array([0,0,-9.81])
    c=ezc3d.c3d(str(body));labels=c['parameters']['ROTATION']['LABELS']['value']
    hand=c['data']['rotations'][:3,:3,labels.index('l_hand_4X4'),:]
    cal=lab/'openbiomechanics/computer_vision/calibration/results'
    intr_path=cal/'calibration_results.json';rig_path=cal/'optitrack_rig_provisional.json'
    intr=json.loads(intr_path.read_text())['calibrations'];rig=json.loads(rig_path.read_text())['camera_poses']
    cameras={}
    for camera in (19,22):
        k=intr[f'optitrack_{camera}'];e=rig[str(camera)]
        if any(k['distortion']):raise ValueError('Unsupported camera distortion')
        cameras[camera]=(np.array([[k['fx_px'],0,k['cx_px']],[0,k['fy_px'],k['cy_px']],[0,0,1.]]),np.array(e['reference_to_camera_rotation_matrix']),np.array(e['reference_to_camera_translation_mm'])/1000)
    unique={}
    for det in later['detections']:
        if det['camera'] in cameras and det['frame'] in (980,981,985):
            key=(det['camera'],det['frame'],det['threshold'],det['mode'])
            if key in unique and unique[key]!=det:raise ValueError('Conflicting duplicated observation')
            unique[key]=det
    rows=[];rate=float(observed['rate_hz']);frames=np.arange(973,977);times=(frames-973)/rate
    for condition in observed['conditions']:
        ball=np.array([p['ball_rig_m'] for p in condition['positions']])
        origin=np.array([p['hand_origin_rig_m'] for p in condition['positions']])
        rotation=np.einsum('ij,jkn->nik',q,hand[:,:,frames+condition['c3d_frame_offset']])
        states=launch_states(times,ball[:4],origin[:4],rotation)
        chain=fit_chain(times,ball[:4],origin[:4],rotation);states['D']=chain_state(chain,times[-1])
        metrics={}
        for name,(position,velocity) in states.items():
            future,_=free_flight(position,velocity,gravity,1/rate,2,9)
            near=compare_vectors(future[:3],ball[4:],1/rate)
            checks=[]
            for det in unique.values():
                pred=project(future[[det['frame']-977]],cameras[det['camera']])[0]
                error=float(np.linalg.norm(pred-det['pixel'])) if det['status']=='candidate' else None
                checks.append(dict(camera=det['camera'],frame=det['frame'],threshold=det['threshold'],mode=det['mode'],status=det['status'],error_px=error))
            summaries={}
            for camera in cameras:
                for mode in ('component','local_ellipse'):
                    selected=[d for d in checks if d['camera']==camera and d['mode']==mode]
                    errors=[d['error_px'] for d in selected if d['error_px'] is not None]
                    summaries[f'{camera}_{mode}']=dict(n=len(errors),failed=len(selected)-len(errors),rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
            metrics[name]=dict(near=near,late=summaries,checks=checks,launch_position_m=position.tolist(),launch_velocity_m_s=velocity.tolist(),future_positions_m=future.tolist(),
                near_position_rmse_mm=float(np.sqrt(np.mean(np.sum((future[:3]-ball[4:])**2,axis=1)))*1000))
        rows.append(dict(threshold=condition['threshold'],offset=condition['c3d_frame_offset'],models=metrics))
    def beats(a,b):
        return all(a[k]['rmse_px'] is not None and b[k]['rmse_px'] is not None and a[k]['rmse_px']<b[k]['rmse_px'] for k in a)
    nominal=next(r for r in rows if r['threshold']==80 and r['offset']==1)['models']
    wins=sum(beats(r['models']['D']['late'],r['models']['B']['late']) for r in rows)
    gates=dict(nominal_late_beats_B=beats(nominal['D']['late'],nominal['B']['late']),nominal_late_beats_C=beats(nominal['D']['late'],nominal['C']['late']),
        robust_late_beats_B=wins>=7,near_speed_better=nominal['D']['near']['speed_mae_kmh']<nominal['B']['near']['speed_mae_kmh'],
        near_direction_better=nominal['D']['near']['mean_direction_error_deg']<nominal['B']['near']['mean_direction_error_deg'])
    result=dict(protocol_sha256=hashlib.sha256((root/'FROZEN_TRANSFER_PROTOCOL.md').read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [observation,later_path,body,alignment_path,intr_path,rig_path]},
        train_frames=frames.tolist(),evaluation_frames=list(range(977,986)),rate_hz=rate,gravity=gravity.tolist(),rows=rows,gates=gates,
        retained_for_further_validation=all(gates.values()),robust_wins_of_9=wins,human_intervention_validated=False,
        scope='Previously inspected single development pitch; future hands excluded from ALL models. Actual observed training ball required. Optical candidates are not ground truth.')
    out=root/'artifacts/frozen-transfer';out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(gates=gates,wins=wins,nominal={k:dict(near=v['near'],late=v['late']) for k,v in nominal.items()}),indent=2))
    return result


if __name__=='__main__':run()
