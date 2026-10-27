"""Reproduce hand-relative launch correction and audit uncertain switch times."""
import json
import hashlib
from pathlib import Path
import numpy as np
import ezc3d
from physical_release_comparison import launch_states, free_flight
from heldout_speed_comparison import evaluate

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = ROOT/'artifacts/multiview-release'


def camera_models():
    directory=LAB/'openbiomechanics/computer_vision/calibration/results'
    intr=json.loads((directory/'calibration_results.json').read_text())['calibrations']
    rig=json.loads((directory/'optitrack_rig_provisional.json').read_text())['camera_poses']
    result={}
    for cam in [19,20,21,22]:
        k=intr[f'optitrack_{cam}'];r=rig[str(cam)]
        if np.any(k['distortion']): raise ValueError('Distortion must be handled explicitly')
        result[cam]=(np.array([[k['fx_px'],0,k['cx_px']],[0,k['fy_px'],k['cy_px']],[0,0,1.]]),
                     np.array(r['reference_to_camera_rotation_matrix']),np.array(r['reference_to_camera_translation_mm'])/1000)
    return result


def project(points, camera):
    k,r,t=camera
    p=np.atleast_2d(points)@r.T+t
    if np.any(p[:,2]<=0):raise ValueError('Nonpositive depth')
    q=p@k.T
    return q[:,:2]/q[:,2,None]


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    source=LAB/'hand-ball-link/artifacts/hand-ball/observation.json'
    observed=json.loads(source.read_text())
    alignment_path=Path(observed['source_sha256']['alignment']['path'])
    body_path=Path(observed['source_sha256']['body']['path'])
    alignment=json.loads(alignment_path.read_text());rotation=np.array(alignment['rotation'])
    c=ezc3d.c3d(str(body_path));labels=c['parameters']['ROTATION']['LABELS']['value']
    hand=c['data']['rotations'][:3,:3,labels.index('l_hand_4X4'),:]
    gravity=rotation@np.array([0,0,-9.81]);rate=float(observed['rate_hz'])
    later_path=LAB/'ball-trajectory-fit/artifacts/trajectory-fit/trajectory_fit.json'
    later=json.loads(later_path.read_text());cams=camera_models()
    detections={}
    for row in later['detections']:
        if row['camera'] in [19,22] and row['frame'] in [980,981,985] and row['mode']=='local_ellipse':
            detections[(row['camera'],row['frame'],row['threshold'])]=row
    rows=[]
    for condition in observed['conditions']:
        frames=np.arange(973,980)
        ball=np.array([p['ball_rig_m'] for p in condition['positions']])
        origin=np.array([p['hand_origin_rig_m'] for p in condition['positions']])
        rotations=np.einsum('ij,jkn->nik',rotation,hand[:,:,frames+condition['c3d_frame_offset']])
        for switch in [974,975,976]:
            count=switch-973+1
            states=launch_states((frames[:count]-973)/rate,ball[:count],origin[:count],rotations[:count])
            models={}
            for name,(pos,vel) in states.items():
                trajectory,velocities=free_flight(pos,vel,gravity,1/rate,2,985-switch)
                near=trajectory[np.arange(977,980)-switch-1]
                metrics=evaluate(near,ball[4:],1/rate,62.7*1.609344)
                checks=[]
                for det in detections.values():
                    err=None
                    if det['status']=='candidate':
                        err=float(np.linalg.norm(project(trajectory[det['frame']-switch-1],cams[det['camera']])[0]-det['pixel']))
                    checks.append(dict(camera=det['camera'],frame=det['frame'],threshold=det['threshold'],error_px=err))
                metrics.update(launch_speed_kmh=float(np.linalg.norm(vel)*3.6),
                               launch_kinetic_energy_j=float(.5*.145*(vel@vel)),
                               late_camera_rmse_px={str(cam):float(np.sqrt(np.mean([d['error_px']**2 for d in checks if d['camera']==cam and d['error_px'] is not None]))) for cam in [19,22]},
                               trajectory_frames=list(range(switch+1,986)),trajectory_m=trajectory.tolist(),late_checks=checks)
                models[name]=metrics
            rows.append(dict(threshold=condition['threshold'],offset=condition['c3d_frame_offset'],switch_frame=switch,models=models))
    nominal=next(r for r in rows if (r['threshold'],r['offset'],r['switch_frame'])==(80,1,976))
    report=dict(rows=rows,nominal=nominal,rate_hz=rate,
        same_evaluation_frames=[977,978,979], late_frames=[980,981,985],
        visually_separated_by_frame=978,physical_release_identified=False,
        scope='One previously inspected baseball throw. A fixed hand-relative point, B linear hand-relative motion, C ball-only. Same training samples. Model switch 974/975/976 is sensitivity, not identified contact loss. No future hand injection. Actual MuJoCo gravity-only free flight; no muscles/contact forces validated. Not OBP3034.',
        video_hand_relation='Theia hand transforms and video alignment are estimates with unresolved synchronization/provenance.',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,alignment_path,body_path,later_path]})
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for row in rows:
        if row['threshold']==80 and row['offset']==1:
            print(row['switch_frame'],{k:round(v['mean_absolute_speed_discrepancy_kmh'],4) for k,v in row['models'].items()})
    print('nominal',json.dumps({k:{n:v[n] for n in ['mean_predicted_speed_kmh','mean_absolute_speed_discrepancy_kmh','position_rmse_mm','late_camera_rmse_px','launch_kinetic_energy_j']} for k,v in nominal['models'].items()}))
    return report


if __name__=='__main__':run()
