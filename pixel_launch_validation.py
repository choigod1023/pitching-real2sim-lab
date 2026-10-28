"""Train-only image-space launch fitting; no measured-speed target."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from physical_release_comparison import free_flight
from heldout_speed_comparison import evaluate

ROOT=Path(__file__).resolve().parent

def trajectory(state,times,gravity):
    t=np.asarray(times)[:,None]
    return state[:3]+t*state[3:]+.5*t*t*np.asarray(gravity)

def project(points,camera):
    k,r,t=camera
    xyz=np.asarray(points)@r.T+t
    if not np.isfinite(xyz).all() or np.any(xyz[:,2]<=0):raise ValueError('Invalid camera depth')
    uv=xyz@k.T
    return uv[:,:2]/uv[:,2:]

def fit(times,points,pixels,cameras,gravity,mode):
    times=np.asarray(times,float);points=np.asarray(points,float);pixels=np.asarray(pixels,float)
    if len(times)<3 or np.any(np.diff(times)<=0) or points.shape!=(len(times),3) or pixels.shape!=(len(cameras),len(times),2):raise ValueError('Invalid training dimensions/time')
    if not all(np.isfinite(a).all() for a in (times,points,pixels)):raise ValueError('Nonfinite training data')
    if mode not in ('C','G','P','R'):raise ValueError('Unknown mode')
    design=np.column_stack([np.ones(len(times)),times])
    corrected=points if mode=='C' else points-.5*times[:,None]**2*np.asarray(gravity)
    state=np.linalg.lstsq(design,corrected,rcond=None)[0].ravel()
    if mode in ('P','R'):
        def residual(x):
            p=trajectory(x,times,gravity)
            return np.concatenate([(project(p,c)-uv).ravel() for c,uv in zip(cameras,pixels)])
        result=least_squares(residual,state,loss='linear' if mode=='P' else 'soft_l1',f_scale=1.,x_scale='jac',max_nfev=1000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        if not result.success:raise RuntimeError(result.message)
        state=result.x
    return state

def run():
    lab=ROOT.parent
    paths=dict(candidates=lab/'speed-journey/artifacts/release-observation/candidate_audit.json',
        heldout=lab/'hand-ball-link/artifacts/heldout-ball/report.json',
        intrinsics=lab/'openbiomechanics/computer_vision/calibration/results/calibration_results.json',
        rig=lab/'openbiomechanics/computer_vision/calibration/results/optitrack_rig_provisional.json',
        alignment=lab/'openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json')
    data={k:json.loads(p.read_text()) for k,p in paths.items()}
    cameras={}
    for n in (19,20,21):
        a=data['intrinsics']['calibrations'][f'optitrack_{n}'];b=data['rig']['camera_poses'][str(n)]
        if any(a['distortion']):raise ValueError('Unsupported distortion')
        cameras[n]=(np.array([[a['fx_px'],0,a['cx_px']],[0,a['fy_px'],a['cy_px']],[0,0,1.]]),np.array(b['reference_to_camera_rotation_matrix']),np.array(b['reference_to_camera_translation_mm'])/1000)
    gravity=np.array(data['alignment']['rotation'])@np.array([0,0,-9.81])
    times=(np.arange(973,977)-976)/360
    results=[]
    for threshold in (60,80,100):
        tri={r['frame']:np.array(r['position_camera19_mm'])/1000 for r in data['candidates']['conditional_triangulation'] if r['threshold']==threshold}
        det={(r['camera'],r['frame']):r for r in data['candidates']['detections'] if r['threshold']==threshold}
        points=np.array([tri[f] for f in range(973,977)])
        pixels=np.array([[det[c,f]['pixel'] for f in range(973,977)] for c in (20,21)])
        for mode in ('C','G','P','R'):
            state=fit(times,points,pixels,[cameras[20],cameras[21]],gravity,mode)
            predicted,v=free_flight(state[:3],state[3:],gravity,substeps=2)
            analytical=trajectory(state,np.arange(1,4)/360,gravity)
            assert np.max(np.abs(predicted-analytical))<1e-10
            metrics=evaluate(predicted,np.array([tri[f] for f in (977,978,979)]),1/360,62.7*1.609344)
            training=np.concatenate([(project(trajectory(state,times,gravity if mode!='C' else np.zeros(3)),cameras[c])-pixels[j]) for j,c in enumerate((20,21))])
            future=np.concatenate([project(predicted,cameras[c])-np.array([det[c,f]['pixel'] for f in (977,978,979)]) for c in (20,21)])
            checks=[]
            for d in data['heldout']['detections']:
                uv=project(predicted[[d['frame']-977]],cameras[19])[0]
                checks.append(dict(frame=d['frame'],threshold=d['threshold'],status=d['status'],error_px=float(np.linalg.norm(uv-d['pixel'])) if d['status']=='candidate' else None))
            errors=[d['error_px'] for d in checks if d['error_px'] is not None]
            results.append(dict(threshold=threshold,mode=mode,state=state.tolist(),metrics=metrics,
                training_pixel_rmse=float(np.sqrt(np.mean(np.sum(training**2,axis=1)))),
                future_camera20_21_pixel_rmse=float(np.sqrt(np.mean(np.sum(future**2,axis=1)))),
                heldout_camera19_pixel_rmse=float(np.sqrt(np.mean(np.square(errors)))),heldout_checks=checks))
    out=ROOT/'artifacts/pixel-launch';out.mkdir(parents=True,exist_ok=True)
    report=dict(training_frames=list(range(973,977)),evaluation_frames=[977,978,979],gravity=gravity.tolist(),
        sources={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
        protocol_sha256=hashlib.sha256((ROOT/'PIXEL_LAUNCH_PROTOCOL.md').read_bytes()).hexdigest(),results=results)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    for r in results:print(r['threshold'],r['mode'],*[round(x,6) for x in [r['metrics']['mean_absolute_speed_discrepancy_kmh'],r['metrics']['mean_absolute_radar_scalar_discrepancy_kmh'],r['future_camera20_21_pixel_rmse'],r['heldout_camera19_pixel_rmse']]])
    return report

if __name__=='__main__':run()
