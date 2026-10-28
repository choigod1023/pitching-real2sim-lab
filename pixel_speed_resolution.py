"""Fixed pixel perturbation stress; not a measurement confidence interval."""
import json
import hashlib
from pathlib import Path
import numpy as np
from pixel_launch_validation import fit,project,trajectory

ROOT=Path(__file__).resolve().parent

def run():
    lab=ROOT.parent
    raw=json.loads((lab/'speed-journey/artifacts/release-observation/candidate_audit.json').read_text())
    cal=lab/'openbiomechanics/computer_vision/calibration/results'
    intr=json.loads((cal/'calibration_results.json').read_text())['calibrations']
    rig=json.loads((cal/'optitrack_rig_provisional.json').read_text())['camera_poses']
    align=json.loads((lab/'openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json').read_text())
    gravity=np.array(align['rotation'])@np.array([0,0,-9.81])
    cameras=[]
    for n in (20,21):
        k=intr[f'optitrack_{n}'];r=rig[str(n)]
        if any(k['distortion']):raise ValueError('Nonzero distortion')
        cameras.append((np.array([[k['fx_px'],0,k['cx_px']],[0,k['fy_px'],k['cy_px']],[0,0,1.]]),np.array(r['reference_to_camera_rotation_matrix']),np.array(r['reference_to_camera_translation_mm'])/1000))
    frames=[977,978,979];times=(np.array(frames)-979)/360
    tri={r['frame']:np.array(r['position_camera19_mm'])/1000 for r in raw['conditional_triangulation'] if r['threshold']==80}
    det={(r['camera'],r['frame']):r for r in raw['detections'] if r['threshold']==80}
    points=np.array([tri[f] for f in frames]);uv=np.array([[det[c,f]['pixel'] for f in frames] for c in (20,21)])
    baseline=fit(times,points,uv,cameras,gravity,'P');base_speed=float(np.linalg.norm(baseline[3:])*3.6)
    # Matched seeded perturbations across amplitudes; no fitting to radar or held-out views.
    noise=np.random.default_rng(20261026).uniform(-1,1,(100,*uv.shape))
    rows=[]
    for amplitude in (.25,.5,1.):
        speeds=[]
        for perturbation in noise:
            state=fit(times,points,uv+amplitude*perturbation,cameras,gravity,'P')
            speeds.append(float(np.linalg.norm(state[3:])*3.6))
        delta=np.array(speeds)-base_speed
        rows.append(dict(amplitude_px=amplitude,n=100,min_delta_kmh=float(delta.min()),max_delta_kmh=float(delta.max()),rms_delta_kmh=float(np.sqrt(np.mean(delta**2))),speeds_kmh=speeds))
    # +/-1 km/h along baseline velocity, holding terminal position fixed.
    norm=np.linalg.norm(baseline[3:]);direction=baseline[3:]/norm
    image_effect=[]
    for change in (-1,1):
        state=baseline.copy();state[3:]+=change/3.6*direction
        shifts=np.array([project(trajectory(state,times,gravity),c)-project(trajectory(baseline,times,gravity),c) for c in cameras])
        image_effect.append(dict(speed_change_kmh=change,max_pixel_shift=float(np.linalg.norm(shifts,axis=-1).max())))
    report=dict(scope='Engineering sensitivity only: independent uniform pixel perturbations, not measured noise or confidence bounds. No correction selected.',seed=20261026,frames=frames,threshold=80,baseline_speed_kmh=base_speed,stress=rows,one_kmh_image_effect=image_effect)
    sources=[lab/'speed-journey/artifacts/release-observation/candidate_audit.json',cal/'calibration_results.json',cal/'optitrack_rig_provisional.json',lab/'openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json',Path(__file__)]
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out=ROOT/'artifacts/pixel-resolution';out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('stress','source_sha256')}))
    for r in rows:print({k:v for k,v in r.items() if k!='speeds_kmh'})

if __name__=='__main__':run()
