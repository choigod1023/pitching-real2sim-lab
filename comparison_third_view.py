"""Fit-excluded third-view check; no detector/calibration or model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ball_holdout_audit import project

ROOT=Path(__file__).resolve().parent

def run(stage="kinematic"):
    if stage not in ("kinematic", "physical"): raise ValueError("Unknown stage")
    paths={
        'predictions':ROOT/('artifacts/physical-comparison/comparison.json' if stage=='physical' else 'artifacts/speed-comparison/comparison.json'),
        'heldout_detections':ROOT.parent/'hand-ball-link/artifacts/heldout-ball/report.json',
        'intrinsics':ROOT.parent/'openbiomechanics/computer_vision/calibration/results/calibration_results.json',
        'rig':ROOT.parent/'openbiomechanics/computer_vision/calibration/results/optitrack_rig_provisional.json',
        'trial_source':ROOT.parent/'openbiomechanics/computer_vision/4d_reconstruction/README.md',
    }
    data=json.loads(paths['predictions'].read_text()); heldout=json.loads(paths['heldout_detections'].read_text())
    intr=json.loads(paths['intrinsics'].read_text())['calibrations']['optitrack_19']
    pose=json.loads(paths['rig'].read_text())['camera_poses']['19']
    if any(intr['distortion']): raise ValueError('Need distortion-aware projection')
    K=np.array([[intr['fx_px'],0,intr['cx_px']],[0,intr['fy_px'],intr['cy_px']],[0,0,1.]])
    P=np.column_stack([pose['reference_to_camera_rotation_matrix'],pose['reference_to_camera_translation_mm']])
    assert data['evaluation_frames']==[977,978,979]
    rows=[]; summaries=[]
    for condition in data['conditions']:
        for name,metrics in condition['metrics'].items():
            for frame,point in zip(data['evaluation_frames'],metrics['predicted_positions_m']):
                pixel=project(np.array(point)*1000,K,P)
                for det in heldout['detections']:
                    if det['frame']!=frame: continue
                    item=dict(source_threshold=condition['threshold'],offset=condition['offset'],model=name,frame=frame,
                              holdout_threshold=det['threshold'],status=det['status'],projected_pixel=pixel.tolist(),heldout_pixel=det['pixel'])
                    item['residual_norm_px']=float(np.linalg.norm(pixel-np.asarray(det['pixel']))) if det['status']=='candidate' else None
                    rows.append(item)
            for threshold in [100,130,160]:
                selected=[r for r in rows if r['source_threshold']==condition['threshold'] and r['offset']==condition['offset'] and r['model']==name and r['holdout_threshold']==threshold]
                values=[r['residual_norm_px'] for r in selected if r['residual_norm_px'] is not None]
                summaries.append(dict(source_threshold=condition['threshold'],offset=condition['offset'],model=name,holdout_threshold=threshold,
                                      count=len(values),failures=len(selected)-len(values),mean_residual_px=float(np.mean(values)) if values else None,
                                      rms_residual_px=float(np.sqrt(np.mean(np.square(values)))) if values else None))
    result=dict(stage=stage,scope='Fit-excluded image checks from camera19; shared provisional calibration and prior development inspection prevent independent ground-truth accuracy claims.',
                units='3D prediction meters converted to camera19 millimeters before projection; residuals pixels.',
                sources_sha256={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},
                original_detections=heldout['detections'],comparisons=rows,summaries=summaries)
    out=ROOT/'artifacts/comparison-validation';out.mkdir(parents=True,exist_ok=True)
    (out/('physical_third_view.json' if stage=='physical' else 'third_view.json')).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for s in summaries:
        if s['source_threshold']==80 and s['offset']==1: print(json.dumps(s))
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['kinematic','physical'],default='kinematic')
    run(parser.parse_args().stage)
