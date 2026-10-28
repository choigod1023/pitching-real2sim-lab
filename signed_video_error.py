"""Signed optical discrepancies, never a multi-pitch accuracy interval."""
import hashlib
import json
from pathlib import Path
import numpy as np
import multiview_release_validation as validation

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
OUT=ROOT/'artifacts/signed-video-error'


def describe(values):
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('At least two finite scalar discrepancies required')
    return dict(n=len(x),mean_signed=float(x.mean()),sample_sd=float(x.std(ddof=1)),
                min_signed=float(x.min()),max_signed=float(x.max()),
                mae=float(np.abs(x).mean()),max_absolute=float(np.abs(x).max()),
                interval_interpretation='Descriptive processing spread; not confidence or population accuracy interval')


def linear_speed(frames,positions):
    times=np.asarray(frames,float)/360
    a=np.column_stack([np.ones(len(times)),times-times.mean()])
    coefficient=np.linalg.lstsq(a,np.asarray(positions),rcond=None)[0]
    return float(np.linalg.norm(coefficient[1])*3.6)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    validation.OUT=OUT/'rollout'
    report=validation.run()
    near={}
    nominal_near={}
    for model in ['A','B','C']:
        differences=[]
        for row in report['rows']:
            m=row['models'][model]
            errors=np.array(m['predicted_interval_speed_kmh'])-m['candidate_interval_speed_kmh']
            differences.append(float(errors.mean()))
        near[model]=describe(differences)
        m=report['nominal']['models'][model]
        nominal_near[model]=dict(signed_interval_errors_kmh=(np.array(m['predicted_interval_speed_kmh'])-m['candidate_interval_speed_kmh']).tolist(),
                                 mean_signed_kmh=m['mean_predicted_speed_kmh']-m['mean_candidate_speed_kmh'])
    later_path=LAB/'ball-trajectory-fit/artifacts/trajectory-fit/trajectory_fit.json'
    later=json.loads(later_path.read_text());references=[];failed=[]
    frames=[980,981,985]
    for case in later['cases']:
        xyz=case.get('primary_positions_camera19_mm')
        if xyz is None and case.get('positions_camera19_mm') is not None:
            xyz=np.array(case['positions_camera19_mm'])[[0,1,5]]
        if xyz is None:
            failed.append(dict(mode=case['mode'],thresholds=case['thresholds'],reason='No complete three-frame optical reference'))
            continue
        references.append(dict(mode=case['mode'],thresholds=case['thresholds'],speed_kmh=linear_speed(frames,np.array(xyz)/1000)))
    primary=next(x for x in references if x['mode']=='local_ellipse' and x['thresholds']==[130,90])
    late={}; late_nominal={}
    for model in ['A','B','C']:
        errors=[]
        for row in report['rows']:
            m=row['models'][model]
            p=np.array(m['trajectory_m'])[[m['trajectory_frames'].index(f) for f in frames]]
            errors.append(linear_speed(frames,p)-primary['speed_kmh'])
        late[model]=describe(errors)
        m=report['nominal']['models'][model]
        p=np.array(m['trajectory_m'])[[m['trajectory_frames'].index(f) for f in frames]]
        speed=linear_speed(frames,p)
        late_nominal[model]=dict(prediction_kmh=speed,primary_signed_error_kmh=speed-primary['speed_kmh'],
                                reference_processing_spread=describe([speed-r['speed_kmh'] for r in references]))
    videos=[]
    for cam in [15,19,20,21,22]:
        folder=LAB/('ball-coordinate-audit/artifacts/video-audit' if cam in [15,19,22] else 'speed-journey/artifacts/release-observation')
        p=folder/f'camera{cam}.mp4'
        videos.append(dict(camera=cam,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                           trial='BaseballThrow_001',role='visibility audit only: body occlusion' if cam==15 else ('training and near reference' if cam in [20,21] else 'later optical reference')))
    result=dict(independent_baseball_trials=1,source_videos=videos,
        sign='prediction minus optical reference, km/h',near_frames=[977,978,979],near_nominal=nominal_near,
        near_mean_error_across_27_processing_conditions=near,
        later_frames=frames,later_primary_reference=primary,later_reference_candidates=references,failed_references=failed,
        later_mean_error_across_27_model_conditions=late,later_nominal=late_nominal,
        scope='Same development throw; optical candidates and provisional calibration, shared source dependencies. No population CI, no new independent trial, no muscle/force validation. Mean plus/minus SD describes processing settings, not repeated independent throws.')
    (OUT/'signed_error_report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['near_nominal','near_mean_error_across_27_processing_conditions','later_primary_reference','later_nominal','failed_references']},indent=2))
    return result


if __name__=='__main__':run()
