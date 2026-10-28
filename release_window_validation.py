"""Four predeclared alignment comparators; no radar/ball outcome fitting."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.optimize import minimize_scalar
from joint_video import load_joints
from fixed_alignment_audit import SEGMENTS, project, similarity
from hand_image_audit import region_distance


def interpolate(data, frames):
    data = np.asarray(data, float)
    f = np.asarray(frames, float)
    if (data.ndim < 2 or f.ndim != 1 or not np.isfinite(data).all()
            or not np.isfinite(f).all() or np.any(f < 0) or np.any(f > len(data)-1)):
        raise ValueError('Finite data and in-range 1D fractional indexes required')
    lo = np.floor(f).astype(int)
    hi = np.minimum(lo + 1, len(data)-1)
    w = (f-lo).reshape((-1,) + (1,)*(data.ndim-1))
    return data[lo]*(1-w) + data[hi]*w


def fit_similarity(source, target):
    x, y = np.asarray(source, float).reshape(-1,3), np.asarray(target,float).reshape(-1,3)
    if x.shape != y.shape or len(x) < 3 or not np.isfinite([x,y]).all():
        raise ValueError('Matched finite Nx3 points required')
    xc, yc = x-x.mean(0), y-y.mean(0)
    if np.linalg.matrix_rank(xc) < 2:
        raise ValueError('Degenerate source geometry')
    u, s, vt = np.linalg.svd(yc.T@xc/len(x))
    d = np.ones(3); d[-1] = np.sign(np.linalg.det(u@vt))
    r = (u*d)@vt
    scale = np.sum(s*d) / np.mean(np.sum(xc*xc,axis=1))
    if scale <= 0:
        raise ValueError('Nonpositive scale')
    return dict(scale=float(scale), rotation=r.tolist(),
                translation=(y.mean(0)-scale*(r@x.mean(0))).tolist())


def fit_models(origins, frames, target, baseline, bounds=(-2.,2.)):
    """Only supplied training rows are visible to this function."""
    def fit_at(delta):
        return fit_similarity(interpolate(origins,frames+delta),target)
    def objective(delta, spatial):
        a = fit_at(delta) if spatial else baseline
        err = similarity(interpolate(origins,frames+delta),a)-target
        return float(np.mean(np.sum(err*err,axis=-1)))
    result = {'M0':dict(offset=1.,alignment=baseline,fit='unchanged')}
    for name, spatial in [('M1',False),('M3',True)]:
        opt = minimize_scalar(lambda d:objective(d,spatial),bounds=bounds,
                              method='bounded',options={'xatol':1e-7})
        # Include endpoints explicitly; bounded minimization does not evaluate them.
        delta = min([float(opt.x),*bounds],key=lambda d:objective(d,spatial))
        result[name] = dict(offset=delta,alignment=fit_at(delta) if spatial else baseline,
                            boundary_solution=bool(min(abs(delta-b) for b in bounds)<1e-4),
                            fit='training_only')
    result['M2'] = dict(offset=1.,alignment=fit_at(1.),fit='training_only')
    profile = [dict(offset=float(d),fixed_mse_m2=objective(d,False),
                    refit_mse_m2=objective(d,True)) for d in np.linspace(*bounds,65)]
    return {k:result[k] for k in ['M0','M1','M2','M3']},profile


def distance_stats(error):
    e = np.asarray(error,float)
    if e.size == 0 or not np.isfinite(e).all():
        raise ValueError('Nonempty finite errors required')
    return dict(n=int(e.size),mean=float(e.mean()),median=float(np.median(e)),
                rmse=float(np.sqrt(np.mean(e*e))),p95=float(np.percentile(e,95)),max=float(e.max()))


def run():
    root=Path(__file__).resolve().parent; lab=root.parent
    out=root/'artifacts/release-window';out.mkdir(parents=True,exist_ok=True)
    up=lab/'openbiomechanics/computer_vision'
    paths=dict(keypoints=up/'4d_reconstruction/release_keypoints.csv',
      alignment=up/'4d_reconstruction/theia_alignment.json',
      body=lab/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d',
      calibration=up/'calibration/results/calibration_results.json',
      rig=up/'calibration/results/optitrack_rig_provisional.json',
      old_regions=root/'hand_image_regions.json',new_regions=root/'release_window_regions.json',
      protocol=root/'RELEASE_WINDOW_PROTOCOL.md',
      followup=root/'RELEASE_WINDOW_FOLLOWUP.md',script=Path(__file__).resolve(),
      ball=lab/'speed-journey/artifacts/release-observation/candidate_audit.json')
    rows=list(csv.DictReader(paths['keypoints'].open()))
    frames=np.array(sorted({int(r['video_frame']) for r in rows}))
    lookup={(int(r['video_frame']),r['segment']):[float(r[k]) for k in ['x_m','y_m','z_m']] for r in rows}
    target=np.array([[lookup[f,s] for s in SEGMENTS] for f in frames])
    xyz,names,rate=load_joints(paths['body']);idx=[names.index(s) for s in SEGMENTS]
    origins=xyz[:,idx]
    train=frames<=954;test=frames>=955
    assert not set(frames[train]) & set(frames[test])
    baseline=json.loads(paths['alignment'].read_text())
    models,profile=fit_models(origins,frames[train],target[train],baseline)
    intrinsic=json.loads(paths['calibration'].read_text())['calibrations']
    poses=json.loads(paths['rig'].read_text())['camera_poses']
    regions=[dict(r,annotation_set=label) for label,key in [('old','old_regions'),('new','new_regions')]
             for r in json.loads(paths[key].read_text())['regions']]
    report=dict(scope='Conditional consistency to contributor 3D points and single-reader image regions, NOT independent anatomical accuracy or physical synchronization.',
                rate_hz=rate,train_frames=frames[train].tolist(),test_frames=frames[test].tolist(),
                models=models,profile=profile,regions=regions,summaries={},region_records=[])
    for name,model in models.items():
        pred=similarity(interpolate(origins,frames+model['offset']),model['alignment'])
        err=np.linalg.norm(pred-target,axis=-1)*1000
        summary={split:dict(all_mm=distance_stats(err[mask]),
                       hand_mm=distance_stats(err[mask,SEGMENTS.index('l_hand')]))
                 for split,mask in [('train',train),('test',test)]}
        rr=[]
        for r in regions:
            p=similarity(interpolate(xyz[:,names.index(r['segment'])],np.array([r['frame']+model['offset']]))[0],model['alignment'])
            uv=project(p,poses[str(r['camera'])],intrinsic[f"optitrack_{r['camera']}"])
            dist,center=region_distance(uv,r['box'])
            rr.append(dict(**r,model=name,pixel=uv.tolist(),outside_px=dist,center_proxy_px=center))
        for label in ['old','new']:
            selected=[r for r in rr if r['annotation_set']==label]
            summary[label+'_regions']=dict(n=len(selected),inside=sum(r['outside_px']==0 for r in selected),
                outside_px=distance_stats([r['outside_px'] for r in selected]),
                center_proxy_px=distance_stats([r['center_proxy_px'] for r in selected]))
        report['region_records']+=rr
        report['summaries'][name]=summary
    # Same observed ball trajectory, before/after hand alignment. No ball data entered fit_models.
    ball=json.loads(paths['ball'].read_text())
    observed=sorted([r for r in ball['conditional_triangulation'] if r['threshold']==80],key=lambda r:r['frame'])
    bf=np.array([r['frame'] for r in observed]); b=np.array([r['position_camera19_mm'] for r in observed])/1000
    dt=1/rate; speed=np.linalg.norm(np.gradient(b,dt,axis=0),axis=1)*3.6
    report['observed_ball']=dict(frames=bf.tolist(),finite_difference_speed_kmh=speed.tolist(),
                                definition='Image-derived conditional velocity, no independent radar target or verified contact time')
    for name,model in models.items():
        hand=similarity(interpolate(xyz[:,names.index('l_hand')],bf+model['offset']),model['alignment'])
        hv=np.gradient(hand,dt,axis=0)
        report['summaries'][name]['ball_hand_conditional']=dict(
            distances_mm=(np.linalg.norm(b-hand,axis=1)*1000).tolist(),
            hand_speed_kmh=(np.linalg.norm(hv,axis=1)*3.6).tolist(),
            same_ball_speed_kmh=speed.tolist())
    for c in [19,20,21]:
        paths[f'video{c}']=lab/('speed-journey/artifacts/release-observation' if c in [20,21] else 'ball-coordinate-audit/artifacts/video-audit')/f'camera{c}.mp4'
    report['inputs']={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()}
    # Predeclared follow-up: no held-out score selects an adopted transform.
    stability=[]
    for label,columns in [('lower_six',list(range(6))),('upper_six',list(range(6,12))),
                           ('throwing_elbow_hand',[SEGMENTS.index('l_larm'),SEGMENTS.index('l_hand')])]:
        fitted,_=fit_models(origins[:,columns],frames[train],target[train][:,columns],baseline)
        stability.append(dict(source=label,offsets={k:v['offset'] for k,v in fitted.items()},
            boundary_solutions={k:v.get('boundary_solution',False) for k,v in fitted.items()}))
    report['landmark_subset_stability']=stability
    report['region_review']=dict(scientific_ground_truth=False,
        note='New approximate annotations include uncertain landmark placement, notably camera20 elbow970 and hand/elbow986 and camera21 hand/elbow986. All are retained, no transform fitted to them, no anatomical accuracy claim.')
    report['scientific_gate_passed']=False
    report['unresolved']=['Independent release-window anatomical hand coordinates and repeatability',
                          'Physical Theia-to-camera timing and its uncertainty',
                          'Contact termination ground truth',
                          'Same-pitch speed reference instrument/time/location']
    (out/'alignment_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:dict(offset=v['offset'],test_all_mm=report['summaries'][k]['test']['all_mm']['rmse'],
            test_hand_mm=report['summaries'][k]['test']['hand_mm']['rmse'],new_outside_px=report['summaries'][k]['new_regions']['outside_px']['mean']) for k,v in models.items()},indent=2))
    return report


if __name__=='__main__':run()
