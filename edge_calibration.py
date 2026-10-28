"""Traceable cube PnP, discrete registration and withheld target checks.

All distances are mm; image coordinates are native pixels. No pitch speed is read.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

OUT = Path('artifacts/edge-calibration')
INTRINSICS = Path('edge_intrinsics_frozen.json')

def camera_parameters(source, i):
    c = source['calibrations'][f'edgertronic_{i}']
    k = np.array([[c['fx_px'], 0, c['cx_px']], [0,c['fy_px'],c['cy_px']], [0,0,1.]])
    return k, np.array(c['distortion'], dtype=float)

def project(points, camera):
    p, _ = cv2.projectPoints(np.asarray(points,dtype=float).reshape(-1,3), cv2.Rodrigues(camera['R'])[0],camera['t'],camera['K'],camera['dist'])
    return p.reshape(-1,2)

def fit_camera(obj, uv, k, dist):
    obj,uv=np.asarray(obj,float),np.asarray(uv,float)
    if len(obj)<6 or np.linalg.matrix_rank(obj-obj.mean(0))<3:
        raise ValueError('At least six non-coplanar observations required')
    ok, r, t = cv2.solvePnP(obj,uv,k,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise ValueError('PnP failed')
    r,t=cv2.solvePnPRefineLM(obj,uv,k,dist,r,t)
    c={'K':k,'dist':dist,'R':cv2.Rodrigues(r)[0],'t':t.reshape(3)}
    return c

def register_cube(camera, quarter_turns, side):
    theta=quarter_turns*np.pi/2
    s=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    centre=np.array([side/2,side/2,0.])
    r=camera['R']@s.T
    return dict(camera,R=r,t=camera['R']@(centre-s.T@centre)+camera['t'])

def triangulate(uv, cameras):
    if len(cameras)<2: raise ValueError('At least two views required')
    if len(uv)!=len(cameras): raise ValueError('One observation per camera required')
    rows=[]
    for q,c in zip(uv,cameras):
        q=cv2.undistortPoints(np.array(q,float).reshape(1,1,2),c['K'],c['dist']).reshape(2)
        p=np.column_stack((c['R'],c['t']))
        rows.extend([q[0]*p[2]-p[0],q[1]*p[2]-p[1]])
    _,_,v=np.linalg.svd(rows)
    if abs(v[-1,3])<1e-12: raise ValueError('Point at infinity')
    x=v[-1,:3]/v[-1,3]
    def residual(x): return np.concatenate([project([x],c)[0]-q for q,c in zip(uv,cameras)])
    x=least_squares(residual,x,method='lm').x
    depth=[float((c['R']@x+c['t'])[2]) for c in cameras]
    return x, np.asarray(residual(x)).reshape(-1,2), depth

def lengths(x):
    return {'O-L':float(np.linalg.norm(x[0]-x[1])), 'O-S':float(np.linalg.norm(x[0]-x[2])), 'L-S':float(np.linalg.norm(x[1]-x[2]))}

def serialise(c): return {k:v.tolist() for k,v in c.items()}

def metric_screen(measured, reference, rms_px, min_depth_mm):
    """Conditional engineering test, not instrument/athlete validation."""
    values=np.array([measured[k]/reference[k]-1 for k in reference],float)
    return bool(np.all(np.isfinite(values)) and np.isfinite(rms_px) and
                np.isfinite(min_depth_mm) and np.max(np.abs(values))<=.01+1e-12 and
                rms_px<=2 and min_depth_mm>0)

def run(annotation_path=Path('edge_target_annotations.json'), out=OUT):
    a=json.loads(annotation_path.read_text()); src=json.loads(INTRINSICS.read_text())
    side=a['cube_side_mm']; coordinates={k:np.array(v)*side for k,v in a['cube_coordinates'].items()}
    cams={}; diagnostics={}
    for i,obs in a['cameras'].items():
        ids=list(obs['fit']); obj=np.array([coordinates[j] for j in ids]); uv=np.array(list(obs['fit'].values()))
        k,d=camera_parameters(src,int(i)); c=fit_camera(obj,uv,k,d); cams[i]=c
        fiterr=project(obj,c)-uv
        held=np.array([coordinates[j] for j in obs['heldout']]); err=project(held,c)-np.array(list(obs['heldout'].values()))
        fitrms=float(np.sqrt(np.mean(np.sum(fiterr**2,1)))); hrms=float(np.sqrt(np.mean(np.sum(err**2,1))))
        depth=(obj@c['R'].T+c['t'])[:,2]
        diagnostics[i]={'fit_count':len(ids),'fit_rms_px':fitrms,'heldout_count':len(held),'heldout_rms_px':hrms,
            'heldout_errors_px':{j:float(np.linalg.norm(e)) for j,e in zip(obs['heldout'],err)},'min_depth_mm':float(depth.min()),
            'det_R':float(np.linalg.det(c['R'])),'pass':bool(fitrms<=2 and hrms<=2 and np.all(depth>0) and abs(np.linalg.det(c['R'])-1)<1e-6)}
    candidates=[]
    # Reference camera 4 fixes the arbitrary horizontal cube axes; all 64 relative rotations retained.
    for turns in itertools.product(range(4),repeat=3):
        cc={i:register_cube(cams[i],turns[int(i)-1] if i!='4' else 0,side) for i in cams}
        points=[]; errors=[]; depths=[]
        for name in ['O','L','S']:
            x,e,z=triangulate([a['cameras'][i]['plane'][name] for i in cc],list(cc.values()))
            points.append(x);errors.extend(e);depths.extend(z)
        candidates.append({'turns':dict(zip(['1','2','3'],turns)),'plane_reprojection_rms_px':float(np.sqrt(np.mean(np.sum(np.array(errors)**2,1)))),
                           'min_depth_mm':min(depths),'points_mm':np.array(points).tolist()})
    candidates.sort(key=lambda r:(r['min_depth_mm']<=0,r['plane_reprojection_rms_px']))
    best=candidates[0]; cc={i:register_cube(cams[i],best['turns'].get(i,0),side) for i in cams}
    target=a['plane_distances_mm']; checks=[]
    for n in [2,4]:
        for ids in itertools.combinations(cc,n):
            points=[];errors=[];depths=[]
            for name in ['O','L','S']:
                x,e,z=triangulate([a['cameras'][i]['plane'][name] for i in ids],[cc[i] for i in ids])
                points.append(x);errors.extend(e);depths.extend(z)
            ll=lengths(points); rel={k:(ll[k]/target[k]-1)*100 for k in ll}
            rms=float(np.sqrt(np.mean(np.sum(np.array(errors)**2,1))))
            checks.append({'cameras':list(ids),'lengths_mm':ll,'signed_length_error_percent':rel,'reprojection_rms_px':rms,'min_depth_mm':min(depths),
                           'pass':metric_screen(ll,target,rms,min(depths))})
    # Fixed +/-1 px coordinate perturbation for each observation axis, not stochastic confidence bounds.
    base=np.array(best['points_mm']); sensitivity=[]
    for i in cc:
        for name_index,name in enumerate(['O','L','S']):
            for axis in range(2):
                for delta in [-1,1]:
                    uv=[list(a['cameras'][j]['plane'][name]) for j in cc];uv[list(cc).index(i)][axis]+=delta
                    x,_,_=triangulate(uv,list(cc.values())); pts=base.copy();pts[name_index]=x
                    sensitivity.append({'camera':i,'point':name,'axis':axis,'delta_px':delta,'lengths_mm':lengths(pts)})
    result={'annotation_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),'intrinsic_source_sha256':hashlib.sha256(INTRINSICS.read_bytes()).hexdigest(),
        'annotation_round':a['annotation_round'],'units':'mm, native pixels','pose_checks':diagnostics,'cube_registration_candidates':candidates,
        'plane_checks':checks,'registered_cameras':{i:serialise(c) for i,c in cc.items()},'plane_coordinate_sensitivity':sensitivity,
        'engineering_pass':bool(all(x['pass'] for x in diagnostics.values()) and all(x['pass'] for x in checks)),
        'scientific_data_gate_pass':False,'speed_error_reduction_kmh':None,
        'plane_metric_status':'UNKNOWN: image centres are provisional; conditional numeric checks only',
        'limitations':['Single-analyst annotations, no independent image truth','CS-200 participates in discrete registration','360 Hz source declaration not independently verified','Speed label mapping unresolved','No accepted 3D hand-ball result if engineering checks fail']}
    out.mkdir(parents=True,exist_ok=True);(out/'calibration-results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'pose_checks':diagnostics,'registration':candidates[:2],'plane_checks':checks,'engineering_pass':result['engineering_pass']},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,default=Path('edge_target_annotations.json'));args=p.parse_args();run(args.annotations)
