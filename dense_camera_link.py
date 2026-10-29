"""Acquire dense visibility, then independently audit a new camera connection."""
import argparse,hashlib,itertools,json,shutil
from pathlib import Path
import cv2,numpy as np
import probe_dynamic_coverage as probe
from dynamic_camera_recovery import OBJ,stereo,board_metrics,board_screen,grid_variants
from edge_calibration import camera_parameters,triangulate,metric_screen

ROOT=Path('artifacts/dense-camera-link')
SOURCE=Path('../dynamic-camera-recovery-20261029/artifacts/dynamic-recovery')

def acquire(max_new=40):
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'coverage').mkdir(exist_ok=True)
    manifest=SOURCE/'dynamic-source-manifest.json';dest=ROOT/manifest.name
    if not dest.exists():shutil.copyfile(manifest,dest)
    for path in (SOURCE/'coverage').glob('cam*_frame*.json'):
        dst=ROOT/'coverage'/path.name
        if not dst.exists():shutil.copyfile(path,dst)
    probe.ROOT=ROOT
    return probe.run(max_new,(1,3),540,612,1,input_root=SOURCE)

def pair_cameras(r,t,source):
    k,d=camera_parameters(source,1);kk,dd=camera_parameters(source,3)
    return [{'R':np.eye(3),'t':np.zeros(3),'K':k,'dist':d},{'R':r,'t':t,'K':kk,'dist':dd}]

def map_cube_name(name,quarter_turn):
    # Explicit right-handed quarter rotation around the horizontal cube centre.
    coords={'A':(1,0),'B':(0,0),'C':(0,1),'D':(1,1)};x,y=coords[name[0]]
    for _ in range(quarter_turn%4):x,y=1-y,x
    letter=next(k for k,v in coords.items() if v==(x,y));return letter+name[1:]

def static_checks(cameras,quarter_turn=1):
    path=Path('edge_target_annotations_plane_round3.json');a=json.loads(path.read_text());left=a['cameras']['1']['fit'];right={map_cube_name(k,quarter_turn):v for k,v in a['cameras']['3']['fit'].items()};points={};projection={};depth=[]
    for name in sorted(set(left)&set(right)):
        x,e,z=triangulate([left[name],right[name]],cameras);points[name]=x;projection[name]=float(np.sqrt(np.mean(np.sum(e**2,axis=1))));depth.extend(z)
    edges=[]
    for p,q in itertools.combinations(points,2):
        if np.count_nonzero(np.array(a['cube_coordinates'][p])-a['cube_coordinates'][q])==1:
            ll=float(np.linalg.norm(points[p]-points[q]));edges.append({'edge':[p,q],'length_mm':ll,'error_percent':100*(ll/698.5-1)})
    xyz=[];ee=[];zz=[]
    for name in ['O','L','S']:
        x,e,z=triangulate([a['cameras'][i]['plane'][name] for i in ['1','3']],cameras);xyz.append(x);ee.extend(e);zz.extend(z)
    lengths={k:float(np.linalg.norm(xyz[i]-xyz[j])) for k,i,j in [('O-L',0,1),('O-S',0,2),('L-S',1,2)]};rms=float(np.sqrt(np.mean(np.sum(np.array(ee)**2,1))))
    cube={'visible_reconstructed_corners':list(points),'available_edge_count':len(edges),'edges':edges,'projection_rms_px_by_corner':projection,'length_mae_mm':float(np.mean([abs(x['length_mm']-698.5) for x in edges])),'max_abs_error_percent':max(abs(x['error_percent']) for x in edges),'min_depth_mm':min(depth),'missing_corners':sorted(set(left)-set(right))}
    plane={'lengths_mm':lengths,'projection_rms_px':rms,'min_depth_mm':min(zz),'conditional_metric_pass':metric_screen(lengths,{'O-L':200,'O-S':150,'L-S':250},rms,min(zz)),'status':'UNKNOWN: centre identities not independently confirmed'}
    return cube,plane

def pose_diversity(ids,data,k,d):
    normals=[];translations=[]
    for frame in ids:
        ok,r,t=cv2.solvePnP(OBJ,data[frame],k,d)
        if ok:normals.append(cv2.Rodrigues(r)[0][:,2]);translations.append(t.ravel())
    angles=[float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1)))) for a,b in itertools.combinations(normals,2)]
    return {'solved':len(normals),'maximum_normal_span_deg':max(angles,default=0.),'translation_axis_span_mm':np.ptp(translations,axis=0).tolist() if translations else None,'interpretation':'Adjacent development frames; span is descriptive, not independent validation'}

def fit():
    cv2.setNumThreads(1);source=json.loads(Path('edge_intrinsics_frozen.json').read_text());coverage=ROOT/'coverage-1-3-540-612-s1.json'
    if not coverage.exists() or not json.loads(coverage.read_text())['complete']:raise ValueError('Finish dense acquisition before fitting')
    allrows={i:{f:json.loads((ROOT/'coverage'/f'cam{i}_frame{f}.json').read_text()) for f in range(540,613)} for i in [1,3]};results={}
    for method in ['native_sb','source720_subpix5']:
        data={i:{f:np.array(row['methods'][method]['points']) for f,row in rows.items() if row['methods'][method]['points'] is not None} for i,rows in allrows.items()};shared=sorted(set(data[1])&set(data[3]));train=[f for f in shared if f%5];held=[f for f in shared if not f%5];record={'shared':shared,'train':train,'heldout':held,'split_rule':'frame modulo5==0 heldout, new camera1-3 experiment'}
        (ROOT/f'split-{method}.json').write_text(json.dumps(record,indent=2))
        if len(shared)<8 or len(train)<5 or len(held)<2:results[method]={**record,'status':'insufficient_observations','scientific_stage_passed':False};continue
        options=[]
        for variant in grid_variants(data[3][train[0]]):
            bb={f:grid_variants(q)[variant] for f,q in data[3].items()};rms,r,t=stereo(train,data[1],bb,*camera_parameters(source,1),*camera_parameters(source,3));options.append((rms,variant,r,t))
        options.sort(key=lambda x:x[0]);rms,variant,r,t=options[0];bb={f:grid_variants(q)[variant] for f,q in data[3].items()};cameras=pair_cameras(r,t,source);metrics=board_metrics(held,data[1],bb,cameras);cube,plane=static_checks(cameras)
        record.update(status='calculated',train_stereo_rms_px=rms,chosen_training_variant=variant,training_variant_rms={v:float(x) for x,v,_,_ in options},R=r.tolist(),t_mm=t.tolist(),baseline_mm=float(np.linalg.norm(t)),heldout_board=metrics,cube=cube,plane=plane,pose_diversity=pose_diversity(train,data[1],*camera_parameters(source,1)),engineering_pass=bool(board_screen(metrics) and cube['max_abs_error_percent']<=1 and max(cube['projection_rms_px_by_corner'].values())<=2 and cube['min_depth_mm']>0 and plane['conditional_metric_pass']),scientific_stage_passed=False)
        results[method]=record
    payload={'methods':results,'fixed_annotation_sha256':hashlib.sha256(Path('edge_target_annotations_plane_round3.json').read_bytes()).hexdigest(),'scientific_stage_passed':False,'speed_error_reduction_kmh':None};(ROOT/'dense-link-results.json').write_text(json.dumps(payload,indent=2));print(json.dumps({m:{k:v.get(k) for k in ['status','train_stereo_rms_px','engineering_pass']}|{'shared':len(v['shared']),'train':len(v['train']),'heldout':len(v['heldout'])} for m,v in results.items()}));return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['acquire','fit']);p.add_argument('--max-new',type=int,default=40);a=p.parse_args();acquire(a.max_new) if a.action=='acquire' else fit()
