"""Matched-frame detector/stereo experiment, cached and independent of speed."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,itertools,json,time
from pathlib import Path
import cv2,numpy as np
from edge_calibration import camera_parameters,project,triangulate,metric_screen

ROOT=Path('artifacts/dynamic-recovery')
OBJ=np.array([[x*100.,y*100.,0.] for y in range(4) for x in range(7)],np.float32)
FLAGS=cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY

def grid_variants(points):
    q=np.asarray(points).reshape(4,7,2)
    return {'identity':q.reshape(-1,2),'reverse_x':q[:,::-1].reshape(-1,2),'reverse_y':q[::-1].reshape(-1,2),'reverse_xy':q[::-1,::-1].reshape(-1,2)}

def align_same_image(points,anchor):
    choices=grid_variants(points);key=min(choices,key=lambda k:np.mean(np.sum((choices[k]-anchor)**2,axis=1)))
    q=choices[key];return q,key,float(np.sqrt(np.mean(np.sum((q-anchor)**2,axis=1))))

def detect(gray,method,anchor=None):
    scale=1.;offset=np.zeros(2);native_refine=False
    if method=='source720_subpix5':
        scale=min(1.,720/gray.shape[1]);image=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA);native_refine=True
    elif method=='native_sb':image=gray
    elif method in ['roi4_sb','roi2_border_sb']:
        if anchor is None:return None,{'failure':'no image-only anchor for target crop'}
        margin=30
        if method=='roi2_border_sb':
            g=anchor.reshape(4,7,2);spacing=np.r_[np.linalg.norm(np.diff(g,axis=0),axis=2).ravel(),np.linalg.norm(np.diff(g,axis=1),axis=2).ravel()];margin=float(2*np.max(spacing)+10)
        lo=np.maximum(np.floor(np.min(anchor,axis=0)-margin).astype(int),0);hi=np.minimum(np.ceil(np.max(anchor,axis=0)+margin).astype(int),[gray.shape[1],gray.shape[0]])
        offset=lo;scale=4. if method=='roi4_sb' else 2.;image=cv2.resize(gray[lo[1]:hi[1],lo[0]:hi[0]],None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    else:raise ValueError(method)
    ok,uv=cv2.findChessboardCornersSB(image,(7,4),flags=FLAGS)
    if not ok:return None,{'failure':'no complete board detected'}
    uv=uv.reshape(-1,2).astype(np.float64)/scale+offset
    before=uv.copy()
    if native_refine:
        q=uv.astype(np.float32).reshape(-1,1,2);cv2.cornerSubPix(gray,q,(5,5),(-1,-1),(3,30,.01));uv=q.reshape(-1,2).astype(float)
    info={'scale':scale,'offset':offset.tolist(),'native_subpix_displacement_rms_px':float(np.sqrt(np.mean(np.sum((uv-before)**2,axis=1))))}
    g=uv.reshape(4,7,2);spacing=np.r_[np.linalg.norm(np.diff(g,axis=0),axis=2).ravel(),np.linalg.norm(np.diff(g,axis=1),axis=2).ravel()]
    info['minimum_adjacent_corner_spacing_px']=float(np.min(spacing))
    if anchor is not None and method!='source720_subpix5':uv,key,rms=align_same_image(uv,anchor);info.update(image_order_alignment=key,displacement_from_source_rms_px=rms)
    return uv,info

def collect(camera,frames):
    cv2.setNumThreads(1);manifest=json.loads((ROOT/'dynamic-source-manifest.json').read_text());entry=next(e for e in manifest if e['name'].endswith(f'_{camera}.mp4'))
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();cache=ROOT/'cache';cache.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(ROOT/entry['name']));result={}
    for index,f in enumerate(frames):
        key=hashlib.sha256(f"{entry['sha256']}:{f}:{implementation}".encode()).hexdigest();path=cache/f'cam{camera}_f{f}_{key[:16]}.json'
        if path.exists():result[str(f)]=json.loads(path.read_text());continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read()
        if not ok:raise ValueError(f'Video {camera} frame {f} decode failed')
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);row={'frame':f,'camera':camera,'video_sha256':entry['sha256'],'implementation_sha256':implementation,'methods':{}}
        start=time.perf_counter();base,info=detect(gray,'source720_subpix5');row['methods']['source720_subpix5']={'points':None if base is None else base.tolist(),'info':info,'seconds':time.perf_counter()-start}
        for method in ['native_sb','roi4_sb']:
            start=time.perf_counter();uv,info=detect(gray,method,base);row['methods'][method]={'points':None if uv is None else uv.tolist(),'info':info,'seconds':time.perf_counter()-start}
        path.write_text(json.dumps(row));result[str(f)]=row
        if index%10==0:print(f'camera {camera}: {index+1}/{len(frames)} frames processed',flush=True)
    cap.release();(ROOT/f'detections_camera{camera}.json').write_text(json.dumps(result,indent=2));return result

def stereo(training,da,db,k1,d1,k4,d4):
    ret=cv2.stereoCalibrate([OBJ.copy() for f in training],[da[f].astype(np.float32).reshape(-1,1,2) for f in training],[db[f].astype(np.float32).reshape(-1,1,2) for f in training],k1.copy(),d1.copy(),k4.copy(),d4.copy(),(1280,1024),flags=cv2.CALIB_FIX_INTRINSIC,criteria=(3,100,1e-7))
    return float(ret[0]),ret[5],ret[6].reshape(3)

def cameras(r,t,source):
    k,d=camera_parameters(source,1);kk,dd=camera_parameters(source,4)
    return [{'R':np.eye(3),'t':np.zeros(3),'K':k,'dist':d},{'R':r,'t':t,'K':kk,'dist':dd}]

def board_metrics(ids,da,db,cams):
    per=[]
    for frame in ids:
        pts=[];ee=[];zz=[]
        for a,b in zip(da[frame],db[frame]):
            x,e,z=triangulate([a,b],cams);pts.append(x);ee.extend(e);zz.extend(z)
        g=np.array(pts).reshape(4,7,3);lengths=np.r_[np.linalg.norm(np.diff(g,axis=0),axis=2).ravel(),np.linalg.norm(np.diff(g,axis=1),axis=2).ravel()]
        per.append({'frame':frame,'reprojection_rms_px':float(np.sqrt(np.mean(np.sum(np.array(ee)**2,1)))),'spacing_mae_mm':float(np.mean(abs(lengths-100))),'spacing_max_abs_error_percent':float(np.max(abs(lengths-100))),'min_depth_mm':float(min(zz))})
    return per

def board_screen(metrics):
    return bool(metrics and all(np.isfinite(v['reprojection_rms_px']) and v['reprojection_rms_px']<=2 and np.isfinite(v['spacing_max_abs_error_percent']) and v['spacing_max_abs_error_percent']<=1 and np.isfinite(v['min_depth_mm']) and v['min_depth_mm']>0 for v in metrics))

def static_metrics(cams):
    a=json.loads(Path('edge_target_annotations_plane_round3.json').read_text());pts={};reproj=[];depth=[]
    for name in a['cameras']['1']['fit']:
        x,e,z=triangulate([a['cameras'][i]['fit'][name] for i in ['1','4']],cams);pts[name]=x;reproj.extend(e);depth.extend(z)
    edges=[]
    for p,q in itertools.combinations(pts,2):
        if np.count_nonzero(np.array(a['cube_coordinates'][p])-a['cube_coordinates'][q])==1:
            ll=float(np.linalg.norm(pts[p]-pts[q]));edges.append({'edge':[p,q],'length_mm':ll,'error_percent':100*(ll/698.5-1)})
    cube={'edges':edges,'length_mae_mm':float(np.mean([abs(x['length_mm']-698.5) for x in edges])),'max_abs_error_percent':max(abs(x['error_percent']) for x in edges),'reprojection_rms_px':float(np.sqrt(np.mean(np.sum(np.array(reproj)**2,1)))),'min_depth_mm':float(min(depth))}
    pp=[];ee=[];zz=[]
    for name in ['O','L','S']:
        x,e,z=triangulate([a['cameras'][i]['plane'][name] for i in ['1','4']],cams);pp.append(x);ee.extend(e);zz.extend(z)
    expected={'O-L':200,'O-S':150,'L-S':250};ll={name:float(np.linalg.norm(pp[i]-pp[j])) for name,i,j in [('O-L',0,1),('O-S',0,2),('L-S',1,2)]};rms=float(np.sqrt(np.mean(np.sum(np.array(ee)**2,1))))
    plane={'lengths_mm':ll,'reprojection_rms_px':rms,'min_depth_mm':float(min(zz)),'metric_pass':metric_screen(ll,expected,rms,min(zz)),'status':'UNKNOWN: marker centre identity unconfirmed; numeric comparison conditional'}
    return cube,plane

def run_fit(d1,d4,methods=('source720_subpix5','native_sb','roi4_sb')):
    source=json.loads(Path('edge_intrinsics_frozen.json').read_text())
    common=sorted(int(f) for f in set(d1)&set(d4) if all(d[f]['methods'][m]['points'] is not None for d in [d1,d4] for m in methods));train=[f for f in common if f%60];held=[f for f in common if not f%60]
    split={'methods':list(methods),'common_frames':common,'train':train,'heldout':held,'rule':'frame modulo 60 == 0 is heldout; primary shift 0; same frames across methods','annotation_sha256':hashlib.sha256(Path('edge_target_annotations_plane_round3.json').read_bytes()).hexdigest()};(ROOT/'frozen-fit-eval-split.json').write_text(json.dumps(split,indent=2))
    if len(common)<8 or len(train)<5 or len(held)<2:
        failure={'status':'insufficient_matched_detections','split':split,'scientific_stage_passed':False,'next_action':'Inspect detector failures and source imagery; do not relax minimum observation counts.'};(ROOT/'dynamic-recovery-results.json').write_text(json.dumps(failure,indent=2));return failure
    results={}
    for method in methods:
        da={int(f):np.array(row['methods'][method]['points']) for f,row in d1.items() if row['methods'][method]['points'] is not None};db={int(f):np.array(row['methods'][method]['points']) for f,row in d4.items() if row['methods'][method]['points'] is not None};options=[]
        for variant in grid_variants(db[train[0]]):
            bb={f:grid_variants(q)[variant] for f,q in db.items()};rms,r,t=stereo(train,da,bb,*camera_parameters(source,1),*camera_parameters(source,4));options.append((rms,variant,r,t))
        options.sort(key=lambda x:x[0]);rms,variant,r,t=options[0];bb={f:grid_variants(q)[variant] for f,q in db.items()};cc=cameras(r,t,source);cube,plane=static_metrics(cc);heldm=board_metrics(held,da,bb,cc)
        result={'train_stereo_rms_px':rms,'selected_training_grid_variant':variant,'grid_variants_train_rms_px':{v:float(e) for e,v,_,_ in options},'R':r.tolist(),'t_mm':t.tolist(),'baseline_mm':float(np.linalg.norm(t)),'heldout_board':heldm,'cube':cube,'plane':plane,'engineering_pass':bool(board_screen(heldm) and cube['max_abs_error_percent']<=1 and cube['reprojection_rms_px']<=2 and cube['min_depth_mm']>0 and plane['metric_pass']),'scientific_stage_passed':False}
        results[method]=result;print(method,'stereo',round(rms,4),'cubeMAE',round(cube['length_mae_mm'],3),'max%',round(cube['max_abs_error_percent'],3),flush=True)
    summary={'split':split,'methods':results,'scientific_stage_passed':False,'speed_error_reduction_kmh':None,'limits':['Development calibration, not independent athlete test','Fixed single-reader static observations','Capture clock/speed labels unresolved','No posture or force intervention tested']};(ROOT/'dynamic-recovery-results.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit-only',action='store_true');args=p.parse_args()
    if args.fit_only:d={i:json.loads((ROOT/f'detections_camera{i}.json').read_text()) for i in [1,4]}
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={i:pool.submit(collect,i,list(range(0,553,12))) for i in [1,4]};d={i:f.result() for i,f in futures.items()}
    run_fit(d[1],d[4])
