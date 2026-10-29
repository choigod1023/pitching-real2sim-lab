"""Training-only per-frame grid association; no static target in optimization."""
import itertools,json
from pathlib import Path
import cv2,numpy as np
from edge_calibration import camera_parameters,project
from dynamic_camera_recovery import OBJ,stereo,grid_variants,board_metrics,board_screen
from dense_camera_link import ROOT,pair_cameras,static_checks,pose_diversity

def pnp(q,k,d):
    ok,rv,t=cv2.solvePnP(OBJ,q,k,d)
    if not ok:raise ValueError('Board PnP failed')
    return cv2.Rodrigues(rv)[0],t.ravel()

def association(r,t,ids,world,db,k,d):
    camera={'R':r,'t':t,'K':k,'dist':d};chosen={};costs={}
    for f in ids:
        prediction=project(world[f],camera);errors=sorted((float(np.sqrt(np.mean(np.sum((q-prediction)**2,1)))),name) for name,q in grid_variants(db[f]).items());name=errors[0][1];chosen[f]=grid_variants(db[f])[name]
        costs[f]={'variant':name,'prediction_rms_px':errors[0][0],'runner_up_rms_px':errors[1][0]}
    return chosen,costs

def fit_correspondence(data,train,source):
    """Only training image observations determine the rig and label initialization."""
    k,d=camera_parameters(source,1);kk,dd=camera_parameters(source,3);ids=sorted(set(data[1])&set(data[3]));pa={f:pnp(data[1][f],k,d) for f in ids};world={f:OBJ@r.T+t for f,(r,t) in pa.items()};seeds=[]
    for f in train:
        ra,ta=pa[f]
        for name,q in grid_variants(data[3][f]).items():
            rb,tb=pnp(q,kk,dd);r=rb@ra.T;t=tb-r@ta;_,costs=association(r,t,train,world,data[3],kk,dd);score=float(np.median([v['prediction_rms_px'] for v in costs.values()]));seeds.append((score,f,name,r,t))
    seeds.sort(key=lambda x:x[0]);score,seedframe,seedname,r,t=seeds[0];history=[];last=None
    for iteration in range(5):
        mapped,costs=association(r,t,train,world,data[3],kk,dd);labels={f:v['variant'] for f,v in costs.items()};rms,r,t=stereo(train,data[1],mapped,k,d,kk,dd);history.append({'iteration':iteration,'train_rms_px':rms,'variant_by_frame':labels})
        if labels==last:break
        last=labels
    allmapped,costs=association(r,t,ids,world,data[3],kk,dd)
    return r,t,rms,history,{'frame':seedframe,'variant':seedname,'median_training_prediction_px':score},allmapped,costs

def run():
    cv2.setNumThreads(1);src=json.loads(Path('edge_intrinsics_frozen.json').read_text());base=json.loads((ROOT/'dense-link-results.json').read_text());out={}
    for method,record in base['methods'].items():
        if record['status']!='calculated':out[method]={'status':'insufficient_observations'};continue
        train=record['train'];held=record['heldout'];ids=record['shared'];data={i:{f:np.array(json.loads((ROOT/'coverage'/f'cam{i}_frame{f}.json').read_text())['methods'][method]['points']) for f in ids} for i in [1,3]};k,d=camera_parameters(src,1);kk,dd=camera_parameters(src,3)
        pa={f:pnp(data[1][f],k,d) for f in ids};world={f:OBJ@r.T+t for f,(r,t) in pa.items()};seeds=[]
        for f in train:
            ra,ta=pa[f]
            for name,q in grid_variants(data[3][f]).items():
                rb,tb=pnp(q,kk,dd);r=rb@ra.T;t=tb-r@ta;_,costs=association(r,t,train,world,data[3],kk,dd);score=float(np.median([v['prediction_rms_px'] for v in costs.values()]));seeds.append((score,f,name,r,t))
        seeds.sort(key=lambda x:x[0]);score,seedframe,seedname,r,t=seeds[0];history=[];last=None
        for iteration in range(5):
            mapped,costs=association(r,t,train,world,data[3],kk,dd);labels={f:v['variant'] for f,v in costs.items()};rms,r,t=stereo(train,data[1],mapped,k,d,kk,dd);history.append({'iteration':iteration,'train_rms_px':rms,'variant_by_frame':labels})
            if labels==last:break
            last=labels
        allmapped,costs=association(r,t,ids,world,data[3],kk,dd);cc=pair_cameras(r,t,src);metrics=board_metrics(held,data[1],allmapped,cc);cube,plane=static_checks(cc);passcheck=bool(board_screen(metrics) and cube['max_abs_error_percent']<=1 and max(cube['projection_rms_px_by_corner'].values())<=2 and cube['min_depth_mm']>0 and plane['conditional_metric_pass'])
        out[method]={'status':'calculated','shared':ids,'train':train,'heldout':held,'seed_training_frame':seedframe,'seed_variant':seedname,'seed_median_training_prediction_px':score,'iterations':history,'train_rms_px':rms,'R':r.tolist(),'t_mm':t.tolist(),'association_costs':costs,'heldout_board':metrics,'cube':cube,'plane':plane,'pose_diversity':pose_diversity(train,data[1],k,d),'engineering_pass':passcheck,'scientific_stage_passed':False,'limitation':'Heldout associations use observed grid points with a trained rig; not independent manual correspondence truth'}
    result={'methods':out,'scientific_stage_passed':False,'speed_error_reduction_kmh':None};(ROOT/'ordered-link-results.json').write_text(json.dumps(result,indent=2));print(json.dumps({m:{'train_rms_px':v.get('train_rms_px'),'cube_mae_mm':v.get('cube',{}).get('length_mae_mm'),'cube_max_error_percent':v.get('cube',{}).get('max_abs_error_percent'),'engineering_pass':v.get('engineering_pass')} for m,v in out.items()}));return result
if __name__=='__main__':run()
