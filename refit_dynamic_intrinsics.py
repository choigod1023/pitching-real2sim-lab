"""Prespecified model classes; refit on training board frames only."""
import copy,json
from pathlib import Path
import cv2,numpy as np
from edge_calibration import camera_parameters
from dynamic_camera_recovery import ROOT,OBJ,stereo,cameras,board_metrics,static_metrics,board_screen

def run():
    cv2.setNumThreads(1);base=json.loads(Path('edge_intrinsics_frozen.json').read_text());source=copy.deepcopy(base);split=json.loads((ROOT/'frozen-fit-eval-split.json').read_text());train=split['train'];held=split['heldout'];d={}
    for i in [1,4]:
        rows=json.loads((ROOT/f'recovered-detections-camera{i}.json').read_text());d[i]={int(f):np.array(v['methods']['native_sb']['points']) for f,v in rows.items() if v['methods']['native_sb']['points'] is not None}
    fits={}
    for i in [1,4]:
        k,dist=camera_parameters(base,i);model=base['calibrations'][f'edgertronic_{i}']['model']
        if model not in ['zero_square','k1_square']:raise ValueError('Unregistered model class')
        flags=cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3
        if model=='zero_square':flags|=cv2.CALIB_FIX_K1
        rms,kk,dd,_,_=cv2.calibrateCamera([OBJ.copy() for f in train],[d[i][f].astype(np.float32).reshape(-1,1,2) for f in train],(1280,1024),k.copy(),dist.copy(),flags=flags,criteria=(3,100,1e-8))
        c=source['calibrations'][f'edgertronic_{i}'];c.update(fx_px=float(kk[0,0]),fy_px=float(kk[1,1]),cx_px=float(kk[0,2]),cy_px=float(kk[1,2]),distortion=dd.ravel().tolist())
        q=np.vstack([d[i][f] for f in train]);hull=cv2.convexHull(q.astype(np.float32));coverage=float(cv2.contourArea(hull)/(1280*1024));max_radius_sq=max(((u-kk[0,2])/kk[0,0])**2+((v-kk[1,2])/kk[1,1])**2 for u in [0,1280] for v in [0,1024]);slope=1+3*dd.ravel()[0]*max_radius_sq
        fits[i]={'model_class':model,'train_rms_px':float(rms),'training_corner_hull_image_fraction':coverage,'minimum_k1_radial_slope_at_full_image_radius':float(slope),'original':base['calibrations'][f'edgertronic_{i}'],'refit':c}
    rms,r,t=stereo(train,d[1],d[4],*camera_parameters(source,1),*camera_parameters(source,4));cc=cameras(r,t,source);cube,plane=static_metrics(cc);bm=board_metrics(held,d[1],d[4],cc)
    result={'split':split,'intrinsic_fits':fits,'train_stereo_rms_px':rms,'R':r.tolist(),'t_mm':t.tolist(),'heldout_board':bm,'cube':cube,'plane':plane,'engineering_pass':bool(board_screen(bm) and all(v['minimum_k1_radial_slope_at_full_image_radius']>0 for v in fits.values()) and cube['max_abs_error_percent']<=1 and cube['reprojection_rms_px']<=2 and cube['min_depth_mm']>0 and plane['metric_pass']),'scientific_stage_passed':False,'speed_error_reduction_kmh':None,'selection':'Prespecified original model classes; all results reported, no selection using target lengths or speed'}
    (ROOT/'train-only-refit-intrinsics.json').write_text(json.dumps(source,indent=2));(ROOT/'train-only-refit-results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'train_stereo_rms_px':rms,'cube_MAE_mm':cube['length_mae_mm'],'cube_max_percent':cube['max_abs_error_percent'],'heldout_board_mean_spacing_MAE_mm':float(np.mean([x['spacing_mae_mm'] for x in bm])),'engineering_pass':result['engineering_pass'],'new_focal_px':{i:v['refit']['fx_px'] for i,v in fits.items()}}))
    return result
if __name__=='__main__':run()
