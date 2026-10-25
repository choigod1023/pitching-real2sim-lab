"""Conditional baseball trajectory audit; no radar-target parameter fitting.
Pixel perturbations are engineering scenarios, not measurement confidence bounds.
"""
from pathlib import Path
import argparse, hashlib, itertools, json
import numpy as np
from scipy import ndimage
from PIL import Image

FRAMES = np.arange(980, 986)

def seed(camera, frame):
    return (785-20*(frame-978),519+5.6*(frame-978)) if camera==19 else (715+17*(frame-980),298-.8*(frame-980))

def extract(image, center, threshold, mode):
    cx,cy=center; x0,y0=int(cx-24),int(cy-16)
    roi=np.asarray(image)[y0:y0+33,x0:x0+49,:3]
    yy,xx=np.indices(roi.shape[:2]); mask=roi.min(axis=2)>threshold
    if mode=='local_ellipse':
        # Predeclared spatial window follows the previous audit's image-only seed.
        # It is not an estimated ball outline and can truncate/contaminate the core.
        mask &= ((xx+x0-cx)/16)**2+((yy+y0-cy)/11)**2<=1
    labels,n=ndimage.label(mask,np.ones((3,3)))
    components=[]
    for k in range(1,n+1):
        y,x=np.where(labels==k)
        if len(x)<5:continue
        cen=np.array([x.mean()+x0,y.mean()+y0])
        components.append((np.linalg.norm(cen-center),cen,x,y))
    if not components:return {'status':'no_component','pixel':None}
    _,cen,x,y=min(components,key=lambda a:a[0])
    touches=bool((x==0).any() or (x==48).any() or (y==0).any() or (y==32).any())
    return {'status':'roi_boundary' if touches else 'candidate','pixel':cen.tolist(),
            'area_px':len(x),'bbox_wh':[int(np.ptp(x)+1),int(np.ptp(y)+1)],'touches_roi':touches}

def load_cameras(directory):
    a=json.loads((directory/'calibration_results.json').read_text())
    r=json.loads((directory/'optitrack_rig_provisional.json').read_text())
    cameras=[]
    for cam in (19,22):
        c=a['calibrations'][f'optitrack_{cam}']; p=r['camera_poses'][str(cam)]
        if np.any(c['distortion']):raise ValueError('Nonzero distortion requires correction')
        K=np.array([[c['fx_px'],0,c['cx_px']],[0,c['fy_px'],c['cy_px']],[0,0,1.]])
        P=np.column_stack((p['reference_to_camera_rotation_matrix'],p['reference_to_camera_translation_mm']))
        cameras.append((K,P))
    return cameras

def triangulate(pixels,cameras):
    equations=[]
    for uv,(K,P) in zip(pixels,cameras):
        q=np.linalg.solve(K,np.r_[uv,1]);equations.extend([q[0]*P[2]-P[0],q[1]*P[2]-P[1]])
    _,_,v=np.linalg.svd(equations); X=v[-1,:3]/v[-1,3]
    residual=[];depths=[]
    for uv,(K,P) in zip(pixels,cameras):
        xc=P@np.r_[X,1];p=K@xc;residual.append(float(np.linalg.norm(p[:2]/p[2]-uv)));depths.append(float(xc[2]))
    return X,residual,depths

def fit_trajectory(times,positions,degree=1):
    times=np.asarray(times,float);positions=np.asarray(positions,float)
    if degree not in (1,2):raise ValueError('Only linear or quadratic fits supported')
    if times.ndim!=1 or positions.shape!=(len(times),3) or not np.isfinite(times).all() or not np.isfinite(positions).all():raise ValueError('Finite times and Nx3 positions required')
    if len(times)<degree+1 or np.any(np.diff(times)<=0):raise ValueError('Need sufficient strictly increasing times')
    center=times.mean(); t=times-center
    A=np.column_stack([t**d for d in range(degree+1)])
    coef=np.linalg.lstsq(A,positions,rcond=None)[0];residual=positions-A@coef
    return {'midpoint_speed_kmh':float(np.linalg.norm(coef[1])*3.6/1000),
            'start_speed_kmh':float(np.linalg.norm(coef[1]+(2*coef[2]*t[0] if degree==2 else 0))*3.6/1000),
            'position_rmse_mm':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
            'acceleration_m_s2':(2*coef[2]/1000).tolist() if degree==2 else None,
            'center_time_s':float(center)}

def run(source,calibration,out,n_mc=500):
    if n_mc<2:raise ValueError('n_mc must be >=2')
    out.mkdir(parents=True,exist_ok=True);cameras=load_cameras(calibration);cases=[];detections=[]
    times=FRAMES/360.
    visible=np.array([0,1,5])  # Image-inspected exclusion: shoe overlap at982-984, not speed selection.
    for mode in ('component','local_ellipse'):
      for th19,th22 in itertools.product((100,130,160),(70,90,110)):
        pixels=[];flags=[]
        for frame in FRAMES:
          pair=[]
          for cam,th in zip((19,22),(th19,th22)):
            row=extract(Image.open(source/f'camera{cam}_frame{frame}.png'),seed(cam,int(frame)),th,mode)
            visual_overlap=cam==19 and frame in (982,983,984)
            row.update(camera=cam,frame=int(frame),threshold=th,mode=mode,visual_shoe_overlap=visual_overlap)
            detections.append(row);pair.append(row['pixel'])
            if visual_overlap or row['status']!='candidate':flags.append([cam,int(frame),row['status'],'shoe_overlap' if visual_overlap else ''])
          pixels.append(pair)
        case={'mode':mode,'thresholds':[th19,th22],'flags':flags,'valid_for_accuracy_claim':False}
        if any(p is None for pair in pixels for p in pair):
            case['status']='missing_detection'
            if not any(p is None for pair in pixels[:2] for p in pair):
                z=np.array([triangulate(pair,cameras)[0] for pair in pixels[:2]])
                case['two_frame']=fit_trajectory(times[:2],z)
            selected=[pixels[i] for i in visible]
            if not any(p is None for pair in selected for p in pair):
                q=np.array([triangulate(pair,cameras)[0] for pair in selected])
                case['linear_visible_primary']=fit_trajectory(times[visible],q)
                case['primary_pixels']=selected
                case['primary_positions_camera19_mm']=q.tolist()
                rng=np.random.default_rng(20261025);mc=[]
                for _ in range(n_mc):
                    # Draw full shape so stress realization matches complete cases.
                    noise=rng.uniform(-1,1,(6,2,2))[visible]
                    z=np.array([triangulate(pair,cameras)[0] for pair in np.array(selected)+noise])
                    mc.append(fit_trajectory(times[visible],z)['midpoint_speed_kmh'])
                case['primary_pixel_stress']={'n':n_mc,'seed':20261025,'speed_sd_kmh':float(np.std(mc,ddof=1)),'speed_range_kmh':[min(mc),max(mc)],'definition':'Independent uniform [-1,1] px stress; not confidence interval'}
            cases.append(case);continue
        pixels=np.array(pixels);sol=[triangulate(p,cameras) for p in pixels];xyz=np.array([s[0] for s in sol])
        case.update(status='conditional_fit',pixels=pixels.tolist(),positions_camera19_mm=xyz.tolist(),
                    reprojection_residual_px=[s[1] for s in sol],positive_depth=bool(min(min(s[2]) for s in sol)>0),
                    two_frame=fit_trajectory(times[:2],xyz[:2]),linear_visible_primary=fit_trajectory(times[visible],xyz[visible]),linear_six=fit_trajectory(times,xyz),quadratic_six=fit_trajectory(times,xyz,2),
                    evaluation_time_s=float(times[0]),evaluation_definition='Compare start_speed_kmh at frame980 for all fits; constant-speed fits independent of evaluation time. Quadratic midpoint also retained.')
        # Identical seed/distribution across methods and thresholds. Uniform pixel stress,
        # independent in each image coordinate; not measured camera noise.
        rng=np.random.default_rng(20261025);speeds=[]
        for _ in range(n_mc):
            pert=pixels+rng.uniform(-1,1,pixels.shape)
            q=np.array([triangulate(p,cameras)[0] for p in pert])
            speeds.append([fit_trajectory(times[:2],q[:2])['midpoint_speed_kmh'],fit_trajectory(times,q)['midpoint_speed_kmh'],fit_trajectory(times,q,2)['midpoint_speed_kmh'],fit_trajectory(times[visible],q[visible])['midpoint_speed_kmh']])
        case['pixel_stress']={'n':n_mc,'seed':20261025,'distribution':'independent uniform [-1,1] px; hypothetical stress, not CI',
            'methods':['two_frame','linear_six','quadratic_six_midpoint','linear_visible_primary'], 'speed_sd_kmh':np.std(speeds,axis=0,ddof=1).tolist(),
            'speed_min_kmh':np.min(speeds,axis=0).tolist(),'speed_max_kmh':np.max(speeds,axis=0).tolist()}
        cases.append(case)
    report={'scope':'Conditional image-derived baseball coordinates; unvalidated synchronization and provisional calibration; shoe overlap retained, not accuracy validation.',
       'primary_frame_indexes':[980,981,985],'primary_reason':'Image-inspected same-ball visibility; frames982-984 shoe overlap excluded, no speed-based frame selection',
       'frame_rate_assumed_hz':360,'times_s':times.tolist(),'frames':FRAMES.tolist(),
       'methods_fixed_before_speed_evaluation':['nearest connected bright component >=5 pixels','same component after image-seed ellipse 16x11px semi-axes'],
       'limitations':['Manual image-only seeds; local ellipse can bias center','Camera19 shoe overlap frames982-984','No measured ball center ground truth','FPS/synchronization provisional; calibration fixed, uncertainty unmodelled','Free flight candidate interval, not anatomically verified release'],
       'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [calibration/'calibration_results.json',calibration/'optitrack_rig_provisional.json']+[source/f'camera{cam}_frame{frame}.png' for cam in (19,22) for frame in FRAMES]},'cases':cases,'detections':detections}
    (out/'trajectory_fit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for mode in ('component','local_ellipse'):
        cs=[c for c in cases if c['mode']==mode and c['status']=='conditional_fit']
        print(mode,json.dumps({key:[min(c[key]['midpoint_speed_kmh'] for c in cs),max(c[key]['midpoint_speed_kmh'] for c in cs)] for key in ('two_frame','linear_visible_primary','linear_six','quadratic_six')}))
        print('stress SD ranges',np.min([c['pixel_stress']['speed_sd_kmh'] for c in cs],axis=0),np.max([c['pixel_stress']['speed_sd_kmh'] for c in cs],axis=0))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('calibration',type=Path);p.add_argument('out',type=Path);p.add_argument('--n-mc',type=int,default=500);a=p.parse_args();run(a.source,a.calibration,a.out,a.n_mc)
