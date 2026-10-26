"""Background-subtracted component boundary ellipse; exploratory, not truth."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
from scipy.optimize import least_squares
from ball_trajectory_fit import triangulate
from ball_holdout_audit import project

ROOT=Path(__file__).resolve().parent

def outline(image,seed,fraction):
    if not 0<fraction<1:raise ValueError('Fraction must be between zero and one')
    x0,y0=int(seed[0]-24),int(seed[1]-16)
    arr=np.asarray(image)
    if x0<0 or y0<0 or x0+49>arr.shape[1] or y0+33>arr.shape[0]:return dict(status='roi_outside',pixel=None)
    gray=arr[y0:y0+33,x0:x0+49,:3].min(axis=2).astype(float)
    border=np.r_[gray[0],gray[-1],gray[:,0],gray[:,-1]]
    signal=np.maximum(gray-np.median(border),0)
    if signal.max()<10:return dict(status='low_contrast',pixel=None)
    labels,n=ndimage.label(signal>fraction*signal.max(),np.ones((3,3)))
    choices=[]
    for k in range(1,n+1):
        yy,xx=np.where(labels==k)
        if len(xx)>=5:choices.append((np.linalg.norm(np.array([xx.mean()+x0,yy.mean()+y0])-seed),k))
    if not choices:return dict(status='no_component',pixel=None)
    mask=labels==min(choices)[1]
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():return dict(status='roi_boundary',pixel=None)
    yy,xx=np.where(mask&~ndimage.binary_erosion(mask))
    if len(xx)<12:return dict(status='insufficient_boundary',pixel=None)
    xy=np.column_stack([xx,yy]);center=xy.mean(axis=0)
    eig,vec=np.linalg.eigh(np.cov(xy.T));axes=np.sqrt(np.maximum(2*eig,4));angle=np.arctan2(vec[1,0],vec[0,0])
    def components(p):
        c,s=np.cos(p[4]),np.sin(p[4]);return (xy-p[:2])@np.array([[c,-s],[s,c]])
    def residual(p):
        uv=components(p)/np.exp(p[2:4]);return (np.linalg.norm(uv,axis=1)-1)*np.exp(p[2:4]).min()
    fit=least_squares(residual,np.r_[center,np.log(axes),angle],loss='soft_l1',f_scale=1,max_nfev=200,
        bounds=([-5,-5,np.log(1),np.log(1),-20],[54,38,np.log(40),np.log(40),20]))
    p=fit.x;axes=np.exp(p[2:4]);uv=components(p)/axes
    bins=np.unique(np.floor((np.arctan2(uv[:,1],uv[:,0])+np.pi)/(2*np.pi)*12).astype(int)%12)
    rms=float(np.sqrt(np.mean(residual(p)**2)))
    accepted=fit.success and axes.min()>=2 and axes.max()<=24 and axes.max()/axes.min()<=3 and len(bins)>=9 and rms<=1.5
    return dict(status='candidate' if accepted else 'unsupported_outline',pixel=(p[:2]+[x0,y0]).tolist() if accepted else None,
        fitted_pixel=(p[:2]+[x0,y0]).tolist(),axes=axes.tolist(),angle=float(p[4]),boundary_rmse_px=rms,angular_bins=len(bins),background=float(np.median(border)))

def run():
    lab=ROOT.parent;source=lab/'speed-journey/artifacts/release-observation/candidate_audit.json'
    hold=lab/'hand-ball-link/artifacts/heldout-ball/report.json'
    cal=lab/'openbiomechanics/computer_vision/calibration/results'
    raw=json.loads(source.read_text());held=json.loads(hold.read_text())
    intr=json.loads((cal/'calibration_results.json').read_text())['calibrations'];rig=json.loads((cal/'optitrack_rig_provisional.json').read_text())['camera_poses']
    cameras={}
    for c in (19,20,21):
        k=intr[f'optitrack_{c}'];r=rig[str(c)]
        if any(k['distortion']):raise ValueError('Unsupported distortion')
        cameras[c]=(np.array([[k['fx_px'],0,k['cx_px']],[0,k['fy_px'],k['cy_px']],[0,0,1.]]),np.column_stack([r['reference_to_camera_rotation_matrix'],r['reference_to_camera_translation_mm']]))
    seeds={(d['camera'],d['frame']):d['manual_seed'] for d in raw['detections']}
    base={d['frame']:d for d in raw['conditional_triangulation'] if d['threshold']==80}
    out=ROOT/'artifacts/outline-center';out.mkdir(parents=True,exist_ok=True)
    detections=[];comparisons=[];summaries=[];sources=[source,hold,cal/'calibration_results.json',cal/'optitrack_rig_provisional.json',ROOT/'OUTLINE_PROTOCOL.md']
    for frac in (.3,.5,.7):
        for frame in range(973,980):
            pair=[]
            for cam in (20,21):
                path=lab/f'speed-journey/artifacts/release-observation/camera{cam}_{frame}.png';sources.append(path)
                im=Image.open(path).convert('RGB');d=outline(im,seeds[cam,frame],frac);d.update(camera=cam,frame=frame,fraction=frac);detections.append(d);pair.append(d)
                if frac==.5 and frame in (973,977):
                    x,y=seeds[cam,frame];crop=im.crop((x-40,y-32,x+40,y+32)).resize((640,512));draw=ImageDraw.Draw(crop)
                    if d['pixel']:
                        u,v=(np.array(d['pixel'])-[x-40,y-32])*8;draw.ellipse((u-7,v-7,u+7,v+7),outline='red',width=2)
                    draw.text((5,5),d['status'],fill='red');crop.save(out/f'camera{cam}_{frame}.png')
            if frame<977:continue
            for h in [h for h in held['detections'] if h['frame']==frame]:
                row=dict(fraction=frac,frame=frame,heldout_threshold=h['threshold'],heldout_status=h['status'],source_status=[d['status'] for d in pair],baseline_error_px=None,new_error_px=None)
                if h['status']=='candidate':
                    row['baseline_error_px']=float(np.linalg.norm(project(base[frame]['position_camera19_mm'],*cameras[19])-h['pixel']))
                    if all(d['status']=='candidate' for d in pair):
                        point,_,depth=triangulate([d['pixel'] for d in pair],[cameras[20],cameras[21]])
                        if min(depth)>0:row['new_error_px']=float(np.linalg.norm(project(point,*cameras[19])-h['pixel']))
                comparisons.append(row)
        common=[r for r in comparisons if r['fraction']==frac and r['new_error_px'] is not None]
        selected=[d for d in detections if d['fraction']==frac]
        summaries.append(dict(fraction=frac,training_candidates=sum(d['status']=='candidate' and d['frame']<977 for d in selected),evaluation_candidates=sum(d['status']=='candidate' and d['frame']>=977 for d in selected),common_comparisons=len(common),
            baseline_rmse_px=float(np.sqrt(np.mean([r['baseline_error_px']**2 for r in common]))) if common else None,new_rmse_px=float(np.sqrt(np.mean([r['new_error_px']**2 for r in common]))) if common else None))
    report=dict(scope='Boundary of thresholded brightness region is only a ball-outline candidate. Shared provisional camera rig, not independent truth. No speed-target selection.',detections=detections,comparisons=comparisons,summaries=summaries,sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(sources)})
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(summaries))

if __name__=='__main__':run()
