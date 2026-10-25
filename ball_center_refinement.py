"""Image-only baseball centroid refinement; thresholds are not radar tuned."""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
from ball_trajectory_fit import load_cameras,triangulate,fit_trajectory

def image_seed(camera,frame):
    d=frame-985
    return (645-23.8*d-.17*d*d,565+6.2*d+.13*d*d) if camera==19 else (804+17.7*d,286-2.3*d)

def center(image,background,seed,threshold=20,method='weighted',radius=32):
    a=np.asarray(image,dtype=float);b=np.asarray(background,dtype=float)
    if a.ndim==3:a=a.mean(axis=2)
    if b.ndim==3:b=b.mean(axis=2)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite equal-size images required')
    if not np.isfinite(threshold) or threshold<=0 or not isinstance(radius,(int,np.integer)) or radius<4:raise ValueError('Finite positive threshold and integer radius >=4 required')
    if method not in ('weighted','binary','gaussian'):raise ValueError('Unknown estimator')
    cx,cy=seed;x0,y0=int(round(cx))-radius,int(round(cy))-radius
    x1,y1=x0+radius*2+1,y0+radius*2+1
    if min(x0,y0)<0 or x1>a.shape[1] or y1>a.shape[0]:return {'status':'image_boundary','pixel':None}
    delta=a[y0:y1,x0:x1]-b[y0:y1,x0:x1];lab,n=ndimage.label(delta>threshold,np.ones((3,3)))
    candidates=[]
    for k in range(1,n+1):
        yy,xx=np.where(lab==k)
        if len(xx)<12:continue
        candidates.append((np.linalg.norm([xx.mean()+x0-cx,yy.mean()+y0-cy]),k,xx,yy))
    if not candidates:return {'status':'no_component','pixel':None}
    _,k,x,y=min(candidates,key=lambda z:z[0]);weights=np.maximum(delta[y,x],0) if method!='binary' else np.ones(len(x))
    pos=np.array([np.average(x,weights=weights),np.average(y,weights=weights)])
    boundary=bool((x==0).any() or (y==0).any() or (x==2*radius).any() or (y==2*radius).any())
    result={'status':'roi_boundary' if boundary else 'candidate','pixel':(pos+[x0,y0]).tolist(),'area_px':int(len(x)),'bbox_wh':[int(np.ptp(x)+1),int(np.ptp(y)+1)]}
    if method=='gaussian' and not boundary:
        # Fit localized contrast. A rotated Gaussian describes blur, not physical ball outline.
        Y,X=np.indices(delta.shape);support=ndimage.binary_dilation(lab==k,iterations=3);xx=X[support];yy=Y[support];zz=delta[support]
        def model(p):
            dx=xx-p[0];dy=yy-p[1];u=np.cos(p[4])*dx+np.sin(p[4])*dy;v=-np.sin(p[4])*dx+np.cos(p[4])*dy
            return p[5]*np.exp(-.5*((u/p[2])**2+(v/p[3])**2))+p[6]
        p0=[*pos,8,4,0,max(float(delta[y,x].max()),1),0]
        fit=least_squares(lambda p:model(p)-zz,p0,bounds=([pos[0]-10,pos[1]-10,1,1,-np.pi,1,-50],[pos[0]+10,pos[1]+10,20,20,np.pi,300,50]),max_nfev=150)
        result.update(pixel=(fit.x[:2]+[x0,y0]).tolist(),gaussian_rmse=float(np.sqrt(np.mean(fit.fun**2))),gaussian_success=bool(fit.success))
        if not fit.success or np.any(np.isclose(fit.x[:2],np.r_[pos-10],atol=.01)) or np.any(np.isclose(fit.x[:2],pos+10,atol=.01)):result['status']='gaussian_fit_failure'
    return result

def run(directory,calibration):
    directory=Path(directory);cameras=load_cameras(Path(calibration));detections=[];cases=[]
    backgrounds={c:np.median(np.stack([np.asarray(Image.open(directory/f'camera{c}_frame{f}.png')).mean(axis=2) for f in range(970,1016)]),axis=0) for c in (19,22)}
    for method in ('binary','weighted','gaussian'):
      for th in (10,20,30):
        pix=[]
        for f in range(985,1002):
          pair=[]
          for c in (19,22):
            r=center(Image.open(directory/f'camera{c}_frame{f}.png'),backgrounds[c],image_seed(c,f),th,method)
            r.update(camera=c,frame=f,threshold=th,method=method);detections.append(r);pair.append(r['pixel'] if r['status']=='candidate' else None)
          pix.append(pair)
        for last in (990,993,995,999,1001):
          chosen=pix[:last-985+1];case={'method':method,'threshold':th,'start':985,'last':last,'camera19_intrinsic_coverage_extrapolation':last>=994,'coverage_note':'Companion camera audit found camera19 pixels outside intrinsic calibration observations from frame994 onward; flags do not establish accuracy inside coverage'}
          if any(p is None for pair in chosen for p in pair):case['status']='invalid_detection'
          else:
            xyz=np.array([triangulate(p,cameras)[0] for p in chosen]);t=np.arange(985,last+1)/360
            fit=fit_trajectory(t,xyz);loo=[fit_trajectory(np.delete(t,i),np.delete(xyz,i,axis=0))['midpoint_speed_kmh'] for i in range(len(t))]
            case.update(status='conditional',**fit,loo_speed_range_kmh=[min(loo),max(loo)],mean_reprojection_px=float(np.mean([triangulate(p,cameras)[1] for p in chosen])),positive_depth=bool(min(min(triangulate(p,cameras)[2]) for p in chosen)>0))
          cases.append(case)
    # Same-frame comparison isolates estimator changes from simply adding frames.
    baselines=[];window_cases=[]
    for baseline_last in (993,999):
      for th in (70,100,130):
        pairs=[];flags=[]
        for f in range(985,baseline_last+1):
          pair=[]
          for c in (19,22):
            r=center(Image.open(directory/f'camera{c}_frame{f}.png'),np.zeros_like(backgrounds[c]),image_seed(c,f),th,'binary',32);pair.append(r['pixel'])
            if r['status']!='candidate':flags.append([c,f,r['status']])
          pairs.append(pair)
        row={'threshold':th,'flags':flags,'start':985,'last':baseline_last}
        if not flags:row.update(fit_trajectory(np.arange(985,baseline_last+1)/360,np.array([triangulate(p,cameras)[0] for p in pairs])))
        baselines.append(row)
    for radius in (24,32,40):
      pairs=[];flags=[]
      for f in range(985,1000):
        pair=[]
        for c in (19,22):
          r=center(Image.open(directory/f'camera{c}_frame{f}.png'),backgrounds[c],image_seed(c,f),20,'weighted',radius);pair.append(r['pixel'])
          if r['status']!='candidate':flags.append([c,f,r['status']])
        pairs.append(pair)
      row={'radius':radius,'flags':flags,'start':985,'last':999}
      if not flags:row.update(fit_trajectory(np.arange(985,1000)/360,np.array([triangulate(p,cameras)[0] for p in pairs])))
      window_cases.append(row)
    report={'same_frame_absolute_brightness_baselines':baselines,'roi_radius_cases':window_cases,'scope':'Image-only centers, provisional calibration, assumed 360 Hz; not real speed accuracy','image_frame_numbers':'zero-based decoded indexes; on-video burned numbers are +1','frames':list(range(985,1002)),'background_frames':list(range(970,1016)),'selection':'Manual visual free-flight visibility; 982-984 shoe overlap excluded;1002 onward bottom clipping in camera19','detections':detections,'cases':cases}
    (directory/'center_tracks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Review every primary frame; no fitted 3D track is used as an image seed.
    montage=Image.new('RGB',(600,100*17))
    for i,f in enumerate(range(985,1002)):
      for j,c in enumerate((19,22)):
        im=Image.open(directory/f'camera{c}_frame{f}.png');s=image_seed(c,f);roi=im.crop((int(s[0])-50,int(s[1])-35,int(s[0])+50,int(s[1])+35)).resize((300,210));d=ImageDraw.Draw(roi)
        for method,color in [('binary','red'),('weighted','lime'),('gaussian','cyan')]:
          r=next(r for r in detections if r['camera']==c and r['frame']==f and r['threshold']==20 and r['method']==method)
          if r['pixel'] is not None:
            x,y=r['pixel'];x=(x-int(s[0])+50)*3;y=(y-int(s[1])+35)*3;d.line((x-5,y,x+5,y),fill=color);d.line((x,y-5,x,y+5),fill=color)
        roi=roi.resize((300,100));ImageDraw.Draw(roi).text((2,2),f'cam{c} frame{f}',fill='yellow');montage.paste(roi,(300*j,100*i))
    montage.save(directory/'centers_review.png')
    return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('calibration');a=p.parse_args();r=run(a.directory,a.calibration);print(json.dumps(r['cases'],indent=2))
