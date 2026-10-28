"""Frame-domain 2D ball observations in newly acquired videos; not speed prediction."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image

CASES={67:dict(first=375,center=[472,796],step=[-42,-6]),
       70:dict(first=455,center=[497,809],step=[-44,-6])}


def bright_component(image,seed,threshold,half_size=(48,32)):
    a=np.asarray(image);cx,cy=map(int,seed);hw,hh=half_size
    x0,y0=cx-hw,cy-hh;x1,y1=cx+hw+1,cy+hh+1
    if min(x0,y0)<0 or x1>a.shape[1] or y1>a.shape[0]:return dict(status='roi_outside',pixel=None)
    gray=a[y0:y1,x0:x1,:3].min(axis=2) if a.ndim==3 else a[y0:y1,x0:x1]
    n,l,st,centers=cv2.connectedComponentsWithStats((gray>threshold).astype(np.uint8),8)
    candidates=[i for i in range(1,n) if st[i,4]>=5]
    if not candidates:return dict(status='no_component',pixel=None)
    i=min(candidates,key=lambda k:np.linalg.norm(centers[k]+[x0,y0]-seed));x,y,w,h,area=map(int,st[i])
    boundary=x==0 or y==0 or x+w==x1-x0 or y+h==y1-y0
    return dict(status='roi_boundary' if boundary else 'candidate',pixel=(centers[i]+[x0,y0]).tolist(),
                area=area,bbox=[x+x0,y+y0,w,h])


def frame_slope(frames,points):
    f=np.asarray(frames,float);p=np.asarray(points,float)
    if f.ndim!=1 or len(f)<3 or p.shape!=(len(f),2) or not np.isfinite(f).all() or not np.isfinite(p).all() or np.any(np.diff(f)<=0):
        raise ValueError('Finite increasing frames and matched2D points required')
    a=np.column_stack([np.ones(len(f)),f-f.mean()]);coef=np.linalg.lstsq(a,p,rcond=None)[0]
    return dict(velocity_px_per_frame=coef[1].tolist(),speed_px_per_frame=float(np.linalg.norm(coef[1])),
                position_rmse_px=float(np.sqrt(np.mean(np.sum((p-a@coef)**2,axis=1)))),speed_kmh=None)


def run():
    root=Path(__file__).resolve().parent;out=root/'artifacts/paired-source';cases=[];detections=[];sources=[]
    for label,definition in CASES.items():
        path=out/f'edgertronic_dynamic_baseballthrow{label}mph_3.mp4';cap=cv2.VideoCapture(str(path));assert cap.isOpened()
        sources.append(dict(trial=f'edge_filename_{label}',camera=3,path=str(path),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),playback_fps=cap.get(cv2.CAP_PROP_FPS)))
        cap.set(cv2.CAP_PROP_POS_FRAMES,definition['first']);frames={}
        for f in range(definition['first'],definition['first']+5):
            ok,bgr=cap.read();assert ok;frames[f]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        cap.release()
        for th in [100,130,160]:
            rows=[]
            for frame,image in frames.items():
                seed=np.array(definition['center'])+(frame-definition['first'])*np.array(definition['step'])
                row=dict(label=label,frame=frame,threshold=th,seed=seed.tolist(),**bright_component(image,seed,th))
                rows.append(row);detections.append(row)
            case=dict(label=label,threshold=th,frames=list(frames),valid_frames=sum(r['status']=='candidate' for r in rows))
            if all(r['status']=='candidate' for r in rows):case['fit']=frame_slope(list(frames),[r['pixel'] for r in rows])
            else:case['fit']=None
            cases.append(case)
    report=dict(scope='New2D ball tracks. File speed labels are not verified measurements; pixel slopes are not3D release speed.',
       sources=sources,cases=cases,detections=detections,
       capture_fps_declared_by_workbook=360,playback_fps_observed=30,
       physical_capture_timing_independently_verified=False,scientific_stage_passed=False,
       protocol_sha256=hashlib.sha256((root/'NEW_PAIRED_SOURCE_PROTOCOL.md').read_bytes()).hexdigest())
    (out/'new_ball_observations.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(cases,indent=2));return report


if __name__=='__main__':run()
