"""Image-only PVC centre-line intersection audit; no model/speed fitting."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from edge_calibration import OUT

def ridge_line(gray,p,q,window=8):
    p,q=np.array(p,float),np.array(q,float)
    direction=(q-p)/np.linalg.norm(q-p); normal=np.array([-direction[1],direction[0]])
    points=[]; profiles=[]
    for t in np.r_[np.linspace(.16,.39,20),np.linspace(.61,.84,20)]:
        centre=p*(1-t)+q*t; offsets=np.linspace(-window,window,window*8+1); loc=centre+offsets[:,None]*normal
        v=map_coordinates(gray.astype(float),[loc[:,1],loc[:,0]],order=1,mode='nearest')
        threshold=v.min()+.65*(v.max()-v.min()); mask=v>=threshold
        # Only a contiguous bright ridge nearest the seeded centre, no isolated background peaks.
        runs=np.split(np.where(mask)[0],np.where(np.diff(np.where(mask)[0])>1)[0]+1)
        runs=[r for r in runs if len(r)>0]
        run=min(runs,key=lambda r:abs(offsets[r].mean()))
        shift=float(np.mean(offsets[run]));points.append(centre+shift*normal)
        profiles.append({'t':float(t),'shift_px':shift,'contrast':float(v.max()-v.min()),'touch_boundary':bool(run[0]==0 or run[-1]==len(offsets)-1)})
    pts=np.array(points);vx,vy,x,y=cv2.fitLine(pts.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
    n=np.array([-vy,vx],float);b=float(n@np.array([x,y]));rms=float(np.sqrt(np.mean((pts@n-b)**2)))
    return n,b,rms,pts,profiles

def main(window=8):
    src=Path('edge_target_annotations.json');a=json.loads(src.read_text());log={}
    for i,obs in a['cameras'].items():
        im=cv2.imread(str(OUT/f'edgertronic_static_cube_{i}_f120.png'));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        lines={}; annotations=obs['fit'];edges=[]
        for j,p in annotations.items():
            for k,q in annotations.items():
                if j>=k:continue
                v=np.array(a['cube_coordinates'][j])-a['cube_coordinates'][k]
                if np.count_nonzero(v)!=1:continue
                n,b,rms,pts,profile=ridge_line(gray,p,q,window);lines[j,k]=(n,b);edges.append({'ids':[j,k],'line_rms_px':rms,'profiles':profile})
                for uv in pts:cv2.circle(im,tuple(np.rint(uv).astype(int)),1,(0,255,255),-1)
        shifts={}
        for j,p in list(annotations.items()):
            relevant=[line for ids,line in lines.items() if j in ids]
            if len(relevant)<2:continue
            n=np.array([r[0] for r in relevant]);b=np.array([r[1] for r in relevant]);q=np.linalg.lstsq(n,b,rcond=None)[0]
            shifts[j]={'original':p,'centreline_intersection':q.tolist(),'shift_px':float(np.linalg.norm(q-p))}
            annotations[j]=q.tolist();cv2.circle(im,tuple(np.rint(q).astype(int)),4,(0,255,0),1)
            cv2.putText(im,j,tuple(np.rint(q+[5,-5]).astype(int)),0,.4,(0,255,0),1)
        log[i]={'edges':edges,'corner_shifts':shifts}
        cv2.imwrite(str(OUT/f'cube{i}_centreline_overlay.png'),im)
    a['annotation_round']=2;a['refinement_window_px']=window;a['method']+=' Round 2: image-only bright PVC centre-line intersections where >=2 incident visible edges exist; full profile/shift logs retained. Rejected as final calibration: held-out checks failed.'
    Path('edge_target_annotations_refined.json').write_text(json.dumps(a,indent=2));(OUT/'centreline-refinement-log.json').write_text(json.dumps(log,indent=2))
    print(json.dumps({i:{'max_shift_px':max(x['shift_px'] for x in c['corner_shifts'].values()),'line_rms_range_px':[min(e['line_rms_px'] for e in c['edges']),max(e['line_rms_px'] for e in c['edges'])]} for i,c in log.items()},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--window',type=int,choices=[8,16],default=8);main(p.parse_args().window)
