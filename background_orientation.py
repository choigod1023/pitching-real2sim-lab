"""Masked room features as an independent camera-rotation diagnostic."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from edge_calibration import camera_parameters,fit_camera,register_cube
from dense_camera_link import ROOT

ASSETS=Path('../edge-calibration-20261029/artifacts/edge-calibration')
BOXES={1:(590,630,1010,1000),3:(320,520,710,910),4:(480,260,790,590)}

def rotation_difference(a,b):return float(np.degrees(np.linalg.norm(cv2.Rodrigues(a@b.T)[0])))

def ratio_matches(a,b):
    if a is None or b is None or len(a)==0 or len(b)<2:return {}
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(a,b,k=2)
    return {m.queryIdx:m.trainIdx for row in pairs if len(row)==2 for m,n in [row] if m.distance<.75*n.distance}

def run():
    cv2.setNumThreads(1);cv2.setRNGSeed(0);source=json.loads(Path('edge_intrinsics_frozen.json').read_text());a=json.loads(Path('edge_target_annotations.json').read_text());sift=cv2.SIFT_create(nfeatures=5000);out=[];local={}
    for i in [1,3,4]:
        obs=a['cameras'][str(i)];obj=np.array([a['cube_coordinates'][j] for j in obs['fit']])*a['cube_side_mm'];local[i]=fit_camera(obj,list(obs['fit'].values()),*camera_parameters(source,i))
    for frame in [0,120,240]:
        features={};hashes={}
        for i in [1,3,4]:
            path=ASSETS/f'edgertronic_static_cube_{i}_f{frame}.png';im=cv2.imread(str(path),0);hashes[i]=hashlib.sha256(path.read_bytes()).hexdigest();mask=np.full(im.shape,255,np.uint8);x,y,xx,yy=BOXES[i];mask[y:yy,x:xx]=0;k,d=sift.detectAndCompute(im,mask);features[i]=(k,d)
        for other in [3,4]:
            ka,da=features[1];kb,db=features[other];forward=ratio_matches(da,db);reverse=ratio_matches(db,da);matches=[(i,j) for i,j in forward.items() if reverse.get(j)==i];record={'frame':frame,'pair':[1,other],'source_image_hashes':{i:hashes[i] for i in [1,other]},'mutual_matches':len(matches)}
            if len(matches)<20:record['status']='insufficient_matches';out.append(record);continue
            ua=np.array([ka[i].pt for i,j in matches]);ub=np.array([kb[j].pt for i,j in matches]);K1,D1=camera_parameters(source,1);K2,D2=camera_parameters(source,other);na=cv2.undistortPoints(ua.reshape(-1,1,2),K1,D1).reshape(-1,2);nb=cv2.undistortPoints(ub.reshape(-1,1,2),K2,D2).reshape(-1,2);focal=(K1[0,0]+K2[0,0])/2
            e,mask=cv2.findEssentialMat(na,nb,np.eye(3),method=cv2.RANSAC,prob=.999,threshold=1/focal);_,hm=cv2.findHomography(ua,ub,cv2.RANSAC,2.);record['homography_inlier_fraction']=float(hm.mean()) if hm is not None else None
            if e is None:record['status']='essential_failed';out.append(record);continue
            candidates=[]
            for matrix in np.asarray(e).reshape(-1,3,3):
                n,r,t,m=cv2.recoverPose(matrix,na,nb,np.eye(3),mask=mask.copy());candidates.append((n,r,t,m))
            n,r,t,m=max(candidates,key=lambda x:x[0]);record.update(essential_inliers=int(np.count_nonzero(mask)),cheirality_inliers=int(n),R=r.tolist(),translation_direction=t.ravel().tolist())
            phase_deltas={}
            for phase in range(4):
                c=register_cube(local[other],phase,698.5);relative=c['R']@local[1]['R'].T;phase_deltas[str(phase)]=rotation_difference(r,relative)
            record['cube_phase_rotation_difference_deg']=phase_deltas;record['status']='descriptive_only' if np.count_nonzero(mask)>=20 and n>=15 and (record['homography_inlier_fraction'] is None or record['homography_inlier_fraction']<.9) else 'insufficient_or_planar';out.append(record)
    result={'results':out,'metric_scale_verified':False,'scientific_stage_passed':False};(ROOT/'background-orientation-results.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:v.get(k) for k in ['pair','frame','mutual_matches','essential_inliers','cheirality_inliers','homography_inlier_fraction','cube_phase_rotation_difference_deg','status']} for v in out],indent=2));return result
if __name__=='__main__':run()
