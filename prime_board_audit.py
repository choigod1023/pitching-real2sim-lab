"""Execute new-frame board checks with unchanged upstream camera parameters."""
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from fetch_prime_calibration import sha

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
OUT=ROOT/'artifacts/prime-camera'
FRAMES=[1013,1413,1813,2213,2613,3013,7413,7813,8213]


def cameras():
    base=LAB/'openbiomechanics/computer_vision/calibration/results'
    a=json.loads((base/'calibration_results.json').read_text())['calibrations']
    b=json.loads((base/'optitrack_rig_provisional.json').read_text())['camera_poses']
    result={}
    for c in [17,20,21]:
        k=a[f'optitrack_{c}'];r=b[str(c)]
        if np.any(k['distortion']):raise ValueError('Unexpected distorted source model')
        result[c]=(np.array([[k['fx_px'],0,k['cx_px']],[0,k['fy_px'],k['cy_px']],[0,0,1.]]),
                   np.array(r['reference_to_camera_rotation_matrix']),np.array(r['reference_to_camera_translation_mm'])/1000.)
    return result


def projection(points,camera):
    k,r,t=camera;view=np.asarray(points)@r.T+t
    if np.any(view[:,2]<=0):raise ValueError('Point behind camera')
    h=view@k.T;return h[:,:2]/h[:,2:]


def triangulate(left,right,camera1,camera2):
    left,right=np.asarray(left,float),np.asarray(right,float)
    if left.shape!=right.shape or left.ndim!=2 or left.shape[1]!=2 or not np.isfinite([left,right]).all():raise ValueError('Invalid image points')
    matrices=[k@np.column_stack([r,t]) for k,r,t in [camera1,camera2]]
    h=cv2.triangulatePoints(*matrices,left.T,right.T).T
    if np.any(np.abs(h[:,3])<1e-12):raise ValueError('Degenerate triangulation')
    points=h[:,:3]/h[:,3,None]
    residual=np.r_[np.linalg.norm(projection(points,camera1)-left,axis=1),np.linalg.norm(projection(points,camera2)-right,axis=1)]
    return points,residual


def detect():
    cache=OUT/'prime_board_detections.json';rows=json.loads(cache.read_text()) if cache.exists() else {}
    manifest=json.loads((OUT/'download_manifest.json').read_text())
    for item in manifest:
        c=item['camera'];p=Path(item['local_path']);assert item['hash_matches_upstream'] and sha(p)==item['expected_sha256']
        cap=cv2.VideoCapture(str(p))
        for frame in FRAMES:
            key=f'{c}_{frame}'
            if key in rows:
                assert rows[key]['source_sha256']==item['expected_sha256'];continue
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
            row={'camera':c,'frame':frame,'source_sha256':item['expected_sha256'],'detected':False}
            if not ok:row['reason']='decode_failed'
            else:
                assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(frame+1))<.01
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);scale=min(1.,720./gray.shape[1])
                small=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                ok,corners=cv2.findChessboardCornersSB(small,(7,4),cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
                if ok:
                    corners=corners.astype(np.float32)/scale
                    cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,.001))
                    row.update(detected=True,corners=corners.reshape(-1,2).tolist())
                else:row['reason']='board_not_detected'
            rows[key]=row;cache.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
        cap.release()
    return rows


def lengths(points):
    p=np.asarray(points).reshape(4,7,3)
    edges=np.r_[np.linalg.norm(np.diff(p,axis=0),axis=2).ravel(),np.linalg.norm(np.diff(p,axis=1),axis=2).ravel()]*1000
    diagonals=np.r_[np.linalg.norm(p[1:,1:]-p[:-1,:-1],axis=2).ravel(),np.linalg.norm(p[1:,:-1]-p[:-1,1:],axis=2).ravel()]*1000
    return {'count':len(edges),'mean_mm':float(edges.mean()),'mae_from100mm':float(np.abs(edges-100).mean()),'max_abs_error_mm':float(np.abs(edges-100).max()),'all_edge_lengths_mm':edges.tolist(),
            'diagonal_count':len(diagonals),'diagonal_expected_mm':float(100*np.sqrt(2)),
            'diagonal_mae_mm':float(np.abs(diagonals-100*np.sqrt(2)).mean()),'all_diagonal_lengths_mm':diagonals.tolist()}


def run():
    obs=detect();cam=cameras();rows=[]
    for a,b in itertools.combinations(cam,2):
        for f in FRAMES:
            left,right=obs[f'{a}_{f}'],obs[f'{b}_{f}'];row={'cameras':[a,b],'frame':f,'status':'excluded'}
            if not left['detected'] or not right['detected']:row['reason']='missing_detection'
            else:
                alternatives=[]
                for reverse in [False,True]:
                    uv=np.array(right['corners']);uv=uv[::-1] if reverse else uv
                    try:
                        points,residual=triangulate(left['corners'],uv,cam[a],cam[b])
                        alternatives.append({'reverse180':reverse,'mean_reprojection_px':float(residual.mean()),'points':points,'lengths':lengths(points)})
                    except ValueError as e:alternatives.append({'reverse180':reverse,'error':str(e)})
                valid=[r for r in alternatives if 'points' in r]
                if valid:
                    best=min(valid,key=lambda r:r['mean_reprojection_px']);row.update(status='evaluated',reverse180=best['reverse180'],mean_reprojection_px=best['mean_reprojection_px'],lengths=best['lengths'])
                    if f==1013:
                        for camera,r in [(a,left),(b,right)]:
                            cap=cv2.VideoCapture(str(OUT/f'opti_dynamic_checkerboard_cam{camera}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();cap.release()
                            if ok:
                                cv2.drawChessboardCorners(im,(7,4),np.float32(r['corners']).reshape(-1,1,2),True);cv2.imwrite(str(OUT/f'board{camera}_{f}_detections.png'),im)
                else:row['reason']='all_correspondence_hypotheses_invalid'
                row['alternatives']=[{k:v for k,v in r.items() if k!='points'} for r in alternatives]
            rows.append(row)
    evaluated=[r for r in rows if r['status']=='evaluated']
    result={'protocol_sha256':sha(ROOT/'PRIME_CAMERA_PROTOCOL.md'),'observations_sha256':sha(OUT/'prime_board_detections.json'),'frames':FRAMES,'rows':rows,
            'evaluated_pairs':len(evaluated),'excluded_pairs':len(rows)-len(evaluated),'camera_parameters_modified':False,
            'independent_target':False,'actual_speed_gain_kmh':None,'scientific_stage_passed':False}
    (ROOT/'prime_board_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'evaluated':len(evaluated),'excluded':len(rows)-len(evaluated),'per_pair':{f'{a}_{b}':{'n':len(x:=[r for r in evaluated if r['cameras']==[a,b]]),'edge_mae_mm':float(np.mean([r['lengths']['mae_from100mm'] for r in x])) if x else None,'reprojection_px':float(np.mean([r['mean_reprojection_px'] for r in x])) if x else None} for a,b in itertools.combinations(cam,2)}}))
    return result


if __name__=='__main__':run()
