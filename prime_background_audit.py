"""Image-background consistency, separate from 3D calibration accuracy."""
import json
from pathlib import Path
import cv2
import numpy as np
from prime_board_audit import ROOT,OUT
from fetch_prime_calibration import sha


def frame(path,index):
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read();cap.release()
    if not ok:raise ValueError('Could not decode frame')
    return image


def compare(left,right):
    if left.shape!=right.shape:raise ValueError('Different image dimensions')
    cv2.setRNGSeed(0);mask=np.ones(left.shape[:2],np.uint8)*255;mask[200:650,350:1000]=0
    sift=cv2.SIFT_create(nfeatures=3000)
    k1,d1=sift.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),mask)
    k2,d2=sift.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),mask)
    if d1 is None or d2 is None:return {'status':'insufficient_descriptors'}
    matches=[a for a,b in cv2.BFMatcher().knnMatch(d1,d2,k=2) if a.distance<.75*b.distance]
    if len(matches)<8:return {'status':'insufficient_matches','matches':len(matches)}
    p=np.float32([k1[m.queryIdx].pt for m in matches]);q=np.float32([k2[m.trainIdx].pt for m in matches])
    h,inliers=cv2.findHomography(p,q,cv2.RANSAC,2.,maxIters=2000)
    if h is None:return {'status':'homography_failed','matches':len(matches)}
    selected=inliers.ravel().astype(bool);d=np.linalg.norm(q[selected]-p[selected],axis=1)
    predicted=cv2.perspectiveTransform(p.reshape(-1,1,2),h).reshape(-1,2)
    residual=np.linalg.norm(predicted[selected]-q[selected],axis=1)
    return {'status':'evaluated','features':[len(k1),len(k2)],'ratio_matches':len(matches),'ransac_inliers':int(selected.sum()),
            'inlier_fraction':float(selected.mean()),'raw_inlier_displacement_median_px':float(np.median(d)),
            'raw_inlier_displacement_p95_px':float(np.percentile(d,95)),
            'all_ratio_match_displacement_median_px':float(np.median(np.linalg.norm(q-p,axis=1))),
            'post_homography_residual_median_px':float(np.median(residual)),
            'homography':h.tolist(),'matched_background_is_not_pose_ground_truth':True}


def run():
    result={'protocol_sha256':sha(ROOT/'BACKGROUND_DIAGNOSTIC_PROTOCOL.md'),'rows':[],'applied_to_pitching':False}
    for c in [17,20,21]:
        cube=cv2.imread(str(OUT/f'opti_cube_cam{c}.png'))
        pitching=ROOT.parent/('release-window-validation-20261028/artifacts/release-window/camera17.mp4' if c==17 else f'speed-journey/artifacts/release-observation/camera{c}.mp4')
        for label,path,index in [('checkerboard',OUT/f'opti_dynamic_checkerboard_cam{c}.mp4',1013),('baseball',pitching,940)]:
            result['rows'].append({'camera':c,'comparison':'cube0_to_'+label+str(index),'source_path':str(path),'source_sha256':sha(path),**compare(cube,frame(path,index))})
    (ROOT/'prime_background_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k in ['camera','comparison','ransac_inliers','raw_inlier_displacement_median_px','raw_inlier_displacement_p95_px','inlier_fraction','status']} for r in result['rows']]))
    return result


if __name__=='__main__':run()
