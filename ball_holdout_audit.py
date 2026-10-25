"""Third-view check of fixed cameras20/21 ball candidates; no refit or speed target."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from ball_trajectory_fit import extract, seed


def project(point_mm, intrinsic, extrinsic):
    xyz=np.asarray(extrinsic,dtype=float)@np.r_[np.asarray(point_mm,dtype=float),1.]
    if not np.isfinite(xyz).all() or xyz[2]<=0:
        raise ValueError('Point must have finite positive depth')
    uv=np.asarray(intrinsic,dtype=float)@xyz
    return uv[:2]/uv[2]


def run(root=None):
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    lab=root.parent;out=root/'artifacts/heldout-ball';out.mkdir(parents=True,exist_ok=True)
    source=lab/'speed-journey/artifacts/release-observation/candidate_audit.json'
    calpath=lab/'openbiomechanics/computer_vision/calibration/results/calibration_results.json'
    posepath=calpath.parent/'optitrack_rig_provisional.json'
    candidates=json.loads(source.read_text())['conditional_triangulation']
    intr=json.loads(calpath.read_text())['calibrations']['optitrack_19']
    pose=json.loads(posepath.read_text())['camera_poses']['19']
    if any(intr['distortion']):raise ValueError('Nonzero distortion needs explicit projection')
    K=np.array([[intr['fx_px'],0,intr['cx_px']],[0,intr['fy_px'],intr['cy_px']],[0,0,1.]])
    P=np.column_stack([pose['reference_to_camera_rotation_matrix'],pose['reference_to_camera_translation_mm']])
    detections=[]; comparisons=[];sources=[source,calpath,posepath]
    for frame in [977,978,979]:
        path=lab/f'tracking-calibration/artifacts/center-refinement/camera19_frame{frame}.png'
        sources.append(path);im=Image.open(path)
        for threshold in [100,130,160]:
            det=extract(im,seed(19,frame),threshold,'local_ellipse')
            det.update(frame=frame,threshold=threshold,camera=19)
            detections.append(det)
            if det['status']!='candidate':continue
            for candidate in [c for c in candidates if c['frame']==frame]:
                uv=project(candidate['position_camera19_mm'],K,P)
                delta=uv-np.array(det['pixel'])
                comparisons.append(dict(frame=frame,source_threshold=candidate['threshold'],holdout_threshold=threshold,
                    heldout_pixel=det['pixel'],projected_pixel=uv.tolist(),residual_xy_px=delta.tolist(),residual_norm_px=float(np.linalg.norm(delta))))
    summaries=[]
    for frame in [977,978,979]:
        rows=[c for c in comparisons if c['frame']==frame]
        values=[r['residual_norm_px'] for r in rows]
        summaries.append(dict(frame=frame,count=len(rows),min_px=min(values) if values else None,max_px=max(values) if values else None,
            median_px=float(np.median(values)) if values else None))
    report=dict(scope='Camera19 image observations not used to triangulate cameras20/21 points. Shared provisional rig is not independent calibration. Matching frame indexes is assumed; no shift or calibration fitted.',
        protocol='Fixed frames977-979 chosen by visible separated-ball outline, source thresholds60/80/100 all reported; camera19 thresholds100/130/160 inherited from earlier image extraction. Image-only seed extrapolated from previous audit, not projected source points.',
        limitations=['Bright-core centroids are not measured true ball centers; motion blur and threshold bias remain.',
                     'Shared calibration and unverified sensor synchronization can produce cross-view disagreement.',
                     'Third-view residual is not a conversion to km/h accuracy or ground truth.',
                     'Only three separated-ball frames; no true contact or release-time claim.'],
        detections=detections,comparisons=comparisons,frame_summary=summaries,
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report

if __name__=='__main__':
    print(json.dumps(run()['frame_summary'],indent=2))
