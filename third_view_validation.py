"""Fresh camera17 ball observations score fixed camera20/21 reconstruction."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from ball_trajectory_fit import extract,triangulate,fit_trajectory

SEEDS={973:(475,430),974:(467,420),975:(456,410),976:(447,397),
       977:(435,385),978:(428,373),979:(417,361)}


def cameras(calibration,rig,ids):
    output=[]
    for c in ids:
        intr=calibration[f'optitrack_{c}'];pose=rig[str(c)]
        if np.any(intr['distortion']):raise ValueError('Nonzero distortion unsupported')
        k=np.array([[intr['fx_px'],0,intr['cx_px']],[0,intr['fy_px'],intr['cy_px']],[0,0,1.]])
        p=np.column_stack([pose['reference_to_camera_rotation_matrix'],pose['reference_to_camera_translation_mm']])
        output.append((k,p))
    return output


def run():
    root=Path(__file__).resolve().parent;lab=root.parent;out=root/'artifacts/release-window'
    calroot=lab/'openbiomechanics/computer_vision/calibration/results'
    cal=json.loads((calroot/'calibration_results.json').read_text())['calibrations']
    rig=json.loads((calroot/'optitrack_rig_provisional.json').read_text())['camera_poses']
    c17=cameras(cal,rig,[17])[0];allcams=cameras(cal,rig,[20,21,17])
    oldpath=lab/'speed-journey/artifacts/release-observation/candidate_audit.json'
    old=json.loads(oldpath.read_text());detections=[]
    for th in [100,130,160]:
        for frame,seed in SEEDS.items():
            row=extract(Image.open(out/f'camera17_{frame}.png'),seed,th,'component')
            detections.append(dict(**row,camera=17,frame=frame,threshold=th,seed=seed))
    cases=[]
    for th in [60,80,100]:
        tri={r['frame']:r for r in old['conditional_triangulation'] if r['threshold']==th}
        olddet={(r['camera'],r['frame']):r for r in old['detections'] if r['threshold']==th}
        for th17 in [100,130,160]:
            selected=[r for r in detections if r['threshold']==th17]
            rows=[];valid=[]
            for r in selected:
                frame=r['frame'];k,p=c17
                xc=p@np.r_[tri[frame]['position_camera19_mm'],1.];pred=k@xc;pred=pred[:2]/pred[2]
                row=dict(frame=frame,status=r['status'],pixel=r['pixel'],predicted_pixel=pred.tolist())
                if r['pixel'] is not None:
                    row['heldout_error_px']=float(np.linalg.norm(pred-r['pixel']))
                if (r['status']=='candidate' and tri[frame]['positive_depth']
                        and all(olddet[(c,frame)]['status']=='candidate' for c in [20,21])):
                    pix=[olddet[(20,frame)]['pixel'],olddet[(21,frame)]['pixel'],r['pixel']]
                    x,res,depth=triangulate(pix,allcams)
                    row.update(three_camera_mm=x.tolist(),three_camera_reprojection_px=res,positive_depth=bool(min(depth)>0))
                    valid.append(row)
                rows.append(row)
            case=dict(source_threshold=th,threshold17=th17,rows=rows,n_valid=len(valid))
            if valid:
                case['heldout_mean_px']=float(np.mean([r['heldout_error_px'] for r in valid]))
                case['heldout_max_px']=float(max(r['heldout_error_px'] for r in valid))
            if len(valid)>=3:
                times=np.array([r['frame'] for r in valid])/360
                before=np.array([tri[r['frame']]['position_camera19_mm'] for r in valid])
                after=np.array([r['three_camera_mm'] for r in valid])
                case['two_camera_conditional']=fit_trajectory(times,before)
                case['three_camera_conditional']=fit_trajectory(times,after)
            # Same image-inspected free-flight frames in every finite condition.
            # Extra follow-up after initial all-valid-frame comparison; not a new preregistration.
            fixed=[r for r in valid if r['frame'] in [977,978,979]]
            if len(fixed)==3:
                times=np.array([r['frame'] for r in fixed])/360
                case['same_frames_977_979']=dict(
                    two_camera=fit_trajectory(times,np.array([tri[r['frame']]['position_camera19_mm'] for r in fixed])),
                    three_camera=fit_trajectory(times,np.array([r['three_camera_mm'] for r in fixed])))
            cases.append(case)
    paths=[oldpath,root/'THIRD_VIEW_PROTOCOL.md',out/'camera17.mp4']+[out/f'camera17_{f}.png' for f in SEEDS]
    report=dict(scope='Fresh view consistency, no hand fit or radar reference; not independent true speed accuracy.',
        detections=detections,cases=cases,
        visual_transition=dict(last_ambiguous_frame=975,first_clear_separation_frame=976,
            definition='Single-reader camera17 image separation; physical last contact not established'),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'third_view_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in cases],indent=2))
    return report


if __name__=='__main__':run()
