"""Fixed contributor transform audit; never fits alignment or selects offsets."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SEGMENTS=['l_thigh','r_thigh','l_shank','r_shank','l_foot','r_foot','l_uarm','r_uarm','l_larm','r_larm','l_hand','r_hand']
OFFSETS=(0,1,2)

def similarity(points, alignment):
    return alignment['scale'] * (np.asarray(points) @ np.asarray(alignment['rotation']).T) + alignment['translation']

def sample_origins(origins, video_frames, offset):
    frames=np.asarray(video_frames)+offset
    if not np.issubdtype(frames.dtype,np.integer):
        raise ValueError('Audit offsets and frame indices must be integers')
    if np.any(frames<0) or np.any(frames>=len(origins)):
        raise IndexError('C3D sample outside available range')
    return origins[frames]

def project(points, pose, intrinsic):
    points=np.asarray(points)
    xyz=points@np.asarray(pose['reference_to_camera_rotation_matrix']).T + np.asarray(pose['reference_to_camera_translation_mm'])/1000
    if not np.allclose(intrinsic['distortion'],0):
        raise ValueError('This projection audit explicitly supports supplied zero distortion only')
    if np.any(xyz[...,2]<=0): raise ValueError('Non-positive camera depth')
    return np.stack([intrinsic['fx_px']*xyz[...,0]/xyz[...,2]+intrinsic['cx_px'],intrinsic['fy_px']*xyz[...,1]/xyz[...,2]+intrinsic['cy_px']],axis=-1)

def stats(a):
    a=np.asarray(a)
    return dict(n=int(a.size),median=float(np.median(a)),mean=float(a.mean()),rmse=float(np.sqrt(np.mean(a*a))),p95=float(np.percentile(a,95)),max=float(a.max()))

def main():
    import ezc3d
    root=Path(__file__).resolve().parent
    up=root.parent/'openbiomechanics/computer_vision'
    paths=dict(keypoints=up/'4d_reconstruction/release_keypoints.csv',alignment=up/'4d_reconstruction/theia_alignment.json',c3d=root.parent/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d',intrinsics=up/'calibration/results/calibration_results.json',rig=up/'calibration/results/optitrack_rig_provisional.json')
    rows=list(csv.DictReader(paths['keypoints'].open()))
    frames=sorted({int(r['video_frame']) for r in rows})
    lookup={(int(r['video_frame']),r['segment']):[float(r[a]) for a in ('x_m','y_m','z_m')] for r in rows}
    target=np.array([[lookup[f,s] for s in SEGMENTS] for f in frames])
    align=json.loads(paths['alignment'].read_text())
    c=ezc3d.c3d(str(paths['c3d']))
    labels=c['parameters']['ROTATION']['LABELS']['value']
    idx=[labels.index(s+'_4X4') for s in SEGMENTS]
    origins=c['data']['rotations'][:3,3,idx,:].transpose(2,1,0)
    cal=json.loads(paths['intrinsics'].read_text())['calibrations']; poses=json.loads(paths['rig'].read_text())['camera_poses']
    out=root/'artifacts/fixed-alignment';out.mkdir(parents=True,exist_ok=True)
    details=[];summary={}
    for offset in OFFSETS:
        pred=similarity(sample_origins(origins,frames,offset),align)
        err=np.linalg.norm(pred-target,axis=-1)*1000
        pixels={cam:np.linalg.norm(project(pred,pose,cal['optitrack_'+cam])-project(target,pose,cal['optitrack_'+cam]),axis=-1) for cam,pose in poses.items()}
        summary[str(offset)]={'all_mm':stats(err),'segments_mm':{s:stats(err[:,i]) for i,s in enumerate(SEGMENTS)},'cameras_projected_discrepancy_px':{cam:{'all':stats(e),'l_hand':stats(e[:,10])} for cam,e in pixels.items()},'per_frame_mm':{str(f):{'all_median':float(np.median(err[j])),'l_hand':float(err[j,10]),'l_larm':float(err[j,8]),'r_hand':float(err[j,11])} for j,f in enumerate(frames)}}
        for j,f in enumerate(frames):
            for i,s in enumerate(SEGMENTS):
                details.append(dict(offset=offset,video_frame=f,c3d_array_index=f+offset,segment=s,error_mm=err[j,i],**{f'd{a}_mm':float((pred-target)[j,i,k]*1000) for k,a in enumerate('xyz')},**{f'camera{cam}_projected_discrepancy_px':float(e[j,i]) for cam,e in pixels.items()}))
    report={'scope':'Fixed supplied similarity. Reference 3D keypoints are triangulated contributor data, possible fit overlap. Camera pixel values compare two projected 3D locations, NOT residuals to independent observed 2D annotations. No synchronization or accuracy claim.','offsets_predefined':OFFSETS,'nominal_offset':1,'video_frames':frames,'scale_relative_to_mm_to_m':align['scale']/0.001,'inputs':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},'summary':summary}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'residuals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print(json.dumps({o:{'all_mm':v['all_mm'],'l_hand_mm':v['segments_mm']['l_hand'],'l_larm_mm':v['segments_mm']['l_larm'],'r_hand_mm':v['segments_mm']['r_hand']} for o,v in summary.items()},indent=2))
if __name__=='__main__': main()
