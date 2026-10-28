"""Inspect actual marker channels versus model segment origins; no outcome fit."""
from pathlib import Path
import hashlib
import json
import csv
import numpy as np
import pandas as pd
import ezc3d
from joint_video import load_joints,elbow_angle
from fixed_alignment_audit import similarity


def measured_markers(c, wanted):
    """Native POINT channels only, converted by declared units; preserve gaps."""
    labels=[s.strip() for s in c['parameters']['POINT']['LABELS']['value']]
    units=c['parameters']['POINT']['UNITS']['value'][0].strip().lower()
    scales={'m':1.,'mm':.001,'cm':.01}
    if units not in scales:raise ValueError('Unknown POINT units')
    if any(labels.count(n)!=1 for n in wanted):raise ValueError('Missing/ambiguous marker')
    idx=[labels.index(n) for n in wanted]
    pts=np.asarray(c['data']['points'])[:3,idx,:].transpose(2,1,0)*scales[units]
    residuals=np.asarray(c['data']['meta_points']['residuals'])[0,idx,:].T
    valid=np.isfinite(pts).all(axis=2)&(residuals>=0)
    return pts,valid


def stats(x):
    x=np.asarray(x,float)
    if x.size==0 or not np.isfinite(x).all():raise ValueError('Finite nonempty data required')
    return dict(min=float(x.min()),median=float(np.median(x)),max=float(x.max()),range=float(np.ptp(x)))


def run():
    root=Path(__file__).resolve().parent;lab=root.parent;out=root/'artifacts/marker-definition';out.mkdir(parents=True,exist_ok=True)
    source=lab/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d'
    c=ezc3d.c3d(str(source));A=c['data']['rotations'];labels=c['parameters']['ROTATION']['LABELS']['value']
    xyz,names,rate=load_joints(source)
    report=dict(prime=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
       physical_point_count=int(c['parameters']['POINT']['USED']['value'][0]),
       point_labels=c['parameters']['POINT']['LABELS']['value'],
       rotation_count=len(labels),rotation_labels=labels,frames=len(xyz),rate_hz=rate,
       point_units=c['parameters']['POINT']['UNITS']['value'],
       theia_parameters={k:np.asarray(v['value']).tolist() for k,v in c['parameters']['THEIA3D'].items() if k!='__METADATA__'}),
       origin_mapping=dict(uarm='estimated proximal shoulder point',larm='estimated proximal elbow point',hand='estimated proximal wrist point; not ball center or fingertip'),
       source_urls=['https://docs.theiamarkerless.com/theia3d-documentation/data-formats/c3d-files',
       'https://www.theiamarkerless.com/blog/working-with-theia-markerless-data-in-visual3d',
       'https://github.com/drivelineresearch/openbiomechanics/blob/main/baseball_pitching/README.md'],
       arms={},angle_check=[],marker_based_trials=[])
    for side in ['l','r']:
        h=A[:,:,labels.index(side+'_hand_4X4')].transpose(2,0,1)
        f=A[:,:,labels.index(side+'_larm_4X4')].transpose(2,0,1)
        d=h[:,:3,3]-f[:,:3,3]
        local=np.einsum('fji,fj->fi',f[:,:3,:3],d)
        speed=np.linalg.norm(np.gradient(xyz[:,names.index(side+'_hand')]/1000,1/rate,axis=0),axis=1)*3.6
        report['arms'][side]=dict(
           upper_arm_length_mm=stats(np.linalg.norm(xyz[:,names.index(side+'_uarm')]-xyz[:,names.index(side+'_larm')],axis=1)),
           forearm_length_mm=stats(np.linalg.norm(d,axis=1)),
           forearm_local_wrist_median_mm=np.median(local,axis=0).tolist(),
           hand_rotation_orthogonality_max=float(np.abs(h[:,:3,:3].transpose(0,2,1)@h[:,:3,:3]-np.eye(3)).max()),
           hand_rotation_det=stats(np.linalg.det(h[:,:3,:3])),
           homogeneous_bottom_row_max=float(np.abs(h[:,3,:]-[0,0,0,1]).max()),
           hand_speed_960_990_kmh=stats(speed[960:991]))
    for frame in [970,974,977,979,986]:
        points=dict(zip(names,xyz[frame+1]));left=elbow_angle(points,'l');right=elbow_angle(points,'r')
        report['angle_check'].append(dict(video_frame=frame,c3d_index=frame+1,
             before_wrong_right_interior_deg=right,after_left_interior_deg=left,
             left_flexion_from_straight_deg=180-left,difference_deg=left-right))
    # Name-side check against supplied keypoints, fixed original alignment.
    up=lab/'openbiomechanics/computer_vision/4d_reconstruction'
    align=json.loads((up/'theia_alignment.json').read_text())
    rows=list(csv.DictReader((up/'release_keypoints.csv').open()))
    namecheck=[]
    for segment,opposite in [('l_hand','r_hand'),('l_larm','r_larm'),('l_uarm','r_uarm')]:
        subset=[r for r in rows if r['segment']==segment];frames=np.array([int(r['video_frame'])+1 for r in subset])
        target=np.array([[float(r[k]) for k in ['x_m','y_m','z_m']] for r in subset])
        def rmse(n):return float(np.sqrt(np.mean(np.sum((similarity(xyz[frames,names.index(n)],align)-target)**2,axis=1)))*1000)
        namecheck.append(dict(label=segment,same_side_rmse_mm=rmse(segment),opposite_side_rmse_mm=rmse(opposite)))
    report['fixed_alignment_side_check']=namecheck
    # Separate marker-based 3034 dataset: never used as Prime same-pitch truth.
    data=lab/'openbiomechanics/baseball_pitching/data';meta=pd.read_csv(data/'metadata.csv')
    selection=meta[meta.session_pitch.astype(str).str.startswith('3034_')].head(5)
    times=pd.read_csv(data/'full_sig/landmarks.csv',usecols=['session_pitch','BR_time']).groupby('session_pitch').BR_time.first()
    for row in selection.itertuples():
        paths=list((data/'c3d').rglob(row.filename_new))
        if len(paths)!=1:raise ValueError('Ambiguous original C3D')
        mc=ezc3d.c3d(str(paths[0]));wanted=['RFIN','RWRA','RWRB','RFRM','LWRA','LWRB']
        p,valid=measured_markers(mc,wanted);hz=float(mc['parameters']['POINT']['RATE']['value'][0])
        t=np.arange(len(p))/hz;mask=(abs(t-float(times.loc[row.session_pitch]))<=.05)&valid.all(axis=1)
        if not mask.any():raise ValueError('No valid release-neighborhood markers')
        wrist=(p[:,1]+p[:,2])/2;left_wrist=(p[:,4]+p[:,5])/2
        report['marker_based_trials'].append(dict(pitch=row.session_pitch,source=str(paths[0]),
             sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),point_units=mc['parameters']['POINT']['UNITS']['value'],
             point_count=len(mc['parameters']['POINT']['LABELS']['value']),BR_time=float(times.loc[row.session_pitch]),
             valid_release_frames=int(mask.sum()),invalid_frames_by_marker={n:int((~valid[:,i]).sum()) for i,n in enumerate(wanted)},
             RFIN_to_right_wrist_mid_mm=stats(np.linalg.norm(p[mask,0]-wrist[mask],axis=1)*1000),
             RFIN_to_left_wrist_mid_mm=stats(np.linalg.norm(p[mask,0]-left_wrist[mask],axis=1)*1000),
             wrist_pair_width_mm=stats(np.linalg.norm(p[mask,1]-p[mask,2],axis=1)*1000)))
    report['conclusions']=dict(label_swap_confirmed=False,physical_marker_vs_model_origin_distinct=True,
        corrected_defect='Prime joint_video elbow diagnostic selected right arm; now explicit left/right side required.',
        speed_error_reduction_verified=False,
        caveat='RFIN hand-surface marker and wrist midpoint are not measured ball center/contact point. FILT_FREQ30 is a stored setting, not proof of causal delay or of filtered versus unfiltered export.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['angle_check','fixed_alignment_side_check','marker_based_trials']},indent=2))
    return report


if __name__=='__main__':run()
