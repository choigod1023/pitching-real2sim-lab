"""Recheck primary local files before advancing the hand/ball research gate."""
import hashlib
import json
from pathlib import Path
import cv2
import ezc3d
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
OUT=ROOT/'artifacts/source-audit'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_reference(record):
    p=Path(record['path'])
    exists=p.is_file();actual=digest(p) if exists else None
    return {'path':str(p),'exists':exists,'expected_sha256':record['sha256'],'actual_sha256':actual,
            'hash_matches':bool(exists and actual==record['sha256'])}


def velocity_displacement_budget(target_kmh,rate_hz):
    """Two-frame displacement sensitivity, not an estimated confidence bound."""
    if not np.isfinite([target_kmh,rate_hz]).all() or target_kmh<=0 or rate_hz<=0:
        raise ValueError('Positive finite target and rate required')
    return target_kmh/3.6/rate_hz*1000


def frame_overlap(calibration_frames,ball_frames):
    return sorted(set(map(int,calibration_frames)) & set(map(int,ball_frames)))


def audit(output=OUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    observation_path=LAB/'hand-ball-link/artifacts/hand-ball/observation.json'
    observation=json.loads(observation_path.read_text(encoding='utf-8'))
    chain={k:check_reference(v) for k,v in observation['source_sha256'].items()}
    body_path=Path(observation['source_sha256']['body']['path'])
    alignment_path=Path(observation['source_sha256']['alignment']['path'])
    alignment=json.loads(alignment_path.read_text(encoding='utf-8'))
    c=ezc3d.c3d(str(body_path));labels=c['parameters']['ROTATION']['LABELS']['value']
    h=labels.index('l_hand_4X4');all_transforms=c['data']['rotations'][:,:,h,:]
    rotations=np.moveaxis(all_transforms[:3,:3,:],-1,0)
    cond=next(x for x in observation['conditions'] if x['threshold']==80 and x['c3d_frame_offset']==1)
    frames=[int(x['video_frame']) for x in cond['positions']]
    indices=[int(x['c3d_index']) for x in cond['positions']]
    used=rotations[indices]
    csv=LAB/'openbiomechanics/computer_vision/4d_reconstruction/release_keypoints.csv'
    points=pd.read_csv(csv);kf=sorted(points.video_frame.unique().tolist())
    videos=[]
    video_manifest_path=LAB/'speed-journey/artifacts/release-observation/manifest.json'
    video_manifest=json.loads(video_manifest_path.read_text(encoding='utf-8'))
    for camera in (20,21):
        path=LAB/f'speed-journey/artifacts/release-observation/camera{camera}.mp4'
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise ValueError('Primary video not readable: '+str(path))
        stored=next(x for x in video_manifest if x['camera']==camera)
        actual_hash=digest(path)
        videos.append({'camera':camera,'path':str(path),'sha256':actual_hash,
                       'stored_manifest_hash_matches':actual_hash==stored['sha256'],
                       'stored_container_pts_count':len(stored['pts']),
                       'stored_container_interval_ticks':stored['interval_ticks'],
                       'physical_exposure_synchronization_validated':False,
                       'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                       'nominal_rate_hz':float(cap.get(cv2.CAP_PROP_FPS)),
                       'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
        cap.release()
    events=c['parameters'].get('EVENT',{})
    result={'trial':observation['trial'],'scope':'Fresh source-integrity and observability audit, not a speed improvement experiment',
            'observation_sha256':digest(observation_path),'source_chain':chain,'primary_videos':videos,
            'original_video_manifest_sha256':digest(video_manifest_path),
            'body':{'path':str(body_path),'rotation_labels':labels,'rotation_frames':int(all_transforms.shape[-1]),
                    'point_labels':c['parameters']['POINT']['LABELS']['value'],
                    'point_units':c['parameters']['POINT']['UNITS']['value'],
                    'point_rate_hz':float(c['parameters']['POINT']['RATE']['value'][0]),
                    'event_parameter_keys':list(events),
                    'used_hand_rotation_orthogonality_max':float(np.max(np.abs(np.swapaxes(used,-1,-2)@used-np.eye(3)))),
                    'used_hand_rotation_det_range':[float(np.linalg.det(used).min()),float(np.linalg.det(used).max())]},
            'alignment':{'sha256':digest(alignment_path),'scale_m_per_mm':alignment['scale'],
                         'nonunit_length_scale_percent':(alignment['scale']/.001-1)*100,
                         'fit_code_available':alignment.get('fit_code_available'),'fit_frame_selection':alignment.get('fit_frame_selection'),
                         'status':alignment.get('status'),'fit_cameras':alignment.get('fit_cameras')},
            'temporal_support':{'keypoint_csv_sha256':digest(csv),'keypoint_frame_range':[min(kf),max(kf)],
                                'keypoint_frame_count':len(kf),'ball_frames':frames,'body_indices':indices,
                                'keypoint_ball_overlap':frame_overlap(kf,frames)},
            'finite_difference_sensitivity':{'target_kmh':1.,'rate_hz':360.,
                  'one_interval_displacement_mm':velocity_displacement_budget(1.,360.),
                  'scope':'Exact two-frame conversion only; not actual tracking error or confidence interval; longer-window fitting changes sensitivity.'},
            'data_gate_passed':False,'hand_ball_stage_allowed':False,
            'remaining':['Exact alignment fit indices/code absent; no independent hand/ball-window alignment bound.',
                         'Same nominal frame rates do not alone establish exposure-time correspondence.',
                         'Hand/ball overlap in images does not identify physical contact termination.',
                         'Radar location/time and true release definitions remain unverified.']}
    (output/'local-source-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'chain_matches':{k:v['hash_matches'] for k,v in chain.items()},'videos':len(videos),
                      'overlap':result['temporal_support']['keypoint_ball_overlap'],
                      'rotation_error':result['body']['used_hand_rotation_orthogonality_max'],'data_gate_passed':False}))
    return result


if __name__=='__main__':audit()
