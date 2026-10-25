"""Conditional same-pitch hand-segment/ball decomposition; no radar fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent

def relative_position(ball, origin, rotation):
    """rotation columns are local axes expressed in the global frame."""
    return np.einsum('nji,nj->ni', rotation, ball-origin)

def decompose(ball, origin, rotation, dt):
    r = relative_position(ball, origin, rotation)
    # Exact finite-difference product identity, not instantaneous angular velocity.
    d_origin = np.diff(origin, axis=0)/dt
    rotation_term = np.einsum('nij,nj->ni', np.diff(rotation,axis=0)/dt, (r[1:]+r[:-1])/2)
    relative_term = np.einsum('nij,nj->ni', (rotation[1:]+rotation[:-1])/2, np.diff(r,axis=0)/dt)
    velocity = np.diff(ball,axis=0)/dt
    return r, velocity, d_origin, rotation_term, relative_term

def window_stats(r):
    mean = r.mean(axis=0)
    return {'fixed_local_point_fit_m':mean.tolist(),
            'fixed_point_position_rmse_mm':float(np.sqrt(np.mean(np.sum((r-mean)**2,axis=1)))*1000),
            'endpoint_relative_displacement_mm':float(np.linalg.norm(r[-1]-r[0])*1000),
            'origin_distance_range_mm':[float(np.linalg.norm(r,axis=1).min()*1000),float(np.linalg.norm(r,axis=1).max()*1000)]}

def run():
    import ezc3d
    paths = {'ball':ROOT.parent/'speed-journey/artifacts/release-observation/candidate_audit.json',
             'alignment':ROOT.parent/'openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json',
             'body':ROOT.parent/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d'}
    candidate=json.loads(paths['ball'].read_text())
    alignment=json.loads(paths['alignment'].read_text())
    c=ezc3d.c3d(str(paths['body']))
    labels=c['parameters']['ROTATION']['LABELS']['value']
    a=c['data']['rotations'][:,:,labels.index('l_hand_4X4'),:]
    rate=float(c['parameters']['ROTATION']['RATE']['value'][0])
    q=np.asarray(alignment['rotation']); translation=np.asarray(alignment['translation']); scale=alignment['scale']
    frames=np.arange(973,980)
    conditions=[]
    for threshold in [60,80,100]:
        byframe={v['frame']:v for v in candidate['conditional_triangulation'] if v['threshold']==threshold}
        ball=np.array([byframe[int(i)]['position_camera19_mm'] for i in frames])/1000
        for offset in [0,1,2]:
            origin=(scale*(q@a[:3,3,frames+offset])).T+translation
            rotations=np.einsum('ij,jkn->nik',q,a[:3,:3,frames+offset])
            ortho=float(np.max(np.abs(np.einsum('nji,njk->nik',rotations,rotations)-np.eye(3))))
            if ortho>1e-5 or not np.isfinite(origin).all():
                raise ValueError('Invalid supplied transforms; do not silently repair')
            r,v,vo,vr,vrel=decompose(ball,origin,rotations,1/rate)
            # Float32 source matrices are nearly, not exactly, orthogonal.
            closure=float(np.max(np.linalg.norm(v-vo-vr-vrel,axis=1)))
            row={'threshold':threshold,'c3d_frame_offset':offset,'default_offset':offset==1,
                 'orthogonality_max_abs':ortho,'finite_difference_closure_max_m_s':closure,
                 'windows':{'all_973_979':window_stats(r),'hand_overlap_973_976':window_stats(r[:4]),'separated_outline_977_979':window_stats(r[4:])},
                 'positions':[{'video_frame':int(f),'c3d_index':int(f+offset),'ball_rig_m':b.tolist(),'hand_origin_rig_m':o.tolist(),'ball_local_m':x.tolist()} for f,b,o,x in zip(frames,ball,origin,r)],
                 'intervals':[{'video_frames':[int(frames[i]),int(frames[i+1])],
                               'ball_speed_kmh':float(np.linalg.norm(v[i])*3.6),
                               'origin_velocity_m_s':vo[i].tolist(),'rotation_term_m_s':vr[i].tolist(),'relative_term_m_s':vrel[i].tolist(),
                               'local_relative_fd_speed_m_s':float(np.linalg.norm((r[i+1]-r[i])*rate)),
                               'ball_velocity_m_s':v[i].tolist()} for i in range(len(v))]}
            conditions.append(row)
    # Compare all predeclared processing settings at each matched video frame.
    stack=np.array([[p['ball_local_m'] for p in c['positions']] for c in conditions])
    spread=[float(max(np.linalg.norm(stack[i,k]-stack[j,k]) for i in range(9) for j in range(9))*1000) for k in range(7)]
    def selected_spread(selected):
        points=np.array([[p['ball_local_m'] for p in c['positions']] for c in selected])
        return float(max(np.linalg.norm(points[i,k]-points[j,k])*1000 for i in range(len(points)) for j in range(len(points)) for k in range(7)))
    report={'trial':'OBP_movement_BaseballThrow_001','segment':'l_hand_4X4',
            'scope':'Conditional decomposition with contributor alignment, filtered Theia hand and bright-core ball candidates. Hand origin is not independently verified anatomical wrist. No contact/release truth, radar target, or independent accuracy validation. NOT 3034_1.',
            'source_sha256':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},
            'rate_hz':rate,'similarity_scale':scale,'rotation_storage':'4x4 local-to-lab homogeneous matrix: upper left rotation, last column origin; lab-to-rig left-multiplied rotation; no SO3 repair',
            'predeclared_thresholds':[60,80,100],'predeclared_offsets':[0,1,2],
            'threshold_only_local_spread_at_default_offset_mm':selected_spread([c for c in conditions if c['c3d_frame_offset']==1]),
            'timing_only_local_spread_at_middle_threshold_mm':selected_spread([c for c in conditions if c['threshold']==80]),
            'local_position_max_processing_spread_mm_by_frame':dict(zip(map(str,frames),spread)),
            'conditions':conditions}
    out=ROOT/'artifacts/hand-ball'; out.mkdir(parents=True,exist_ok=True)
    (out/'observation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in conditions:
        print(row['threshold'],row['c3d_frame_offset'],json.dumps(row['windows']))
    print('max_processing_spread_mm',max(spread),'max_closure_m_s',max(c['finite_difference_closure_max_m_s'] for c in conditions))
    return report

if __name__=='__main__':
    run()
