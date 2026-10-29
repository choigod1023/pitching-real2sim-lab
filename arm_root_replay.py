"""Source-compatible arm replay versus a different fixed-torso assumption."""
import json
from pathlib import Path
import ezc3d
import numpy as np
from joint_origin_audit import relative_origin, validate_transforms, fixed_local_prediction, sha
from hand_speed_derivative import numerical_rotations, metrics_for_state
from physical_release_comparison import launch_states

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent


def replay(transforms, train, fixed_torso=False):
    t={k:validate_transforms(v) for k,v in transforms.items()}
    if len(set(len(v) for v in t.values()))!=1:
        raise ValueError('Frame counts differ')
    train=np.asarray(train)
    if train.ndim!=1 or not len(train) or np.any(train<0) or np.any(train>=len(t['torso'])):
        raise ValueError('Invalid training indices')
    elbow_local=relative_origin(t['l_uarm'],t['l_larm'])[train].mean(axis=0)
    wrist_local=relative_origin(t['l_larm'],t['l_hand'])[train].mean(axis=0)
    shoulder=t['l_uarm'][:,:3,3].copy()
    if fixed_torso:
        local=relative_origin(t['torso'],t['l_uarm'])[train]
        shoulder,_=fixed_local_prediction(t['torso'],local)
    elbow=shoulder+np.einsum('nij,j->ni',t['l_uarm'][:,:3,:3],elbow_local)
    wrist=elbow+np.einsum('nij,j->ni',t['l_larm'][:,:3,:3],wrist_local)
    return np.stack([shoulder,elbow,wrist],axis=1)


def run():
    p=LAB/'hand-ball-link/artifacts/hand-ball/observation.json';o=json.loads(p.read_text(encoding='utf-8'))
    for entry in o['source_sha256'].values():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Source changed')
    c=ezc3d.c3d(o['source_sha256']['body']['path']);a=json.loads(Path(o['source_sha256']['alignment']['path']).read_text(encoding='utf-8'))
    labels=c['parameters']['ROTATION']['LABELS']['value']
    transforms={name:np.moveaxis(c['data']['rotations'][:,:,labels.index(name+'_4X4'),:],-1,0) for name in ['torso','l_uarm','l_larm','l_hand']}
    train=np.array([940,950,960])+1;evaluate=np.array([945,955,965,970,977,979])+1
    q,shift,scale=np.array(a['rotation']),np.array(a['translation']),float(a['scale'])
    truth=np.stack([transforms[name][:,:3,3] for name in ['l_uarm','l_larm','l_hand']],axis=1)
    raw_to_rig=lambda x:scale*(x@q.T)+shift
    ballrow=next(x for x in o['conditions'] if x['threshold']==80 and x['c3d_frame_offset']==1)
    ball=np.array([x['ball_rig_m'] for x in ballrow['positions']]);idx=np.arange(973,977)+1;rate=float(o['rate_hz'])
    rotation,_=numerical_rotations(q@transforms['l_hand'][idx,:3,:3]);gravity=q@np.array([0,0,-9.81]);models={}
    for name,fixed in [('F_source_free_root',False),('T_imposed_fixed_torso',True)]:
        raw=replay(transforms,train,fixed);rig=raw_to_rig(raw)
        errors=np.linalg.norm(raw[evaluate]-truth[evaluate],axis=2)*scale*1000
        launch=launch_states(np.arange(4)/rate,ball[:4],rig[idx,2],rotation)['B']
        metrics=metrics_for_state(*launch,gravity,ball[4:],rate,62.7*1.609344)
        models[name]={'source_estimate_rmse_mm':{joint:float(np.sqrt(np.mean(errors[:,i]**2))) for i,joint in enumerate(['shoulder','elbow','wrist'])},
                      'source_estimate_max_mm':float(errors.max()),'B_ball_evaluation':metrics,
                      'actual_joint_error_mm':None,'required_external_work_j':None,'force_capacity_verified':False}
    result={'reference':'OBP_movement_BaseballThrow_001','models':models,
            'training_video_frames':(train-1).tolist(),'evaluation_video_frames':(evaluate-1).tolist(),
            'baseline_replaced':False,'new_shoulder_observation_correction_applied':False,
            'scientific_stage_passed':False,'actual_performance_gain_kmh':None,
            'interpretation':'Kinematic source reproduction and structural sensitivity, not ground truth anatomical accuracy, muscle validation or intervention.'}
    (ROOT/'arm_root_replay_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'source_rmse_mm':v['source_estimate_rmse_mm'],'ball_speed_mae':v['B_ball_evaluation']['mean_absolute_speed_discrepancy_kmh'],
                         'ball_position_rmse_mm':v['B_ball_evaluation']['position_rmse_mm']} for k,v in models.items()}))
    return result


if __name__=='__main__':run()
