"""Train-only analytic hand/relative launch velocity; fixed evaluation source."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import ezc3d,numpy as np
from scipy.spatial.transform import Rotation
from physical_release_comparison import launch_states,free_flight
from heldout_speed_comparison import evaluate
from relative_ball_state import ball_velocity

ROOT=Path('artifacts/hand-speed-derivative')
SOURCE=Path('../hand-ball-link/artifacts/hand-ball/observation.json')

def numerical_rotations(rotation):
    """Remove float32 storage roundoff, never anatomical/large matrix errors."""
    rotation=np.asarray(rotation,float);error=float(np.max(abs(np.swapaxes(rotation,1,2)@rotation-np.eye(3))))
    if not np.isfinite(rotation).all() or error>16*np.finfo(np.float32).eps or np.any(np.linalg.det(rotation)<=0):raise ValueError('Rotation error exceeds source float32 precision')
    u,_,v=np.linalg.svd(rotation);proper=u@v
    return proper,{'source_orthogonality_max_abs':error,'maximum_matrix_frobenius_correction':float(np.max(np.linalg.norm(proper-rotation,axis=(1,2)))),'criterion':'16*float32 epsilon; numerical storage check, not a scientific accuracy threshold'}

def analytic_state(times,ball,origin,rotation,degree=2,relative_degree=1):
    times,ball,origin,rotation=map(lambda x:np.asarray(x,float),(times,ball,origin,rotation))
    if degree not in [1,2] or relative_degree not in [1,2] or len(times)<max(degree,relative_degree)+2 or ball.shape!=origin.shape or ball.shape!=(len(times),3) or rotation.shape!=(len(times),3,3):raise ValueError('Invalid training shape/degree')
    if not all(np.isfinite(x).all() for x in [times,ball,origin,rotation]) or np.any(np.diff(times)<=0):raise ValueError('Finite increasing training times required')
    if not np.allclose(np.swapaxes(rotation,1,2)@rotation,np.eye(3),atol=1e-8) or not np.allclose(np.linalg.det(rotation),1.,atol=1e-8):raise ValueError('Proper training rotations required')
    tau=times-times[-1];span=times[-1]-times[0];u=tau/span;design=np.column_stack([u**i for i in range(1,degree+1)])
    end=rotation[-1];logs=Rotation.from_matrix(np.einsum('ij,njk->nik',end.T,rotation)).as_rotvec()
    if np.max(np.linalg.norm(logs,axis=1))>=np.pi-.01:raise ValueError('Rotation log branch ambiguous')
    origin_coeff=np.linalg.lstsq(design,origin-origin[-1],rcond=None)[0];log_coeff=np.linalg.lstsq(design,logs,rcond=None)[0]
    local=np.einsum('nji,nj->ni',rotation,ball-origin);relative_coeff=np.linalg.lstsq(np.column_stack([u**i for i in range(relative_degree+1)]),local,rcond=None)[0]
    offset=relative_coeff[0];relative_velocity=relative_coeff[1]/span;origin_velocity=origin_coeff[0]/span;omega=end@(log_coeff[0]/span)
    components=ball_velocity(origin_velocity,end,omega,offset,relative_velocity);position=origin[-1]+end@offset
    detail={'degree':degree,'relative_degree':relative_degree,'training_count':len(times),'origin_velocity_m_s':origin_velocity.tolist(),'omega_world_rad_s':omega.tolist(),'rotation_component_m_s':np.cross(omega,end@offset).tolist(),'relative_component_m_s':(end@relative_velocity).tolist(),'local_offset_m':offset.tolist(),'training_ball_used':True,'future_body_or_ball_used':False,'polynomial_design_condition':float(np.linalg.cond(design))}
    return position,components['ball_velocity_mps'],detail

def metrics_for_state(position,velocity,gravity,truth,rate,label):
    coarse,vc=free_flight(position,velocity,gravity,1/rate,1);fine,vf=free_flight(position,velocity,gravity,1/rate,2);metrics=evaluate(fine,truth,1/rate,label);t=np.arange(1,4)/rate;exact=position+t[:,None]*velocity+.5*t[:,None]**2*gravity
    metrics.update(launch_position_m=np.asarray(position).tolist(),launch_velocity_m_s=np.asarray(velocity).tolist(),launch_speed_kmh=float(np.linalg.norm(velocity)*3.6),coarse_fine_position_max_m=float(np.max(np.linalg.norm(coarse-fine,axis=1))),coarse_fine_velocity_max_m_s=float(np.max(np.linalg.norm(vc-vf,axis=1))),analytic_position_max_m=float(np.max(np.linalg.norm(fine-exact,axis=1))))
    return metrics

def improves(candidate,baseline):
    fields=['mean_absolute_speed_discrepancy_kmh','mean_velocity_vector_discrepancy_kmh','position_rmse_mm']
    return bool(all(candidate[k]<baseline[k] for k in fields))

def run():
    ROOT.mkdir(parents=True,exist_ok=True);observed=json.loads(SOURCE.read_text(encoding='utf-8'));ledger=observed['source_sha256']
    for role,source in ledger.items():
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:raise ValueError(f'Changed source {role}')
    alignment=json.loads(Path(ledger['alignment']['path']).read_text());basis=np.array(alignment['rotation']);c3d=ezc3d.c3d(ledger['body']['path']);labels=c3d['parameters']['ROTATION']['LABELS']['value'];rr=c3d['data']['rotations'][:3,:3,labels.index('l_hand_4X4'),:];frames=np.arange(973,980);rate=float(observed['rate_hz']);time=(frames-frames[0])/rate;gravity=basis@np.array([0.,0.,-9.81]);label=62.7*1.609344;rows=[]
    for condition in observed['conditions']:
        if [p['video_frame'] for p in condition['positions']]!=frames.tolist():raise ValueError('Unexpected frame order')
        ball=np.array([p['ball_rig_m'] for p in condition['positions']]);origin=np.array([p['hand_origin_rig_m'] for p in condition['positions']]);offset=condition['c3d_frame_offset'];raw_rotation=np.einsum('ij,jkn->nik',basis,rr[:,:,frames[:4]+offset]);raw_states=launch_states(time[:4],ball[:4],origin[:4],raw_rotation);rotation,rotation_info=numerical_rotations(raw_rotation);states=launch_states(time[:4],ball[:4],origin[:4],rotation);states['B_raw']=raw_states['B'];detail={'rotation_numeric_correction':rotation_info,'B_raw_to_canonical_launch_velocity_difference_m_s':float(np.linalg.norm(raw_states['B'][1]-states['B'][1]))}
        for degree in [1,2]:
            p,v,info=analytic_state(time[:4],ball[:4],origin[:4],rotation,degree);name=f'D{degree}';states[name]=(p,v);detail[name]=info
            if not np.allclose(p,states['B'][0],rtol=0,atol=1e-10):raise ValueError('Velocity-only experiment changed launch position')
        metrics={name:metrics_for_state(*state,gravity,ball[4:],rate,label) for name,state in states.items()};rows.append({'threshold':condition['threshold'],'offset':offset,'metrics':metrics,'analytic_details':detail,'D2_beats_B_all_metrics':improves(metrics['D2'],metrics['B']),'D2_beats_C_all_metrics':improves(metrics['D2'],metrics['C'])})
    primary=next(r for r in rows if r['threshold']==80 and r['offset']==1)
    result={'trial':observed['trial'],'training_frames':frames[:4].tolist(),'evaluation_frames':frames[4:].tolist(),'primary_condition':{'threshold':80,'offset':1},'primary':primary,'conditions':rows,'source_label_kmh':label,'source_label_instrument_definition_verified':False,'reference':'Provisional triangulated future ball positions; not ground truth','gravity_rig_m_s2':gravity.tolist(),'source_ledger':ledger,'observation_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(Path('HAND_SPEED_DERIVATIVE_PROTOCOL.md').read_bytes()).hexdigest(),'scientific_stage_passed':False,'causal_intervention_validated':False,'body_motion_changed':False,'force_work_changed':False}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'primary':{m:{k:v[k] for k in ['launch_speed_kmh','mean_absolute_speed_discrepancy_kmh','mean_velocity_vector_discrepancy_kmh','position_rmse_mm','mean_absolute_radar_scalar_discrepancy_kmh']} for m,v in primary['metrics'].items()},'D2_beats_B_all_metrics':primary['D2_beats_B_all_metrics'],'D2_beats_C_all_metrics':primary['D2_beats_C_all_metrics']},indent=2));return result
if __name__=='__main__':run()
