"""Identity-safe speed ladder and outcome-free shoulder-attachment calibration."""
import json
from pathlib import Path
import numpy as np
import mujoco
from coupled_body_search import build, inverse, positions
from baseline_calibration import fit_attachment, endpoint
from trace_error_audit import ObservedPitch, integrate_inputs
from torque_arm import derivatives
from coupled_payload import model_for
from speed_path import rotation_velocity, retarget_velocity

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/same-pitch'
PITCHES = ('3034_1','3034_2','3034_3','3034_4','3034_5')


def require_same_evaluation(before, after):
    """Reject meaningless before/after arithmetic across datasets and metrics."""
    fields=('dataset','pitch','athlete','session','target','event_or_window','metric','units')
    for key in fields:
        if key not in before or key not in after or before[key] != after[key]:
            raise ValueError(f'Incomparable evaluation: {key}')
    if before.get('value') is None or after.get('value') is None:
        raise ValueError('An unobserved result is not a zero-error result')
    if not np.isfinite([before['value'],after['value']]).all():
        raise ValueError('Non-finite evaluation')
    return float(after['value']-before['value'])


def speed(v): return float(np.linalg.norm(v)*3.6)


def identity(pitch):
    return {'dataset':'OBP/baseball_pitching','pitch':pitch,'athlete':2,'session':3034}


def gap_record(pitch, value):
    return {**identity(pitch),'target':'recorded_radar_scalar','event_or_window':'provided_BR_vs_recorded_unspecified_measurement',
            'metric':'absolute_record_gap','units':'km/h','value':value,
            'validated_release_error':False}


def ladder(observation):
    t=observation.b; p=observation.points(t); v=observation.points(t,1)
    rigid_wrist=retarget_velocity(p,v,observation.lengths)
    rotation=rotation_velocity(observation.hand,t,np.array([.15,0,0]))
    vectors={'processed_hand_landmark':observation.hand_proxy(t,1),'RFIN_marker':observation.fin(t,1),
             'processed_wrist':v[2],'fixed_length_wrist':rigid_wrist,
             'hand_rotation_component':rotation,'continuous_assumed_ball':rigid_wrist+rotation}
    return {'direct_ball_velocity_mps':None,'direct_ball_speed_kmh':None,
            'direct_ball_missing_reason':'No identified direct ball trajectory for this same pitch; hand markers are not ball observations.',
            'vectors_mps':{k:x.tolist() for k,x in vectors.items()},'speeds_kmh':{k:speed(x) for k,x in vectors.items()},
            'step_speed_differences_kmh':{'fixed_length_wrist_minus_processed_wrist':speed(rigid_wrist)-speed(v[2]),
                                        'assumed_ball_minus_fixed_wrist':speed(rigid_wrist+rotation)-speed(rigid_wrist)},
            'warning':'Marker/landmark speeds are proxies; vector norms do not add.'}


def run(output=OUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    observations={p:ObservedPitch(p) for p in PITCHES}
    training=observations['3034_1'];pad=50
    tm,tq,tdt,tt,tp=build(1/5760,'3034_1',training,pad,output/'training')
    offsets=fit_attachment(tm,tq[pad:-pad],{n:v[pad:-pad] for n,v in tp.items()})
    for body,site in (('upper','shoulder_jc'),('glove_upper','glove_shoulder_jc')):
        tm.body_pos[tm.body(body).id]=offsets[site]
    mujoco.mj_setConst(tm,mujoco.MjData(tm))
    mujoco.mj_saveLastXML(str(output/'training/frozen_skeleton.xml'),tm)
    records=[]; ledgers=[]
    for pitch,obs in observations.items():
        directory=output/pitch
        m,qp,dt,t,observed=build(1/5760,pitch,obs,pad,directory/'nominal')
        q=qp[pad:-pad]; time=t[pad:-pad];n=len(q)-1
        if n % 2:raise ValueError('Fixed two-resolution grid requires even number of intervals')
        arm=model_for(obs.lengths,dt);aq=obs.sample(t);av,_=derivatives(arm,aq,dt)
        free_arm_v=endpoint(arm,aq[-pad-1],av[-pad-1])
        paths=ladder(obs)
        paths['speeds_kmh']['free_shoulder_arm_reference']=speed(free_arm_v)
        paths['vectors_mps']['free_shoulder_arm_reference']=free_arm_v.tolist()
        entry={**identity(pitch),'recorded_speed_kmh':obs.ledger['recorded_speed_kmh'],
               'split':'geometry_calibration' if pitch=='3034_1' else 'reused_development_check',
               'ladder':paths,'models':{}}
        for mode in ('nominal','frozen_attachment','frozen_skeleton'):
            model=mujoco.MjModel.from_xml_path(str(output/'training/frozen_skeleton.xml' if mode=='frozen_skeleton' else directory/'nominal/model.xml'))
            if model.nq != m.nq or model.nv != m.nv:raise ValueError('Frozen skeleton has incompatible joints')
            if mode=='frozen_attachment':
                for body,site in (('upper','shoulder_jc'),('glove_upper','glove_shoulder_jc')):
                    model.body_pos[model.body(body).id]=offsets[site]
                mujoco.mj_setConst(model,mujoco.MjData(model))
            loads,vp=inverse(model,qp,dt);f=loads[pad:-pad];vr=vp[pad:-pad]
            rv=endpoint(model,q[-1],vr[-1]);names,rpos=positions(model,q)
            mujoco.mj_saveLastXML(str(directory/(mode+'.xml')),model)
            result={'reference_speed_kmh':speed(rv),'reference_velocity_mps':rv.tolist(),'resolutions':{}}
            for stride in (2,1):
                step=dt*stride
                simulated,vel,info=integrate_inputs(model,q[0],vr[0],f[::stride],step,policy='foh')
                _,sp=positions(model,simulated);sv=endpoint(model,simulated[-1],vel[-1])
                ap=obs.points(time[::stride])
                joint_rmse={name:float(np.sqrt(np.mean(np.sum((sp[:,names.index(name)]-ap[:,j])**2,axis=1)))*1000)
                            for j,name in enumerate(('shoulder_jc','elbow_jc','wrist'))}
                ref_rmse=float(np.sqrt(np.mean(np.sum((sp-rpos[::stride])**2,axis=2)))*1000)
                result['resolutions'][str(stride)]={'dt_s':step,'speed_kmh':speed(sv),'velocity_mps':sv.tolist(),
                      'record_gap':gap_record(pitch,abs(speed(sv)-entry['recorded_speed_kmh'])),
                      'integration_speed_difference_kmh':speed(sv)-speed(rv),'reference_site_rmse_mm':ref_rmse,
                      'observed_joint_rmse_mm':joint_rmse,'state_resets':info['state_resets']}
                if stride==1:
                    arm_points=obs.points(time)
                    measured={k:v[pad:-pad] for k,v in observed.items()}
                    measured.update(shoulder_jc=arm_points[:,0],elbow_jc=arm_points[:,1],wrist=arm_points[:,2])
                    np.savez_compressed(directory/(mode+'.npz'),qpos=simulated,qvel=vel,time_s=time,
                                        observed_positions=np.stack([measured.get(name,np.full((len(time),3),np.nan)) for name in names],axis=1))
            result['coarse_fine_speed_difference_kmh']=abs(result['resolutions']['1']['speed_kmh']-result['resolutions']['2']['speed_kmh'])
            entry['models'][mode]=result
        before=entry['models']['nominal']['resolutions']['1'];after=entry['models']['frozen_skeleton']['resolutions']['1']
        entry['geometry_calibration_record_gap_change_kmh']=require_same_evaluation(before['record_gap'],after['record_gap'])
        entry['attribution_kmh']={'continuous_assumption_minus_record':paths['speeds_kmh']['continuous_assumed_ball']-entry['recorded_speed_kmh'],
            'free_arm_minus_continuous_assumption':speed(free_arm_v)-paths['speeds_kmh']['continuous_assumed_ball'],
            'body_reference_minus_free_arm':entry['models']['nominal']['reference_speed_kmh']-speed(free_arm_v),
            'forward_minus_body_reference':before['integration_speed_difference_kmh']}
        closure=sum(entry['attribution_kmh'].values())-(before['speed_kmh']-entry['recorded_speed_kmh'])
        if abs(closure)>1e-9:raise RuntimeError('Speed ladder failed telescoping closure')
        entry['attribution_closure_kmh']=float(closure)
        records.append(entry);ledgers.append(obs.ledger)
        print(json.dumps({'pitch':pitch,'before_kmh':before['speed_kmh'],'after_kmh':after['speed_kmh'],
                          'gap_change_kmh':entry['geometry_calibration_record_gap_change_kmh'],
                          'position_before':before['observed_joint_rmse_mm'],'position_after':after['observed_joint_rmse_mm']}),flush=True)
    aggregate={}
    for mode in ('nominal','frozen_attachment','frozen_skeleton'):
        predicted=np.array([r['models'][mode]['resolutions']['1']['speed_kmh'] for r in records]);truth=np.array([r['recorded_speed_kmh'] for r in records])
        changes=(predicted[1:]-predicted[0])-(truth[1:]-truth[0])
        aggregate[mode]={'reused_four_pitch_record_gap_mae_kmh':float(np.abs(predicted[1:]-truth[1:]).mean()),
                         'reused_four_pitch_change_mae_kmh':float(np.abs(changes).mean()),
                         'fast_3034_2_minus_slow_3034_3_predicted_kmh':float(predicted[1]-predicted[2]),
                         'fast_3034_2_minus_slow_3034_3_recorded_kmh':float(truth[1]-truth[2])}
    truth=np.array([r['recorded_speed_kmh'] for r in records]);aggregate['no_change_mae_kmh']=float(np.abs(truth[1:]-truth[0]).mean())
    nominal=aggregate['nominal'];corrected=aggregate['frozen_skeleton']
    checks={'absolute_gap_improved':bool(corrected['reused_four_pitch_record_gap_mae_kmh']<nominal['reused_four_pitch_record_gap_mae_kmh']),
            'change_beats_nominal_and_no_change':bool(corrected['reused_four_pitch_change_mae_kmh']<min(nominal['reused_four_pitch_change_mae_kmh'],aggregate['no_change_mae_kmh'])),
            'all_observed_joint_errors_nonincreasing':all(r['models']['frozen_skeleton']['resolutions']['1']['observed_joint_rmse_mm'][name] <=
                r['models']['nominal']['resolutions']['1']['observed_joint_rmse_mm'][name]+1e-6
                for r in records[1:] for name in ('shoulder_jc','elbow_jc','wrist')),
            'actual_ball_release_validated':False,'effort_capacity_validated':False}
    adopt=all(checks.values())
    result={'protocol':'SAME_PITCH_PROTOCOL.md','source_ledgers':ledgers,'records':records,'aggregate':aggregate,
            'calibration':{'pitch':'3034_1','features_used':'source landmark geometry plus shoulder/glove shoulder positions; no recorded speed',
                           'recorded_speed_used':False,'mass_or_grip_changed':False,
                           'frozen_skeleton_inertias':'training geometry inertias, not rescaled to improve speed',
                           'frozen_offsets_m':offsets},
            'adopt_as_performance_model':adopt,'adoption_checks':checks,'actual_athlete_improvement_demonstrated':False,
            'same_force_work_intervention_test':False,'direct_ball_prediction_error':None,
            'limitation':'Geometry calibration recomputes reference demand per model; radar gap is not verified same-instant release error. All five pitches are reused development data.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(aggregate,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':run()
