"""Same-pitch, same-event speed accounting; never fit grip/timing to radar."""
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import mujoco
import ezc3d
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, RotationSpline
from coupled_payload import model_for, inverse_and_rollout
from torque_arm import derivatives
from speed_path import retarget_velocity, rotation_velocity
from ball_release import hand_frame
from pilot import regular_time, vector


def speed(v):
    return float(np.linalg.norm(v)*3.6)


def state_components(model, q, v):
    d=mujoco.MjData(model); d.qpos[:]=q; d.qvel[:]=v
    mujoco.mj_forward(model,d)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,d,jp,jr,model.site('ball_center').id)
    ball=jp@v
    mujoco.mj_jacSite(model,d,jp,jr,model.site('wrist').id)
    wrist=jp@v
    r=d.site_xpos[model.site('ball_center').id]-d.site_xpos[model.site('wrist').id]
    rotation=np.cross(jr@v,r)
    np.testing.assert_allclose(ball,wrist+rotation,atol=1e-10)
    return {'ball_vector_mps':ball.tolist(),'wrist_vector_mps':wrist.tolist(),
            'rotation_vector_mps':rotation.tolist(),'ball_speed_kmh':speed(ball),
            'wrist_speed_kmh':speed(wrist),'rotation_component_norm_kmh':speed(rotation),
            'rotation_speed_increment_kmh':speed(ball)-speed(wrist),
            'vector_identity_residual_mps':float(np.linalg.norm(ball-wrist-rotation))}


def run(root=None):
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    lab=root.parent; out=root/'artifacts/gap-numeric'; out.mkdir(parents=True,exist_ok=True)
    reference=lab/'grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz'
    trace=lab/'speed-journey/artifacts/speed-video/rollout.npz'
    reference_config=reference.with_name('report.json')
    config=json.loads(reference_config.read_text(encoding='utf-8'))
    assert config['pitch']=='3034_1' and config['cutoff_hz']==20.
    with np.load(reference) as f:
        q=f['qpos'];dt=float(f['dt']);lengths=f['lengths']
    with np.load(trace) as f:
        actualq=f['qpos'];actualv=f['qvel'];times=f['times'];stored=f['speed_kmh']
        np.testing.assert_allclose(f['reference_qpos'],q,atol=0,rtol=0)
        np.testing.assert_allclose(f['dt'],dt,atol=0,rtol=0)
    m=model_for(lengths,dt)
    vel,_=derivatives(m,q,dt)
    ref=state_components(m,q[-1],vel[-1]);actual=state_components(m,actualq[-1],actualv[-1])
    np.testing.assert_allclose(actual['ball_speed_kmh'],stored[-1],atol=1e-10)
    data=lab/'openbiomechanics/baseball_pitching/data'
    meta=pd.read_csv(data/'metadata.csv');row=meta[meta.session_pitch.eq('3034_1')].iloc[0]
    radar=float(row.pitch_speed_mph)*1.609344
    g=pd.read_csv(data/'full_sig/landmarks.csv');g=g[g.session_pitch.eq('3034_1')].sort_values('time')
    t=regular_time(g);release=float(g.BR_time.iloc[0]);np.testing.assert_allclose(times[-1],release,atol=1e-9)
    p=CubicSpline(t,np.stack([vector(g,n+'_jc') for n in ['shoulder','elbow','wrist']],axis=1))
    wrist=retarget_velocity(p(release),p(release,1),lengths)
    paths=list((data/'c3d').rglob(row.filename_new));assert len(paths)==1
    c=ezc3d.c3d(str(paths[0]));labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
    rate=float(c['parameters']['POINT']['RATE']['value'][0]);scale={'m':1.,'cm':.01,'mm':.001}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
    raw=[c['data']['points'][:,labels.index(n),:].T for n in ['RFIN','RWRA','RWRB']]
    assert all(np.isfinite(a).all() and np.all(a[:,3]>=0) for a in raw)
    np.testing.assert_allclose(t,np.arange(len(t))/rate,atol=.000051,rtol=0)
    raw=[a[:,:3]*scale for a in raw]
    marks=[sosfiltfilt(butter(4,20,fs=rate,output='sos'),a,axis=0) for a in raw]
    _,frames=hand_frame(*marks);rot=RotationSpline(t,Rotation.from_matrix(frames))
    rotational=rotation_velocity(rot,release,np.array([.15,0.,0.]))
    analytic=wrist+rotational
    stages={'processed_joint_wrist_kmh':speed(p(release,1)[2]),
            'retargeted_wrist_kmh':speed(wrist),
            'analytic_assumed_ball_kmh':speed(analytic),
            'reference_endpoint_ball_kmh':ref['ball_speed_kmh'],
            'integrated_ball_kmh':actual['ball_speed_kmh']}
    increments={
        'fixed_length_retargeting_kmh':stages['retargeted_wrist_kmh']-stages['processed_joint_wrist_kmh'],
        'assumed_rotating_offset_kmh':stages['analytic_assumed_ball_kmh']-stages['retargeted_wrist_kmh'],
        'endpoint_derivative_and_pose_discretization_kmh':stages['reference_endpoint_ball_kmh']-stages['analytic_assumed_ball_kmh'],
        'forward_rollout_vs_reference_kmh':stages['integrated_ball_kmh']-stages['reference_endpoint_ball_kmh']}
    np.testing.assert_allclose(sum(increments.values()),stages['integrated_ball_kmh']-stages['processed_joint_wrist_kmh'],atol=1e-10)
    proxies={}
    for name,markers in [('raw',raw),('20hz',marks)]:
        proxies[name]={'rfin_marker_kmh':speed(CubicSpline(t,markers[0])(release,1)),
                       'wrist_marker_midpoint_kmh':speed(CubicSpline(t,(markers[1]+markers[2])/2)(release,1))}
    gap=radar-stages['integrated_ball_kmh']
    # Recompute inverse demand for the SAME motion, not equal-capacity candidates.
    inertia=[]
    for mass_scale in [.8,1.,1.2]:
        variant=model_for(lengths,dt,hand_mass_scale=mass_scale)
        variant.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
        result,loads=inverse_and_rollout(variant,q,dt)
        inertia.append({'hand_mass_scale':mass_scale,
                        'reference_ball_speed_kmh':state_components(variant,q[-1],vel[-1])['ball_speed_kmh'],
                        'rollout_ball_speed_kmh':result['conditional_release_speed_kmh'],
                        'applied_wrist_peak_torque_norm_nm':float(np.linalg.norm(loads[:-1,8:11],axis=1).max()),
                        'reference_shoulder_net_work_j':result['external_shoulder_net_work_j']})
    report={'pitch':'3034_1','athlete_user_id':int(row.user),'session_id':int(row.session),
            'event_time_s':release,'dt_s':dt,'source_sample_hz':rate,
            'recorded_speed_kmh':radar,'recorded_speed_scope':'Scalar metadata; measurement time/location are unresolved, not established BR truth.',
            'conditional_gap_kmh':gap,'stages':stages,'successive_signed_speed_increments':increments,
            'reference_components':ref,'integrated_components':actual,
            'analytic_vectors_mps':{'wrist':wrist.tolist(),'rotating_offset':rotational.tolist(),'assumed_ball':analytic.tolist()},
            'marker_proxies_not_ball_measurements':proxies,
            'residual_record_minus_analytic_assumed_ball_kmh':radar-speed(analytic),
            'numerical_reference_to_rollout_gap_fraction':abs(increments['forward_rollout_vs_reference_kmh'])/abs(gap),
            'reference_to_rollout_velocity_vector_difference_kmh':speed(np.array(actual['ball_vector_mps'])-np.array(ref['ball_vector_mps'])),
            'same_motion_recomputed_inverse_hand_inertia_stress':inertia,
            'inertia_stress_scope':'Engineering +/-20 percent generic hand mass/inertia; not a biological confidence interval. Each inverse force is recomputed: not fixed force or equal athlete capacity.',
            'sources_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [reference,reference_config,trace,paths[0]]},
            'limitations':['No target-based tuning and no new filter sweep.',
                          'Component vector norms are not additive; only successive scalar differences telescope.',
                          'Integration-reference agreement establishes reconstruction, not unseen-pitch prediction.',
                          'Unexplained record/model gap is not uniquely assignable to fingers, filtering, grip or radar without direct synchronized ball observations.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    r=run();print(json.dumps({k:r[k] for k in ['conditional_gap_kmh','stages','successive_signed_speed_increments','marker_proxies_not_ball_measurements','reference_components','integrated_components']},indent=2))
