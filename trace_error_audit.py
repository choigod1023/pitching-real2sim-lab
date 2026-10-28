"""Separate geometry, sampled inverse demand and integration errors; no speed fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import ezc3d
import mujoco
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt
from scipy.spatial.transform import Rotation,RotationSpline
from coupled_payload import fit_pose,model_for
from ball_release import hand_frame
from pilot import regular_time,vector
from torque_arm import derivatives
from joint_intervention import rotation_differences

DATA=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class ObservedPitch:
    def __init__(self,pitch,data=DATA,reconstruct_rounding=False):
        data=Path(data);self.pitch=pitch
        meta=pd.read_csv(data/'metadata.csv');row=meta[meta.session_pitch.eq(pitch)].iloc[0]
        poi=pd.read_csv(data/'poi/poi_metrics.csv');pr=poi[poi.session_pitch.eq(pitch)].iloc[0]
        if pr.p_throws!='R':raise ValueError('This frame model supports right-handed pitches only')
        paths=list((data/'c3d').rglob(row.filename_new))
        if len(paths)!=1:raise ValueError('Nonunique C3D')
        c=ezc3d.c3d(str(paths[0]));labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
        rate=float(c['parameters']['POINT']['RATE']['value'][0]);units=c['parameters']['POINT']['UNITS']['value'][0].strip().lower()
        scale={'m':1.,'cm':.01,'mm':.001}[units];sos=butter(4,20.,fs=rate,output='sos');markers={}
        for name in ['RFIN','RWRA','RWRB','RFRM']:
            raw=c['data']['points'][:,labels.index(name),:].T
            if not np.isfinite(raw).all() or np.any(raw[:,3]<0):raise ValueError('Missing marker, no interpolation of gaps')
            markers[name]=sosfiltfilt(sos,raw[:,:3]*scale,axis=0)
        all_g=pd.read_csv(data/'full_sig/landmarks.csv');g=all_g[all_g.session_pitch.eq(pitch)].sort_values('time')
        rounding_info=None
        if reconstruct_rounding:
            from quantization_reconstruction import reconstruct_landmarks
            g,rounding_info=reconstruct_landmarks(g)
        t=regular_time(g)
        if len(t)!=len(markers['RFIN']) or not np.allclose(t,np.arange(len(t))/rate,atol=.000051,rtol=0):raise ValueError('Time alignment mismatch')
        _,frames=hand_frame(markers['RFIN'],markers['RWRA'],markers['RWRB'])
        self.hand=RotationSpline(t,Rotation.from_matrix(frames))
        self.points=CubicSpline(t,np.stack([vector(g,n+'_jc') for n in ['shoulder','elbow','wrist']],axis=1))
        self.fore=CubicSpline(t,markers['RFRM']);self.fin=CubicSpline(t,markers['RFIN'])
        self.hand_proxy=CubicSpline(t,vector(g,'hand_jc'))
        self.a=float(g.fp_10_time.iloc[0]);self.b=float(g.BR_time.iloc[0]);self.tmin=t[0];self.tmax=t[-1]
        self.lengths=np.median(np.linalg.norm(np.diff(self.points(np.linspace(self.a,self.b,100)),axis=1),axis=2),axis=0)
        self.ledger=dict(rounding_reconstruction=rounding_info,pitch=pitch,user=int(row.user),session=int(row.session),recorded_speed_kmh=float(row.pitch_speed_mph)*1.609344,
            rate_hz=rate,raw_point_units=units,processed_position_units='m',frame_count=len(t),fp10_s=self.a,br_s=self.b,
            point_processing='4th-order zero-phase 20Hz Butterworth on hand/forearm markers; supplied processed joint centers unchanged',
            source_paths={'c3d':str(paths[0]),'landmarks':str(data/'full_sig/landmarks.csv')},
            source_sha256={name:sha(path) for name,path in [('c3d',paths[0]),('metadata',data/'metadata.csv'),('poi',data/'poi/poi_metrics.csv'),('landmarks',data/'full_sig/landmarks.csv'),('joint_angles',data/'full_sig/joint_angles.csv'),('joint_velos',data/'full_sig/joint_velos.csv')]},
            lengths_m=self.lengths.tolist(),arm_masses_kg=[2.14154,1.280182],hand_mass_kg=.482044,ball_mass_kg=.145,
            grip_m=[.15,0,0],coordinates='laboratory coordinates retained; marker-frame rotations not anatomical angles',
            br_detection='unconfirmed',radar_measurement_position_and_time='unconfirmed',capacity='not measured; generic inertias')
    def sample(self,times):
        if times.min()<self.tmin or times.max()>self.tmax:raise ValueError('No extrapolation')
        return fit_pose(self.points(times),self.fore(times),self.hand(times).as_matrix())[0]


def reference_inputs(observed,requested_dt,padded=False):
    n=int(np.ceil((observed.b-observed.a)/requested_dt))+1;t=np.linspace(observed.a,observed.b,n);dt=float(t[1]-t[0])
    m=model_for(observed.lengths,dt)
    if padded:
        tq=t[0]+np.arange(-2,n+2)*dt;qp=observed.sample(tq);vp,ap=derivatives(m,qp,dt);q=qp[2:-2];v=vp[2:-2];a=ap[2:-2]
    else:q=observed.sample(t);v,a=derivatives(m,q,dt)
    d=mujoco.MjData(m);loads=[];residual=[]
    for i,(pos,vel,acc) in enumerate(zip(q,v,a)):
        d.qpos[:]=pos;d.qvel[:]=vel;d.qacc[:]=acc;mujoco.mj_inverse(m,d);f=d.qfrc_inverse.copy();loads.append(f)
        if i in (0,len(q)//2,len(q)-1):
            d.qfrc_applied[:]=f;mujoco.mj_forward(m,d);residual.append(float(np.max(np.abs(d.qacc-acc))))
    return m,t,q,v,a,np.array(loads),max(residual)


def integrate_inputs(model,q0,v0,loads,dt,substeps=1,policy='zoh'):
    """FOH evaluates prescribed force at every RK4 stage through a scoped callback."""
    if policy not in ('zoh','foh') or substeps<1:raise ValueError('Invalid integrator policy')
    old=mujoco.get_mjcb_control()
    if old is not None:raise RuntimeError('Existing callback would affect inverse/forward audit')
    model.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4;model.opt.timestep=dt/substeps
    d=mujoco.MjData(model);d.qpos[:]=q0;d.qvel[:]=v0
    callback_times=[]
    def callback(m,state):
        x=np.clip(state.time/dt,0,len(loads)-1);i=min(int(np.floor(x)),len(loads)-2);u=x-i
        state.qfrc_applied[:]=(1-u)*loads[i]+u*loads[i+1]
        callback_times.append(float(state.time))
    positions=[d.qpos.copy()];velocities=[d.qvel.copy()]
    try:
        if policy=='foh':mujoco.set_mjcb_control(callback)
        for i in range(len(loads)-1):
            if policy=='zoh':d.qfrc_applied[:]=loads[i]
            for _ in range(substeps):
                mujoco.mj_step(model,d)
                if d.warning.number.any() or not np.isfinite(d.qvel).all():raise RuntimeError('Unstable rollout')
            positions.append(d.qpos.copy());velocities.append(d.qvel.copy())
    finally:mujoco.set_mjcb_control(old)
    return np.array(positions),np.array(velocities),dict(state_resets=0,callback_evaluations=len(callback_times),end_time_s=float(d.time),terminal_load_scope='ZOH excludes terminal row; FOH uses endpoint value only by interpolation in final interval')


def kinematics(model,q,v):
    d=mujoco.MjData(model);positions=[];vels=[];rotations=[]
    for pos,vel in zip(q,v):
        d.qpos[:]=pos;d.qvel[:]=vel;mujoco.mj_forward(model,d)
        p=[];vv=[]
        for name in ['upper','forearm','hand']:
            body=model.body(name).id;jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacBody(model,d,jp,jr,body);p.append(d.xpos[body].copy());vv.append(jp@vel)
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);site=model.site('ball_center').id
        mujoco.mj_jacSite(model,d,jp,jr,site);p.append(d.site_xpos[site].copy());vv.append(jp@vel)
        positions.append(p);vels.append(vv);rotations.append(d.xmat[model.body('hand').id].reshape(3,3).copy())
    return np.array(positions),np.array(vels),np.array(rotations)


def point_errors(actual,target):
    diff=np.linalg.norm(actual-target,axis=-1)
    return dict(rmse_mm=(np.sqrt(np.mean(diff**2,axis=0))*1000).tolist(),max_mm=(diff.max(axis=0)*1000).tolist(),final_mm=(diff[-1]*1000).tolist())


def summarize(observed,model,t,q,v,pos,vel):
    rp,rv,rr=kinematics(model,q,v);sp,sv,sr=kinematics(model,pos,vel)
    observed_p=observed.points(t);observed_v=observed.points(t,1)
    fr_error=Rotation.from_matrix(np.einsum('nji,njk->nik',observed.hand(t).as_matrix(),rr)).magnitude()
    speed=lambda x:float(np.linalg.norm(x)*3.6)
    return dict(points_order=['shoulder','elbow','wrist','assumed_ball'],
        observed_to_fk_position=point_errors(rp[:,:3],observed_p),
        observed_to_fk_final_velocity_vector_difference_kmh=(np.linalg.norm(rv[-1,:3]-observed_v[-1],axis=-1)*3.6).tolist(),
        observed_hand_frame_to_fk_max_deg=float(np.degrees(fr_error).max()),
        fk_to_rollout_position=point_errors(sp,rp),fk_to_rollout_joint_rotation_max_deg=rotation_differences(model,q,pos),
        fk_to_rollout_final_velocity_vector_difference_kmh=(np.linalg.norm(sv[-1]-rv[-1],axis=-1)*3.6).tolist(),
        final_reference_speed_kmh=speed(rv[-1,3]),final_rollout_speed_kmh=speed(sv[-1,3]),
        final_reference_wrist_speed_kmh=speed(rv[-1,2]),final_rollout_wrist_speed_kmh=speed(sv[-1,2]),
        observed_processed_wrist_speed_kmh=speed(observed_v[-1,2]),observed_rfin_proxy_speed_kmh=speed(observed.fin(t[-1],1)),
        observed_processed_hand_proxy_speed_kmh=speed(observed.hand_proxy(t[-1],1)),
        assumed_ball_rotation_component_kmh=speed(rv[-1,3]-rv[-1,2]),
        record_minus_rollout_kmh=observed.ledger['recorded_speed_kmh']-speed(sv[-1,3]))


def run(out,pitches,grids):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);results=[];ledgers=[]
    for pitch in pitches:
        obs=ObservedPitch(pitch);ledgers.append(obs.ledger)
        for rate in grids:
            for padded in (False,True):
                m,t,q,v,a,loads,residual=reference_inputs(obs,1/rate,padded);dt=float(t[1]-t[0])
                for policy in ('zoh','foh'):
                    for sub in ((1,2) if rate==grids[0] else (1,)):
                        pos,vel,diag=integrate_inputs(m,q[0],v[0],loads,dt,sub,policy)
                        r=summarize(obs,m,t,q,v,pos,vel);r.update(pitch=pitch,source_rate_requested=rate,source_dt=dt,padded_derivatives=padded,policy=policy,substeps=sub,inverse_forward_acceleration_max_abs=residual,**diag)
                        results.append(r)
                        name=f'{pitch}-{rate}-pad{int(padded)}-{policy}-sub{sub}'
                        np.savez_compressed(out/f'{name}.npz',qreference=q,vreference=v,areference=a,controls=loads,qpos=pos,qvel=vel,dt=dt,lengths=obs.lengths)
                        print(name,round(r['final_rollout_speed_kmh'],6),round(r['fk_to_rollout_joint_rotation_max_deg']['marker_roll'],6),flush=True)
    report=dict(protocol_sha256=sha('TRACE_ERROR_PROTOCOL.md'),code_sha256=sha(__file__),ledger=ledgers,runs=results,scope='Offline numerical trace audit; no actual ball tracking or intervention effect measured')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='artifacts/trace-error');p.add_argument('--pitches',nargs='+',default=['3034_4','3034_2']);p.add_argument('--grids',nargs='+',type=int,default=[11520,23040,46080]);a=p.parse_args();run(a.out,a.pitches,a.grids)
