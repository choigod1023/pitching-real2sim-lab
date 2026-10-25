"""Observed-motion inverse/forward consistency in a simplified two-link arm.

No muscle or ball model. The free shoulder is driven by an explicitly reported
external wrench. This is a numerical dynamics benchmark, not athlete capacity.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from pilot import regular_time, vector


def unit(v):
    lengths=np.linalg.norm(v,axis=-1,keepdims=True)
    if np.any(lengths<1e-8):raise ValueError('Degenerate segment')
    return v/lengths


def arm_poses(shoulder,elbow,wrist):
    """Infer upper-arm twist from elbow plane; do not claim it was measured."""
    u=unit(elbow-shoulder);v=unit(wrist-elbow)
    cos=np.clip(np.sum(u*v,axis=1),-1,1)
    x=-unit(v-cos[:,None]*u);z=-u;y=np.cross(z,x)
    matrices=np.stack([x,y,z],axis=-1)
    xyzw=Rotation.from_matrix(matrices).as_quat()
    quats=xyzw[:,[3,0,1,2]]
    for i in range(1,len(quats)):
        if quats[i-1]@quats[i]<0:quats[i]*=-1
    return np.column_stack([shoulder,quats,np.arccos(cos)])


def make_model(lengths,step,masses=(2.14154,1.280182),radii=(.04,.03)):
    """Generic uniform cylinders, nominal masses; not an OpenSim conversion."""
    if any(x<=0 or not np.isfinite(x) for x in [*lengths,step,*masses,*radii]):
        raise ValueError('Positive finite geometry, mass and time step required')
    def inertia(m,l,r):
        transverse=m*(3*r*r+l*l)/12
        return f'{transverse} {transverse} {m*r*r/2}'
    l1,l2=lengths;m1,m2=masses;r1,r2=radii
    xml=f'''<mujoco model="two-link torque benchmark">
    <compiler angle="radian"/>
    <option timestep="{step}" gravity="0 0 -9.81" integrator="Euler"/>
    <default><geom contype="0" conaffinity="0"/></default>
    <worldbody><body name="upper">
      <freejoint name="shoulder"/>
      <inertial pos="0 0 {-l1/2}" mass="{m1}" diaginertia="{inertia(m1,l1,r1)}"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-l1}" size="{r1}" rgba=".9 .5 .2 1"/>
      <body name="forearm" pos="0 0 {-l1}">
        <joint name="elbow" type="hinge" axis="0 1 0" limited="false"/>
        <inertial pos="0 0 {-l2/2}" mass="{m2}" diaginertia="{inertia(m2,l2,r2)}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-l2}" size="{r2}" rgba=".3 .6 .8 1"/>
        <site name="wrist" pos="0 0 {-l2}" size=".01"/>
      </body></body></worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def derivatives(model,q,dt):
    if len(q)<3:raise ValueError('Three or more frames required')
    velocity=np.zeros((len(q),model.nv))
    for i in range(len(q)):
        a=max(0,i-1);b=min(len(q)-1,i+1)
        mujoco.mj_differentiatePos(model,velocity[i],(b-a)*dt,q[a],q[b])
    return velocity,np.gradient(velocity,dt,axis=0,edge_order=2)


def run_dynamics(model,q,dt,force_scale=1.0):
    """Inverse reference then open-loop forward integration; no state resets."""
    velocity,acceleration=derivatives(model,q,dt)
    inverse=mujoco.MjData(model);forces=[];reference_wrist=[]
    for pos,vel,acc in zip(q,velocity,acceleration):
        inverse.qpos[:]=pos;inverse.qvel[:]=vel;inverse.qacc[:]=acc
        mujoco.mj_inverse(model,inverse)
        forces.append(inverse.qfrc_inverse.copy())
        reference_wrist.append(inverse.site_xpos[model.site('wrist').id].copy())
    forces=np.array(forces)
    simulation=mujoco.MjData(model)
    simulation.qpos[:]=q[0];simulation.qvel[:]=velocity[0]
    mujoco.mj_forward(model,simulation)
    wrist=[simulation.site_xpos[model.site('wrist').id].copy()]
    poses=[simulation.qpos.copy()]
    for i in range(len(q)-1):
        simulation.qfrc_applied[:]=force_scale*forces[i]
        mujoco.mj_step(model,simulation)
        # Refresh derived positions without changing integrated qpos/qvel.
        mujoco.mj_forward(model,simulation)
        if simulation.warning.number.any() or not np.isfinite(simulation.qpos).all():
            raise RuntimeError('Unstable forward trajectory')
        wrist.append(simulation.site_xpos[model.site('wrist').id].copy())
        poses.append(simulation.qpos.copy())
    error=np.linalg.norm(np.array(wrist)-reference_wrist,axis=1)
    report={'step_s':dt,'frames':len(q),'simulated_seconds':float(simulation.time),
            'wrist_rmse_mm':float(np.sqrt(np.mean(error**2))*1000),
            'wrist_max_error_mm':float(error.max()*1000),
            'shoulder_external_force_peak_n':float(np.linalg.norm(forces[:,:3],axis=1).max()),
            'shoulder_external_torque_peak_nm':float(np.linalg.norm(forces[:,3:6],axis=1).max()),
            'elbow_torque_peak_nm':float(np.abs(forces[:,6]).max()),
            'applied_force_scale':force_scale,
            'state_resets_after_initialization':0,'feedback_controller':False,
            'muscle_capacity_validated':False,'ball_speed_prediction':False}
    return report,dict(reference_qpos=q,integrated_qpos=np.array(poses),required_generalized_forces=forces,
                       wrist_error_m=error)


def experiment(data,pitch,out,steps=(1/1440,1/2880,1/5760)):
    meta=pd.read_csv(data/'metadata.csv')
    poi=pd.read_csv(data/'poi/poi_metrics.csv',usecols=['session_pitch','p_throws'])
    meta=meta.merge(poi,on='session_pitch',validate='one_to_one')
    eligible=meta[meta.p_throws=='R'].sort_values(['user','session_pitch'])
    if pitch:eligible=eligible[eligible.session_pitch==pitch]
    if eligible.empty:raise ValueError('No right-handed trial')
    selected=eligible.iloc[0]
    all_data=pd.read_csv(data/'full_sig/landmarks.csv')
    g=all_data[all_data.session_pitch==selected.session_pitch].sort_values('time')
    t=regular_time(g)
    a=float(g.fp_10_time.iloc[0]);b=float(g.BR_time.iloc[0])
    if not t[1]<a<b<t[-2]:raise ValueError('Events outside interpolation support')
    coordinates=np.stack([vector(g,n+'_jc') for n in ['shoulder','elbow','wrist']],axis=1)
    if not np.isfinite(coordinates).all():raise ValueError('Nonfinite landmarks')
    spline=CubicSpline(t,coordinates,axis=0)
    observed=spline(np.linspace(a,b,100))
    lengths=np.median(np.linalg.norm(np.diff(observed,axis=1),axis=2),axis=0)
    out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for index,step in enumerate(steps):
        n=int(np.ceil((b-a)/step))+1;dense=np.linspace(a,b,n);dt=float(dense[1]-dense[0])
        points=spline(dense);q=arm_poses(points[:,0],points[:,1],points[:,2])
        model=make_model(lengths,dt)
        result,arrays=run_dynamics(model,q,dt)
        result['requested_step_s']=step
        reports.append(result)
        np.savez(out/f'rollout-{index}.npz',time=dense,**arrays)
    report={'pitch':str(selected.session_pitch),'segment_lengths_m':lengths.tolist(),
            'nominal_masses_kg':[2.14154,1.280182], 'cylinder_radii_m':[.04,.03],
            'mass_source':'Nominal upper arm and ulna+radius masses from Rajagopal2015 OpenSense; not participant masses.',
            'source':'https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal_OpenSense/Rajagopal2015_opensense.osim',
            'inertia_assumption':'Uniform solid cylinders, not source model inertia tensors or subject calibrated tissue.',
            'pose_assumption':'Upper-arm twist inferred from elbow plane; shoulder translation observed; no independent humeral rotation measurement.',
            'runs':reports,'scope':'Numerical inverse-forward consistency using external shoulder wrench and elbow torque; no muscle, hand, ball, contact or physiological limits.',
            'error_reference':'Forward wrist versus fixed-length retargeted reference, not raw measured wrist.',
            'optimization_ready':False}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--pitch')
    parser.add_argument('--output',type=Path,default=Path('artifacts/torque-arm'))
    args=parser.parse_args()
    print(json.dumps(experiment(args.data,args.pitch,args.output),indent=2))
