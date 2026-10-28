"""Bench-gated radial gripper on existing arm with explicit opening work."""
import json
from pathlib import Path
import numpy as np
import mujoco
from coupled_payload import model_for
from grip_bench import DIR,DEPTH,SMOOTH,opening
from curl_geometry import load_frame,path

RADIUS=.0364
PAD_RADIUS=.008
DIRECTIONS=DIR


def contact_force(normal, penetration, relative_velocity, stiffness, damping, mu, smoothing=.1):
    """Force on ball; energy and dissipated power for clipped normal damping."""
    if penetration<=0 or stiffness<=0:
        return np.zeros(3),0.,0.
    n=np.asarray(normal);v=np.asarray(relative_velocity)
    vn=float(n@v);vt=v-vn*n;speed=float(np.linalg.norm(vt))
    fn=max(0.,stiffness*penetration-damping*vn)
    ft=-mu*fn*vt/np.sqrt(speed*speed+smoothing*smoothing)
    loss=(stiffness*penetration-fn)*vn-float(ft@vt)
    return fn*n+ft,.5*stiffness*penetration**2,loss


def rollout(source,stiffness=10000.,release_ms=10.,mu=.6,substeps=4,no_contact=False,aligned=False,curl_deg=0.):
    if release_ms != 10.:
        raise ValueError('Protocol fixes geometric opening duration to 10ms')
    dt=float(source['dt']);n=int(round(.030/dt));start=len(source['qpos'])-1-n;duration=n*dt
    m=model_for(source['lengths'],dt/substeps,free_ball=True);m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    d=mujoco.MjData(m);d.qpos[:13]=source['qpos'][start];d.qvel[:11]=source['qvel'][start];d.qpos[16]=1.
    mujoco.mj_forward(m,d);site=m.site('ball_center').id;body=m.body('free_ball').id;hand=m.body('hand').id
    jp=np.zeros((3,m.nv));jr=jp.copy();jhb=jp.copy();jhr=jp.copy();jbb=jp.copy();jbr=jp.copy()
    mujoco.mj_jacSite(m,d,jp,jr,site);d.qpos[13:16]=d.site_xpos[site];d.qvel[11:14]=jp@d.qvel;d.qvel[14:17]=jr@d.qvel
    initial_q=d.qpos.copy();initial_v=d.qvel.copy();loads=np.asarray(source['controls']);grid=np.arange(len(loads))*dt
    load_dirs,axis,load_unit,load_world=load_frame(source)
    initial_dirs=load_dirs if aligned else DIRECTIONS
    def apply(model,state):
        query=start*dt+state.time
        if abs(query-grid[-1])<1e-12:query=grid[-1]
        control=np.array([np.interp(query,grid,loads[:,j],left=0.,right=0.) for j in range(11)])
        state.qfrc_applied[:]=0.;state.qfrc_applied[:11]=control
        amount,opening_rate=opening(state.time,duration-.010,duration)
        k=stiffness/3; c=2/3
        center=state.xpos[body];rotation=state.xmat[hand].reshape(3,3)
        local_dirs,dir_rate=path(state.time,initial_dirs,axis,curl_deg)
        directions=local_dirs@rotation.T
        radius=RADIUS+PAD_RADIUS-DEPTH+amount
        pads=state.site_xpos[site]+radius*directions
        actuator_vel=(opening_rate*local_dirs+radius*dir_rate)@rotation.T
        normal_load=friction_load=0.
        finger_power=np.zeros(3)
        net=np.zeros(3);torque=np.zeros(3);elastic=loss=peak=depth=0.;active=0
        for pad_index,(pad,direction) in enumerate(zip(pads,directions)):
            delta=center-pad;distance=float(np.linalg.norm(delta));penetration=RADIUS+PAD_RADIUS-distance
            if penetration<=0 or no_contact:continue
            if distance<1e-10:raise RuntimeError('Ball and pad centers coincide')
            normal=delta/distance;point=center-RADIUS*normal
            mujoco.mj_jac(model,state,jhb,jhr,point,hand);mujoco.mj_jac(model,state,jbb,jbr,point,body)
            f,e,l=contact_force(normal,penetration,(jbb-jhb)@state.qvel-actuator_vel[pad_index],k,c,mu,SMOOTH)
            state.qfrc_applied[:]+=(jbb-jhb).T@f
            net+=f;torque+=np.cross(point-center,f);elastic+=e
            loss+=l
            finger_power[pad_index]=f@actuator_vel[pad_index]
            fn=(normal@f)*normal
            normal_load+=fn@load_unit;friction_load+=(f-fn)@load_unit
            peak=max(peak,float(np.linalg.norm(f)));depth=max(depth,penetration)
            active+=int(np.linalg.norm(f)>1e-7)
        return control,net,torque,elastic,loss,peak,depth,active,finger_power,normal_load,friction_load
    if mujoco.get_mjcb_control() is not None:raise RuntimeError('Existing callback')
    normal_impulse=friction_impulse=0.;first_loss=None;finger_works=np.zeros((2,3));works=np.zeros((2,11));loss_j=0.;impulse=np.zeros(3);angular_impulse=np.zeros(3)
    max_force=max_pad=max_depth=0.;contact_samples=0;last_contact=None;positions=[];balls=[];velocities=[];h=m.opt.timestep
    try:
        mujoco.set_mjcb_control(apply);mujoco.mj_forward(m,d)
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);initial_energy=float(d.energy.sum())
        prev=apply(m,d);initial_energy+=prev[3];initial_elastic=prev[3];power=prev[0]*d.qvel[:11]
        initial_spin=d.qvel[14:17].copy()
        for i in range(n*substeps):
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d);now=apply(m,d);p=now[0]*d.qvel[:11]
            works[0]+=.5*h*(np.maximum(power,0)+np.maximum(p,0));works[1]+=.5*h*(np.maximum(-power,0)+np.maximum(-p,0))
            finger_works[0]+=.5*h*(np.maximum(prev[8],0)+np.maximum(now[8],0));finger_works[1]+=.5*h*(np.maximum(-prev[8],0)+np.maximum(-now[8],0))
            normal_impulse+=.5*h*(prev[9]+now[9]);friction_impulse+=.5*h*(prev[10]+now[10])
            if not now[7] and first_loss is None:first_loss=d.time
            impulse+=.5*h*(prev[1]+now[1]);angular_impulse+=.5*h*(prev[2]+now[2]);loss_j+=.5*h*(prev[4]+now[4])
            max_force=max(max_force,np.linalg.norm(prev[1]),np.linalg.norm(now[1]));max_pad=max(max_pad,prev[5],now[5]);max_depth=max(max_depth,prev[6],now[6])
            if now[7]:contact_samples+=1;last_contact=d.time
            if d.warning.number.any() or not np.isfinite(d.qvel).all():raise RuntimeError('Unstable integration')
            if (i+1)%substeps==0:positions.append(d.qpos[:13].copy());balls.append(d.xpos[body].copy());velocities.append(d.qvel[:11].copy())
            prev,power=now,p
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);final_energy=float(d.energy.sum())+prev[3]
        mujoco.mj_jac(m,d,jbb,jbr,d.xpos[body],body);spin=jbr@d.qvel
    finally:mujoco.set_mjcb_control(None)
    result=dict(aligned=aligned,curl_deg=curl_deg,first_loss_ms=None if first_loss is None else first_loss*1000,normal_load_impulse_ns=normal_impulse,friction_load_impulse_ns=friction_impulse,load_direction_world=load_unit.tolist(),load_world_m_s2=load_world.tolist(),initial_directions=initial_dirs.tolist(),curl_axis=axis.tolist(),stiffness=stiffness,release_ms=10.,mu=mu,substeps=substeps,no_contact=no_contact,initial_elastic_j=initial_elastic,
        finger_work_positive_j=finger_works[0].tolist(),finger_work_negative_j=finger_works[1].tolist(),
        speed_kmh=float(np.linalg.norm(d.qvel[11:14])*3.6),spin_rpm=float(np.linalg.norm(spin)*60/(2*np.pi)),
        ball_velocity_m_s=d.qvel[11:14].tolist(),initial_ball_velocity_m_s=initial_v[11:14].tolist(),
        linear_impulse_residual_ns=(m.body_mass[body]*(d.qvel[11:14]-initial_v[11:14]-m.opt.gravity*duration)-impulse).tolist(),
        angular_impulse_residual_nms=(m.body_inertia[body,0]*(spin-initial_spin)-angular_impulse).tolist(),
        initial_spin_rpm=float(np.linalg.norm(initial_spin)*60/(2*np.pi)),spin_vector_rad_s=spin.tolist(),
        contact_fraction=contact_samples/(n*substeps),last_contact_ms=None if last_contact is None else last_contact*1000,
        max_penetration_mm=float(max_depth*1000),peak_net_force_n=float(max_force),peak_pad_force_n=float(max_pad),
        impulse_ns=impulse.tolist(),angular_impulse_nms=angular_impulse.tolist(),loss_j=float(loss_j),
        work_positive_j=works[0].tolist(),work_negative_j=works[1].tolist(),initial_energy_j=initial_energy,final_energy_j=final_energy,
        energy_balance_residual_j=float(final_energy-initial_energy-(works[0].sum()-works[1].sum())-finger_works[0].sum()+finger_works[1].sum()+loss_j))
    return result,dict(qpos=np.array(positions),qvel=np.array(velocities),ball=np.array(balls),dt=dt,lengths=source['lengths'],initial_q=initial_q,initial_v=initial_v)




def job(args):
    from pathlib import Path
    root=Path(__file__).resolve().parent;source=np.load(root.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz')
    aligned,angle,sub=args;r,state=rollout(source,substeps=sub,aligned=aligned,curl_deg=angle)
    out=root/'artifacts/contact-curl';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'state-{int(aligned)}-{angle}-{sub}.npz',**state)
    return r


def violations(base,case):
    result={}
    for key in ['work_positive_j','work_negative_j','finger_work_positive_j','finger_work_negative_j','peak_net_force_n','peak_pad_force_n']:
        extra=float(np.max(np.asarray(case[key])-base[key]))
        if extra>1e-6:result[key]=extra
    if case['max_penetration_mm']>2:result['penetration_over_2mm']=case['max_penetration_mm']-2
    return result


def run():
    from concurrent.futures import ProcessPoolExecutor
    root=Path(__file__).resolve().parent;conditions=[(False,0),(True,0),(True,-10),(True,-5),(True,5),(True,10)]
    jobs=[(a,c,s) for s in [8,16] for a,c in conditions]
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(job,jobs))
    fine=rows[6:];base=fine[0];accepted=[]
    for coarse,case in zip(rows[:6],fine):
        case['violations']=violations(base,case);case['gain_kmh']=case['speed_kmh']-base['speed_kmh']
        case['resolution_bound_kmh']=abs(coarse['speed_kmh']-case['speed_kmh'])+abs(rows[0]['speed_kmh']-base['speed_kmh'])
        case['accepted']=not case['violations'] and not violations(rows[0],coarse) and case['gain_kmh']>case['resolution_bound_kmh'] and case['last_contact_ms']>base['last_contact_ms']
        if case['accepted']:accepted.append(case)
    selected=max(accepted,key=lambda r:r['gain_kmh']) if accepted else min(fine[1:],key=lambda r:r['max_penetration_mm'])
    result=dict(rows=rows,accepted_count=len(accepted),selected=selected)
    (root/'curl_pitch_results.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':run()
