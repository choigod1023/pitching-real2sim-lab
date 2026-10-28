"""Unilateral finger-pad surrogate. Assumed geometry, not measured fingers."""
import json
from pathlib import Path
import numpy as np
import mujoco
from coupled_payload import model_for
from compliant_timing import taper

RADIUS=.0364
PAD_RADIUS=.008
DIRECTIONS=np.array([[.35,.45,.82],[.35,-.45,.82],[-.3,0,-.954]])
DIRECTIONS/=np.linalg.norm(DIRECTIONS,axis=1)[:,None]


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


def rollout(source,stiffness=10000.,release_ms=2.,mu=.6,substeps=4,no_contact=False):
    dt=float(source['dt']);n=int(round(.030/dt));start=len(source['qpos'])-1-n;duration=n*dt
    m=model_for(source['lengths'],dt/substeps,free_ball=True);m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    d=mujoco.MjData(m);d.qpos[:13]=source['qpos'][start];d.qvel[:11]=source['qvel'][start];d.qpos[16]=1.
    mujoco.mj_forward(m,d);site=m.site('ball_center').id;body=m.body('free_ball').id;hand=m.body('hand').id
    jp=np.zeros((3,m.nv));jr=jp.copy();jhb=jp.copy();jhr=jp.copy();jbb=jp.copy();jbr=jp.copy()
    mujoco.mj_jacSite(m,d,jp,jr,site);d.qpos[13:16]=d.site_xpos[site];d.qvel[11:14]=jp@d.qvel;d.qvel[14:17]=jr@d.qvel
    initial_q=d.qpos.copy();initial_v=d.qvel.copy();loads=np.asarray(source['controls']);grid=np.arange(len(loads))*dt
    local_pads=np.array([.15,0,0])+(RADIUS+PAD_RADIUS)*DIRECTIONS
    def apply(model,state):
        query=start*dt+state.time
        if abs(query-grid[-1])<1e-12:query=grid[-1]
        control=np.array([np.interp(query,grid,loads[:,j],left=0.,right=0.) for j in range(11)])
        state.qfrc_applied[:]=0.;state.qfrc_applied[:11]=control
        scale,rate=taper(state.time,duration,release_ms*.001)
        k=stiffness*scale/3; c=2*scale/3
        center=state.xpos[body];pads=state.xpos[hand]+local_pads@state.xmat[hand].reshape(3,3).T
        net=np.zeros(3);torque=np.zeros(3);elastic=loss=peak=depth=0.;active=0
        for pad in pads:
            delta=center-pad;distance=float(np.linalg.norm(delta));penetration=RADIUS+PAD_RADIUS-distance
            if penetration<=0 or no_contact:continue
            if distance<1e-10:raise RuntimeError('Ball and pad centers coincide')
            normal=delta/distance;point=center-RADIUS*normal
            mujoco.mj_jac(model,state,jhb,jhr,point,hand);mujoco.mj_jac(model,state,jbb,jbr,point,body)
            f,e,l=contact_force(normal,penetration,(jbb-jhb)@state.qvel,k,c,mu)
            state.qfrc_applied[:]+=(jbb-jhb).T@f
            net+=f;torque+=np.cross(point-center,f);elastic+=e
            loss+=l-.5*(stiffness/3)*rate*penetration**2
            peak=max(peak,float(np.linalg.norm(f)));depth=max(depth,penetration)
            active+=int(np.linalg.norm(f)>1e-7)
        return control,net,torque,elastic,loss,peak,depth,active
    if mujoco.get_mjcb_control() is not None:raise RuntimeError('Existing callback')
    works=np.zeros((2,11));loss_j=0.;impulse=np.zeros(3);angular_impulse=np.zeros(3)
    max_force=max_pad=max_depth=0.;contact_samples=0;last_contact=None;positions=[];balls=[];velocities=[];h=m.opt.timestep
    try:
        mujoco.set_mjcb_control(apply);mujoco.mj_forward(m,d)
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);initial_energy=float(d.energy.sum())
        prev=apply(m,d);power=prev[0]*d.qvel[:11]
        initial_spin=d.qvel[14:17].copy()
        for i in range(n*substeps):
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d);now=apply(m,d);p=now[0]*d.qvel[:11]
            works[0]+=.5*h*(np.maximum(power,0)+np.maximum(p,0));works[1]+=.5*h*(np.maximum(-power,0)+np.maximum(-p,0))
            impulse+=.5*h*(prev[1]+now[1]);angular_impulse+=.5*h*(prev[2]+now[2]);loss_j+=.5*h*(prev[4]+now[4])
            max_force=max(max_force,np.linalg.norm(prev[1]),np.linalg.norm(now[1]));max_pad=max(max_pad,prev[5],now[5]);max_depth=max(max_depth,prev[6],now[6])
            if now[7]:contact_samples+=1;last_contact=d.time
            if d.warning.number.any() or not np.isfinite(d.qvel).all():raise RuntimeError('Unstable integration')
            if (i+1)%substeps==0:positions.append(d.qpos[:13].copy());balls.append(d.xpos[body].copy());velocities.append(d.qvel[:11].copy())
            prev,power=now,p
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);final_energy=float(d.energy.sum())+prev[3]
        mujoco.mj_jac(m,d,jbb,jbr,d.xpos[body],body);spin=jbr@d.qvel
    finally:mujoco.set_mjcb_control(None)
    result=dict(stiffness=stiffness,release_ms=release_ms,mu=mu,substeps=substeps,no_contact=no_contact,
        speed_kmh=float(np.linalg.norm(d.qvel[11:14])*3.6),spin_rpm=float(np.linalg.norm(spin)*60/(2*np.pi)),
        ball_velocity_m_s=d.qvel[11:14].tolist(),initial_ball_velocity_m_s=initial_v[11:14].tolist(),
        linear_impulse_residual_ns=(m.body_mass[body]*(d.qvel[11:14]-initial_v[11:14]-m.opt.gravity*duration)-impulse).tolist(),
        angular_impulse_residual_nms=(m.body_inertia[body,0]*(spin-initial_spin)-angular_impulse).tolist(),
        initial_spin_rpm=float(np.linalg.norm(initial_spin)*60/(2*np.pi)),spin_vector_rad_s=spin.tolist(),
        contact_fraction=contact_samples/(n*substeps),last_contact_ms=None if last_contact is None else last_contact*1000,
        max_penetration_mm=float(max_depth*1000),peak_net_force_n=float(max_force),peak_pad_force_n=float(max_pad),
        impulse_ns=impulse.tolist(),angular_impulse_nms=angular_impulse.tolist(),loss_j=float(loss_j),
        work_positive_j=works[0].tolist(),work_negative_j=works[1].tolist(),initial_energy_j=initial_energy,final_energy_j=final_energy,
        energy_balance_residual_j=float(final_energy-initial_energy-(works[0].sum()-works[1].sum())+loss_j))
    return result,dict(qpos=np.array(positions),qvel=np.array(velocities),ball=np.array(balls),dt=dt,lengths=source['lengths'],initial_q=initial_q,initial_v=initial_v)


def run_job(args):
    k,t,mu=args;root=Path(__file__).resolve().parent
    source=np.load(root.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz')
    return rollout(source,k,t,mu)[0]


def run():
    import itertools,hashlib
    from concurrent.futures import ProcessPoolExecutor
    root=Path(__file__).resolve().parent;out=root/'artifacts/friction-contact';out.mkdir(parents=True,exist_ok=True)
    source=np.load(root.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz')
    jobs=list(itertools.product([5000.,10000.,20000.],[1.,2.,4.],[.2,.6,1.]))
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[]
        for i,row in enumerate(pool.map(run_job,jobs)):
            rows.append(row)
            print('completed',i+1,'of',len(jobs),flush=True)
    nominal=next(r for r in rows if (r['stiffness'],r['release_ms'],r['mu'])==(10000.,2.,.6))
    checks=[]
    for r in [nominal,min(rows,key=lambda r:r['speed_kmh']),max(rows,key=lambda r:r['speed_kmh'])]:
        fine,state=rollout(source,r['stiffness'],r['release_ms'],r['mu'],8);fine['coarse_speed_kmh']=r['speed_kmh'];checks.append(fine)
        if r is nominal:np.savez_compressed(out/'nominal.npz',**state)
    zero,_=rollout(source,no_contact=True,substeps=8)
    result=dict(rows=rows,checks=checks,no_contact=zero,protocol_sha256=hashlib.sha256((root/'FRICTION_CONTACT_PROTOCOL.md').read_bytes()).hexdigest())
    (root/'friction_contact_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(nominal=nominal,speed_range=[min(r['speed_kmh'] for r in rows),max(r['speed_kmh'] for r in rows)],checks=checks,no_contact=zero),indent=2))


if __name__=='__main__':run()
