"""Two-way central compliant hand/ball coupling; not anatomical contact."""
import hashlib,itertools,json
from pathlib import Path
import mujoco
import numpy as np
from coupled_payload import model_for


def coupling(anchor, ball, anchor_v, ball_v, stiffness, damping):
    delta=np.asarray(ball)-anchor;length=float(np.linalg.norm(delta))
    radial=float((np.asarray(ball_v)-anchor_v)@delta/length) if length>1e-12 else 0.
    force=-stiffness*delta
    if length>1e-12:force-=damping*radial*delta/length
    return force,.5*stiffness*length**2,damping*radial**2


def taper(t,duration,release_duration=.002):
    if t<duration-release_duration:return 1.,0.
    if t>=duration:return 0.,0.
    return (duration-t)/release_duration,-1/release_duration


def rollout(source,shifts,substeps,stiffness=10000.,release_duration=.002):
    dt=float(source['dt']);n=int(round(.030/dt));start=len(source['qpos'])-1-n;duration=n*dt
    m=model_for(source['lengths'],dt/substeps,free_ball=True);m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    d=mujoco.MjData(m);d.qpos[:13]=source['qpos'][start];d.qvel[:11]=source['qvel'][start]
    d.qpos[16]=1.;mujoco.mj_forward(m,d)
    site=m.site('ball_center').id;body=m.body('free_ball').id
    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,site)
    d.qpos[13:16]=d.site_xpos[site];d.qvel[11:14]=jp@d.qvel;d.qvel[14:17]=jr@d.qvel
    initial_q=d.qpos.copy();initial_v=d.qvel.copy();loads=source['controls'];grid=np.arange(len(loads))*dt
    offsets=np.zeros(11)
    for sl,shift in zip([slice(3,6),slice(6,7),slice(8,11)],shifts):offsets[sl]=shift*.001
    def apply(model,state):
        query=start*dt+state.time-offsets
        # RK4 accumulation can overshoot an endpoint by roundoff only.
        query=np.where(np.abs(query-grid[-1])<1e-12,grid[-1],query)
        query=np.where(np.abs(query)<1e-12,0.,query)
        control=np.array([np.interp(query[j],grid,loads[:,j],left=0.,right=0.) for j in range(11)])
        state.qfrc_applied[:]=0.;state.qfrc_applied[:11]=control
        mujoco.mj_jacSite(model,state,jp,jr,site)
        a=state.site_xpos[site];b=state.xpos[body];av=jp@state.qvel;bv=state.qvel[11:14]
        scale,rate=taper(state.time,duration,release_duration)
        f,elastic,loss=coupling(a,b,av,bv,stiffness*scale,2*scale)
        state.qfrc_applied[:]-=jp.T@f;state.qfrc_applied[11:14]+=f
        release_loss=-.5*stiffness*rate*float(np.sum((b-a)**2))
        return control,np.linalg.norm(f),elastic,loss+release_loss,np.linalg.norm(b-a)
    old=mujoco.get_mjcb_control()
    if old is not None:raise RuntimeError('Existing callback')
    works=np.zeros((2,11));peak=np.zeros(11);group_peak=np.zeros(4);group_work=np.zeros(2);loss_j=0.;max_force=0.;max_gap=0.
    positions=[];velocities=[];balls=[]
    try:
        mujoco.set_mjcb_control(apply);mujoco.mj_forward(m,d)
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);initial_energy=float(d.energy.sum())
        prev=apply(m,d);power=prev[0]*d.qvel[:11];translation=float(power[:3].sum())
        for i in range(n*substeps):
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            now=apply(m,d);p=now[0]*d.qvel[:11];tr=float(p[:3].sum());h=m.opt.timestep
            works[0]+=.5*h*(np.maximum(power,0)+np.maximum(p,0));works[1]+=.5*h*(np.maximum(-power,0)+np.maximum(-p,0))
            group_work+=.5*h*(np.array([max(translation,0),max(-translation,0)])+np.array([max(tr,0),max(-tr,0)]))
            loss_j+=.5*h*(prev[3]+now[3]);peak=np.maximum(peak,np.maximum(abs(prev[0]),abs(now[0])))
            for j,sl in enumerate([slice(0,3),slice(3,6),slice(6,7),slice(8,11)]):group_peak[j]=max(group_peak[j],np.linalg.norm(prev[0][sl]),np.linalg.norm(now[0][sl]))
            max_force=max(max_force,prev[1],now[1]);max_gap=max(max_gap,prev[4],now[4])
            if d.warning.number.any() or not np.isfinite(d.qvel).all():raise RuntimeError('Unstable integration')
            if (i+1)%substeps==0:positions.append(d.qpos[:13].copy());velocities.append(d.qvel[:11].copy());balls.append(d.xpos[body].copy())
            prev,power,translation=now,p,tr
    finally:mujoco.set_mjcb_control(old)
    mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);final_energy=float(d.energy.sum())+prev[2]
    result=dict(shifts_ms=list(shifts),substeps=substeps,speed_kmh=float(np.linalg.norm(d.qvel[11:14])*3.6),work_positive_j=works[0].tolist(),work_negative_j=works[1].tolist(),peak_per_dof=peak.tolist(),group_peaks=group_peak.tolist(),translation_work_pm_j=group_work.tolist(),coupling_peak_force_n=float(max_force),coupling_loss_j=float(loss_j),max_grip_extension_mm=float(max_gap*1000),terminal_spring_energy_j=float(prev[2]),duration_ms=duration*1000,
        initial_energy_j=initial_energy,final_energy_j=final_energy,energy_balance_residual_j=float(final_energy-initial_energy-(works[0].sum()-works[1].sum())+loss_j),stiffness_n_m=stiffness,release_duration_ms=release_duration*1000)
    return result,np.array(positions),np.array(velocities),np.array(balls),initial_q,initial_v


def budget(base,case):
    violations={}
    for key in ['work_positive_j','work_negative_j','peak_per_dof','group_peaks','translation_work_pm_j','coupling_peak_force_n','coupling_loss_j']:
        excess=float(np.max(np.asarray(case[key])-base[key]))
        if excess>1e-8:violations[key]=excess
    return violations


def run():
    root=Path(__file__).resolve().parent;path=root.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz';source=np.load(path)
    out=root/'artifacts/compliant-timing';out.mkdir(parents=True,exist_ok=True)
    grid=[(0,0,0)]+[x for x in itertools.product([-.5,0,.5],repeat=3) if x!=(0,0,0)];rows=[];bases={}
    arm=model_for(source['lengths'],float(source['dt']),ball_mass=0.);dq=np.zeros(11)
    for sub in (1,2):
        for shifts in grid:
            r,q,v,b,iq,iv=rollout(source,shifts,sub)
            if shifts==(0,0,0):bases[sub]=(r,q,b)
            baseline,bq,bb=bases[sub];rot=[];wrist=[];d=mujoco.MjData(arm);e=mujoco.MjData(arm)
            for qa,qb in zip(q,bq):
                mujoco.mj_differentiatePos(arm,dq,1.,qb,qa)
                rot.append(max(np.linalg.norm(dq[3:6]),abs(dq[6]),abs(dq[7]),np.linalg.norm(dq[8:11])))
                d.qpos[:]=qa;e.qpos[:]=qb;mujoco.mj_forward(arm,d);mujoco.mj_forward(arm,e);wrist.append(np.linalg.norm(d.site_xpos[0]-e.site_xpos[0]))
            r.update(gain_kmh=r['speed_kmh']-baseline['speed_kmh'],violations=budget(baseline,r),max_rotation_deg=float(np.rad2deg(max(rot))),max_wrist_delta_mm=float(max(wrist)*1000))
            r['small_motion']=r['max_rotation_deg']<=3 and r['max_wrist_delta_mm']<=10
            rows.append(r)
            if sub==2:np.savez_compressed(out/('state_'+str(shifts)+'.npz'),qpos=q,qvel=v,ball=b,initial_q=iq,initial_v=iv,dt=source['dt'],lengths=source['lengths'])
        print('finished resolution',sub,flush=True)
    accepted=[]
    for a,b in zip(rows[:27],rows[27:]):
        bound=abs(a['speed_kmh']-b['speed_kmh'])+abs(bases[1][0]['speed_kmh']-bases[2][0]['speed_kmh'])
        b['gain_resolution_bound_kmh']=bound
        if b['shifts_ms']!=[0,0,0] and not a['violations'] and not b['violations'] and a['small_motion'] and b['small_motion'] and b['gain_kmh']>bound:accepted.append(b)
    result=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((root/'COMPLIANT_PROTOCOL.md').read_bytes()).hexdigest(),pitch='3034_4',recorded_speed_kmh=128.4256512,accepted_count=len(accepted),rows=rows,scope='Engineering compliant grip with prescribed release, not finger contact or validated athlete gain')
    (root/'compliant_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(accepted=len(accepted),baseline=bases[2][0],fastest=max(rows[28:],key=lambda r:r['gain_kmh'])),indent=2))


if __name__=='__main__':run()
