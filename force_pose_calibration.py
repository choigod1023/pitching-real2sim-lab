"""Outcome-blind residual reduction with bounded floating-base corrections.

No muscle inference: internal loads are recalculated and every residual root
wrench is reported. Ground-only replay is a separate falsification check.
"""
import json
from pathlib import Path
import numpy as np
import mujoco, ezc3d
from scipy.interpolate import CubicSpline, BSpline
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt
import coupled_body_search as body
from trace_error_audit import ObservedPitch, integrate_inputs
from torque_arm import derivatives
from baseline_calibration import endpoint
from baseline_ground_balance import wrench_at_root

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/force-pose'
MODEL=ROOT.parent/'baseline-calibration/artifacts/baseline-calibration/calibrated.xml'

def rms(x):return float(np.sqrt(np.mean(np.sum(np.asarray(x)**2,axis=-1))))

def corrected(m,q,basis,x):
    # Each component bounded; norm bound is sqrt(3) times component bound.
    delta=basis@x.reshape(-1,6); result=q.copy()
    for row,change in zip(result,delta):
        vel=np.zeros(m.nv);vel[:6]=change
        mujoco.mj_integratePos(m,row,vel,1.)
    return result,delta

def load(pitch):
    obs=ObservedPitch(pitch)
    _,q,dt,t,_=body.build(1/2880,pitch,obs,2,OUT/'scratch')
    m=mujoco.MjModel.from_xml_path(str(MODEL))
    c=ezc3d.c3d(obs.ledger['source_paths']['c3d'],extract_forceplat_data=True)
    rate=float(c['parameters']['ANALOG']['RATE']['value'][0]);sos=butter(4,40,fs=rate,output='sos')
    fs=[];ms=[];centers=[]
    for p in c['data']['platform']:
        if (p['unit_force'],p['unit_moment'],p['unit_position'])!=('N','Nm','m'):raise ValueError('units')
        ts=np.arange(p['force'].shape[1])/rate
        fs.append(CubicSpline(ts,sosfiltfilt(sos,p['force'].T,axis=0)))
        ms.append(CubicSpline(ts,sosfiltfilt(sos,p['moment'].T,axis=0)))
        centers.append(p['corners'].mean(1))
    return obs,m,q,dt,t,fs,ms,np.array(centers)

def demands(m,q,dt,t,fs,ms,centers):
    v,a=derivatives(m,q,dt);loads=[];ground=[];d=mujoco.MjData(m)
    for i in range(2,len(q)-2):
        d.qpos[:]=q[i];d.qvel[:]=v[i];d.qacc[:]=a[i];mujoco.mj_inverse(m,d)
        loads.append(d.qfrc_inverse.copy())
        ground.append(wrench_at_root(m,d,[f(t[i]) for f in fs],[f(t[i]) for f in ms],centers))
    return v,a,np.array(loads),np.array(ground)

def ground_replay(m,q,v,loads,dt,t,fs,ms,centers,sub=2):
    m.opt.timestep=dt/sub;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    d=mujoco.MjData(m);d.qpos[:]=q[0];d.qvel[:]=v[0];grid=np.arange(len(q))*dt
    spline=CubicSpline(grid,loads)
    def cb(model,state):
        tm=float(np.clip(state.time,0,grid[-1]));state.qfrc_applied[:]=spline(tm)
        tt=t[0]+tm
        state.qfrc_applied[:6]=wrench_at_root(model,state,[f(tt) for f in fs],[f(tt) for f in ms],centers)
    if mujoco.get_mjcb_control() is not None:raise RuntimeError('callback occupied')
    states=[q[0].copy()];vel=[v[0].copy()]
    try:
        mujoco.set_mjcb_control(cb);mujoco.mj_forward(m,d)
        for _ in range(len(q)-1):
            for _ in range(sub):mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all() or d.warning.number.any():raise RuntimeError('unstable')
            states.append(d.qpos.copy());vel.append(d.qvel.copy())
    finally:mujoco.set_mjcb_control(None)
    return np.array(states),np.array(vel)

def run(pitch='3034_4'):
    OUT.mkdir(parents=True,exist_ok=True)
    obs,m,q,dt,t,fs,ms,centers=load(pitch)
    # Nonnegative cubic B-spline bases do not overshoot control bounds. Eight control points,
    # fixed a priori; speed is absent from the objective and selection.
    knots=np.r_[np.repeat(t[0],4),np.linspace(t[0],t[-1],6)[1:-1],np.repeat(t[-1],4)]
    basis=BSpline.design_matrix(t,knots,3).toarray()
    # Fit on full grid, padded derivatives, all force and moment samples.
    scale=np.tile([.005]*3+[np.deg2rad(.5)]*3,8)
    def objective(z):
        qq,delta=corrected(m,q,basis,z*scale)
        _,_,forces,ground=demands(m,qq,dt,t,fs,ms,centers)
        residual=(forces[:,:6]-ground)/np.array([1000]*3+[400]*3)
        # Pose preservation and smooth control corrections; engineering weights,
        # not measurement confidence intervals or physiological limits.
        pose=delta/np.array([.005]*3+[np.deg2rad(.5)]*3)
        return np.r_[residual.ravel(),pose.ravel()*.5]
    fit=least_squares(objective,np.zeros(48),bounds=(-1,1),max_nfev=12,diff_step=1e-4,
                      ftol=1e-5,xtol=1e-5,gtol=1e-5)
    qc,delta=corrected(m,q,basis,fit.x*scale)
    rows=[];saved={};names,original=body.positions(m,q[2:-2])
    for name,qp in [('before',q),('after',qc)]:
        v,a,loads,ground=demands(m,qp,dt,t,fs,ms,centers);vv=v[2:-2];qq=qp[2:-2]
        residual=loads[:,:6]-ground
        # Inverse-force replay includes explicitly quantified residual assistance.
        pos,vel,_=integrate_inputs(m,qq[0],vv[0],loads,dt,2,'foh')
        _,sites=body.positions(m,pos)
        observed=obs.points(t[2:-2])
        landmarks=sites[:,[names.index(n) for n in ['shoulder_jc','elbow_jc','wrist']]]
        row=dict(mode=name,speed_kmh=float(np.linalg.norm(endpoint(m,pos[-1],vel[-1]))*3.6),
            measured_speed_kmh=obs.ledger['recorded_speed_kmh'],
            pose_change_rmse_mm=rms((sites-original).reshape(-1,3))*1000,
            observed_arm_rmse_mm=rms((landmarks-observed).reshape(-1,3))*1000,
            residual_force_rms_n=rms(residual[:,:3]),residual_torque_rms_nm=rms(residual[:,3:]),
            internal_positive_work_j=float(np.sum(np.maximum(loads[:,6:]*vv[:,6:],0))*dt),
            residual_positive_work_j=float(np.sum(np.maximum(np.sum(residual*vv[:,:6],axis=1),0))*dt))
        row['speed_error_kmh']=row['speed_kmh']-row['measured_speed_kmh']
        gp,gv=ground_replay(m,qq,vv,loads,dt,t[2:-2],fs,ms,centers)
        _,gs=body.positions(m,gp)
        row.update(ground_only_speed_kmh=float(np.linalg.norm(endpoint(m,gp[-1],gv[-1]))*3.6),
                   ground_only_original_pose_rmse_mm=rms((gs-original).reshape(-1,3))*1000)
        rows.append(row);saved[name]=sites;saved[name+'_ground_only']=gs
        print(pitch,row,flush=True)
    result=dict(pitch=pitch,rows=rows,optimizer=dict(nfev=fit.nfev,status=int(fit.status),cost=fit.cost),
        correction_translation_max_mm=float(np.max(np.linalg.norm(delta[:,:3],axis=1))*1000),
        correction_rotation_max_deg=float(np.rad2deg(np.max(np.linalg.norm(delta[:,3:],axis=1)))),
        protocol='8 cubic B-spline control points, each translation component <=5mm and rotation component <=0.5deg; ground residual/pose weighted fit, no target speed. Root correction only, internal relative joint angles unchanged. Residual assisted and ground-only rollouts separated. Reused development data, not same-ability optimization.')
    np.savez_compressed(OUT/f'{pitch}_overlay.npz',names=names,time=t[2:-2],observed_arm=obs.points(t[2:-2]),delta=delta[2:-2],**saved)
    np.savez_compressed(OUT/f'{pitch}_fit.npz',q_before=q,q_after=qc,time=t,dt=dt,coefficients=fit.x*scale)
    (OUT/f'{pitch}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else '3034_4')
