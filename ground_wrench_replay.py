"""Replace floating-root inverse demand with measured wrench; diagnostic only."""
import json
from pathlib import Path
import numpy as np
import mujoco
import ezc3d
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt
import coupled_body_search as body
from trace_error_audit import ObservedPitch
from torque_arm import derivatives
from baseline_ground_balance import wrench_at_root
from baseline_calibration import endpoint

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/whole-chain'


def run():
    obs=ObservedPitch('3034_4');_,qp,dt,t,_=body.build(1/2880,'3034_4',obs,2,OUT/'ground-scratch')
    model_path=ROOT.parent/'baseline-calibration/artifacts/baseline-calibration/calibrated.xml'
    m=mujoco.MjModel.from_xml_path(str(model_path));vp,ap=derivatives(m,qp,dt)
    q,v,a,t=qp[2:-2],vp[2:-2],ap[2:-2],t[2:-2];grid=np.arange(len(q))*dt
    d=mujoco.MjData(m);loads=[]
    for qq,vv,aa in zip(q,v,a):
        d.qpos[:]=qq;d.qvel[:]=vv;d.qacc[:]=aa;mujoco.mj_inverse(m,d);loads.append(d.qfrc_inverse.copy())
    loads=np.array(loads)
    c=ezc3d.c3d(obs.ledger['source_paths']['c3d'],extract_forceplat_data=True)
    rate=float(c['parameters']['ANALOG']['RATE']['value'][0]);sos=butter(4,40,fs=rate,output='sos')
    fs=[];ms=[];centers=[]
    for p in c['data']['platform']:
        if (p['unit_force'],p['unit_moment'],p['unit_position'])!=('N','Nm','m'):raise ValueError('Force units')
        ts=np.arange(p['force'].shape[1])/rate
        fs.append(CubicSpline(ts,sosfiltfilt(sos,p['force'].T,axis=0)))
        ms.append(CubicSpline(ts,sosfiltfilt(sos,p['moment'].T,axis=0)))
        centers.append(p['corners'].mean(1))
    _,reference=body.positions(m,q);rows=[]
    for mode in ['inverse_root_reference','measured_global_wrench']:
        for sub in [1,2,4]:
            m.opt.timestep=dt/sub;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
            d=mujoco.MjData(m);d.qpos[:]=q[0];d.qvel[:]=v[0]
            def control(model,state):
                tm=float(np.clip(state.time,grid[0],grid[-1]))
                state.qfrc_applied[:]=[np.interp(tm,grid,loads[:,j]) for j in range(model.nv)]
                if mode=='measured_global_wrench':
                    absolute=t[0]+tm
                    force=np.array([f(absolute) for f in fs]);moment=np.array([f(absolute) for f in ms])
                    state.qfrc_applied[:6]=wrench_at_root(model,state,force,moment,centers)
            if mujoco.get_mjcb_control() is not None:raise RuntimeError('Control callback occupied')
            states=[d.qpos.copy()];status='success'
            try:
                mujoco.set_mjcb_control(control);mujoco.mj_forward(m,d)
                for i in range(len(q)-1):
                    for _ in range(sub):mujoco.mj_step(m,d)
                    if not np.isfinite(d.qpos).all() or d.warning.number.any():
                        status='unstable';break
                    states.append(d.qpos.copy())
            finally:mujoco.set_mjcb_control(None)
            row=dict(mode=mode,substeps=sub,status=status,completed_samples=len(states))
            if status=='success':
                _,predicted=body.positions(m,np.array(states))
                row.update(speed_kmh=float(np.linalg.norm(endpoint(m,d.qpos,d.qvel))*3.6),
                           all_sites_rmse_mm=float(np.sqrt(np.mean(np.sum((predicted-reference)**2,axis=2)))*1000),
                           pelvis_position_error_mm=float(np.linalg.norm(d.qpos[:3]-q[-1,:3])*1000))
            rows.append(row);print(row,flush=True)
    result=dict(pitch='3034_4',rows=rows,recorded_speed_kmh=obs.ledger['recorded_speed_kmh'],
        scope='Replace only floating-root inverse wrench by measured global plate wrench evaluated at each RK4 state. Internal inverse-demand torques unchanged. Forces applied to pelvis to test whole-body balance, not a foot-contact model. Negative result diagnoses inconsistency, not ground-force causal contribution or athlete intervention.')
    (OUT/'ground_replay.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':run()
