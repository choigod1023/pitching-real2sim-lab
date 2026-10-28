"""Whole-chain kinematic discrepancy attribution, not causal power attribution."""
import itertools
import json
import math
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
import ezc3d
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt
import coupled_body_search as body
from trace_error_audit import ObservedPitch
from full_body_comparison import DATA
from pilot import vector,regular_time
from torque_arm import derivatives
from baseline_ground_balance import wrench_at_root

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/whole-chain'
NAMES=['pelvis_translation','pelvis_to_shoulder','shoulder_to_elbow','elbow_to_wrist']


def speed(v):return float(np.linalg.norm(v)*3.6)


def replace_components(base, observed, hand):
    base,observed,hand=np.asarray(base),np.asarray(observed),np.asarray(hand)
    values={}
    for bits in itertools.product([0,1],repeat=4):
        v=np.where(np.array(bits)[:,None],observed,base).sum(axis=0)+hand
        values[bits]=speed(v)
    # Exact all-subsets marginal change of speed; no causal interpretation.
    attribution=[]
    for i in range(4):
        value=0.
        for bits in values:
            if bits[i]:continue
            k=sum(bits);other=list(bits);other[i]=1
            value+=math.factorial(k)*math.factorial(3-k)/math.factorial(4)*(values[tuple(other)]-values[bits])
        attribution.append(value)
    return values,attribution


def site_state(m,d,name):
    j=np.zeros((3,m.nv));jr=j.copy();sid=m.site(name).id
    mujoco.mj_jacSite(m,d,j,jr,sid)
    return d.site_xpos[sid].copy(),j@d.qvel


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    model_path=ROOT.parent/'baseline-calibration/artifacts/baseline-calibration/calibrated.xml'
    allg=pd.read_csv(DATA/'full_sig/landmarks.csv');rows=[]
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        obs=ObservedPitch(pitch)
        _,qp,dt,tp,points=body.build(1/2880,pitch,obs,2,OUT/'scratch')
        m=mujoco.MjModel.from_xml_path(str(model_path));vp,ap=derivatives(m,qp,dt)
        q,v,a,t=qp[2:-2],vp[2:-2],ap[2:-2],tp[2:-2]
        g=allg[allg.session_pitch.eq(pitch)].sort_values('time');native=regular_time(g)
        pelvis=CubicSpline(native,(vector(g,'rear_hip')+vector(g,'lead_hip'))/2)
        observations={'pelvis':pelvis,'shoulder':lambda x,nu=0:obs.points(x,nu)[:,0],
                      'elbow':lambda x,nu=0:obs.points(x,nu)[:,1],'wrist':lambda x,nu=0:obs.points(x,nu)[:,2]}
        d=mujoco.MjData(m);poses={n:[] for n in ['pelvis','shoulder','elbow','wrist']};diagnostics=[]
        capture=ezc3d.c3d(obs.ledger['source_paths']['c3d'],extract_forceplat_data=True)
        rate=float(capture['parameters']['ANALOG']['RATE']['value'][0]);sos=butter(4,40,fs=rate,output='sos')
        forces=[];moments=[];centers=[]
        for plate in capture['data']['platform']:
            if (plate['unit_force'],plate['unit_moment'],plate['unit_position'])!=('N','Nm','m'):raise ValueError('Unexpected units')
            ts=np.arange(plate['force'].shape[1])/rate
            forces.append(CubicSpline(ts,sosfiltfilt(sos,plate['force'].T,axis=0))(t))
            moments.append(CubicSpline(ts,sosfiltfilt(sos,plate['moment'].T,axis=0))(t));centers.append(plate['corners'].mean(1))
        forces,moments=np.array(forces),np.array(moments);residuals=[]
        indices={len(t)-1:0,len(t)-1-round((1/360)/dt):-1}
        for i in range(len(t)):
            d.qpos[:]=q[i];d.qvel[:]=v[i];d.qacc[:]=a[i];mujoco.mj_inverse(m,d)
            residuals.append(d.qfrc_inverse[:6]-wrench_at_root(m,d,forces[:,i],moments[:,i],centers))
            hip1,hv1=site_state(m,d,'rear_hip');hip2,hv2=site_state(m,d,'lead_hip')
            sp,sv=site_state(m,d,'shoulder_jc');ep,ev=site_state(m,d,'elbow_jc');wp,wv=site_state(m,d,'wrist');bp,bv=site_state(m,d,'ball_center')
            pp,pv=(hip1+hip2)/2,(hv1+hv2)/2
            for name,p in zip(poses,[pp,sp,ep,wp]):poses[name].append(p)
            if i not in indices:continue
            ov=obs.points(t[i],1);opv=pelvis(t[i],1)
            base=np.array([pv,sv-pv,ev-sv,wv-ev]);observed=np.array([opv,ov[0]-opv,ov[1]-ov[0],ov[2]-ov[1]])
            hand=bv-wv;values,contributions=replace_components(base,observed,hand)
            diagnostics.append(dict(frame_offset=indices[i],time_s=float(t[i]),
                baseline_proxy_speed_kmh=values[(0,0,0,0)],all_observed_to_wrist_proxy_speed_kmh=values[(1,1,1,1)],
                recorded_speed_kmh=obs.ledger['recorded_speed_kmh'],
                exact_velocity_sum_error=float(np.linalg.norm(base.sum(axis=0)+hand-bv)),
                attribution_speed_change_kmh=dict(zip(NAMES,contributions)),
                component_velocity_discrepancy_kmh={n:speed(observed[j]-base[j]) for j,n in enumerate(NAMES)},
                modeled_hand_offset_velocity_m_s=hand.tolist(),observed_RFIN_speed_kmh=speed(obs.fin(t[i],1)),
                all_subset_replacements=[dict(observed_components=[NAMES[j] for j,b in enumerate(bits) if b],speed_kmh=s) for bits,s in values.items()]))
        residuals=np.array(residuals)
        rows.append(dict(pitch=pitch,calibration_pitch=pitch=='3034_1',
            landmark_rmse_mm={n:float(np.sqrt(np.mean(np.sum((np.array(p)-observations[n](t))**2,axis=1)))*1000) for n,p in poses.items()},
            root_force_residual_rms_n=float(np.sqrt(np.mean(np.sum(residuals[:,:3]**2,axis=1)))),
            root_torque_residual_rms_nm=float(np.sqrt(np.mean(np.sum(residuals[:,3:]**2,axis=1)))),diagnostics=diagnostics))
        print(pitch,rows[-1]['landmark_rmse_mm'],diagnostics[-1]['attribution_speed_change_kmh'],flush=True)
    result=dict(rows=rows,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        scope='Same five observed pitches. Frozen first-pitch geometry/mass; original processed coordinate precision. Reference kinematics plus inverse wrench audit, NOT new forward rollout or causal energy attribution. Scalar radar speed has no observed velocity vector. Hand/ball remains assumed and RFIN is not ball.',
        ground_contact_validated=False,muscles_validated=False,hand_ball_observed=False)
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':run()
