"""Measured whole-body wrench balance, not foot contact or speed correction."""
import json
from pathlib import Path
import numpy as np,pandas as pd,ezc3d,mujoco
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt
import coupled_body_search as body
from trace_error_audit import ObservedPitch
from torque_arm import derivatives
from full_body_comparison import DATA
OUT=Path(__file__).resolve().parent/'artifacts/baseline-ground-balance'

def wrench_at_root(m,d,forces,moments,centers):
    q=np.zeros(m.nv)
    for force,moment,point in zip(forces,moments,centers):
        mujoco.mj_applyFT(m,d,force,moment,point,m.body('pelvis').id,q)
    return q[:6]

def run():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for policy,folder in [('segment_axes','baseline-calibration'),('segmented_trunk','baseline-segmented-trunk')]:
        for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
            o=ObservedPitch(pitch);_,qp,dt,t,_=body.build(1/2880,pitch,o,2,OUT/'scratch',frame_policy=policy)
            m=mujoco.MjModel.from_xml_path(str(OUT.parent/folder/'calibrated.xml'));vp,ap=derivatives(m,qp,dt);q=qp[2:-2];v=vp[2:-2];a=ap[2:-2];t=t[2:-2]
            c=ezc3d.c3d(o.ledger['source_paths']['c3d'],extract_forceplat_data=True);rate=float(c['parameters']['ANALOG']['RATE']['value'][0]);platforms=c['data']['platform'];sos=butter(4,40,fs=rate,output='sos');forces=[];moments=[];centers=[];closure=[]
            for p in platforms:
                if (p['unit_force'],p['unit_moment'],p['unit_position'])!=('N','Nm','m'):raise ValueError('Unexpected units')
                tr=np.arange(p['force'].shape[1])/rate;center=p['corners'].mean(1);force=p['force'].T;moment=p['moment'].T
                active=np.linalg.norm(force,axis=1)>20
                equivalent=np.cross(p['center_of_pressure'].T-center,force)+p['Tz'].T
                closure.append(float(np.max(np.linalg.norm((moment-equivalent)[active],axis=1))))
                forces.append(CubicSpline(tr,sosfiltfilt(sos,force,axis=0))(t));moments.append(CubicSpline(tr,sosfiltfilt(sos,moment,axis=0))(t));centers.append(center)
            forces=np.array(forces);moments=np.array(moments);required=[];measured=[];d=mujoco.MjData(m)
            for i,(qq,vv,aa) in enumerate(zip(q,v,a)):
                d.qpos[:]=qq;d.qvel[:]=vv;d.qacc[:]=aa;mujoco.mj_inverse(m,d);required.append(d.qfrc_inverse[:6].copy());measured.append(wrench_at_root(m,d,forces[:,i],moments[:,i],centers))
            required=np.array(required);measured=np.array(measured);residual=required-measured
            norm=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
            row=dict(pitch=pitch,policy=policy,required_force_rms_n=norm(required[:,:3]),residual_force_rms_n=norm(residual[:,:3]),required_torque_rms_nm=norm(required[:,3:]),residual_torque_rms_nm=norm(residual[:,3:]),wrench_equivalence_max_nm=max(closure),body_mass_kg=float(m.body_mass.sum()-.145),ground_contact_model_validated=False,ball_speed_improvement_claim=False)
            rows.append(row);print(row)
    report=dict(rows=rows,scope='Compare whole-body inverse root demand with measured external resultant wrench; no foot assignment/contact constraint or new forward prediction. Moments applied at plate center, equivalent to COP force plus free moment.',filter_hz=40,source='https://raw.githubusercontent.com/pyomeca/ezc3d/dev/src/modules/ForcePlatforms.cpp')
    (OUT/'report.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':run()
