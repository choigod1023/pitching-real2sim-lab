"""2x2 controlled reconstruction audit: rounding bins x mobile shoulder origin.

The mobile origin is a measured diagnostic joint, NOT a validated scapular model.
All additional forces/work are accounted for; no radar labels used in fitting.
"""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
import coupled_body_search as body
from trace_error_audit import ObservedPitch,integrate_inputs
from baseline_calibration import endpoint
from force_pose_calibration import load,demands,ground_replay,rms,MODEL

OUT=Path(__file__).resolve().parent/'artifacts/noise-structure'

def joint_width(m,j):
    return {0:7,1:4,2:1,3:1}[int(m.jnt_type[j])]

def mobile_model(m,q,points,path):
    """Add measured local translations; preserve all original joint coordinates."""
    source=path.with_suffix('.source.xml');mujoco.mj_saveLastXML(str(source),m)
    tree=ET.parse(source)
    for bodyname,prefix in [('upper','pitch_origin'),('glove_upper','glove_origin')]:
        el=tree.find(f'.//body[@name="{bodyname}"]')
        for j,axis in enumerate(['1 0 0','0 1 0','0 0 1']):
            el.insert(j,ET.Element('joint',name=f'{prefix}_{j}',type='slide',axis=axis,
                limited='false',damping='0',armature='0',frictionloss='0'))
    tree.write(path);new=mujoco.MjModel.from_xml_path(str(path));qn=np.tile(new.qpos0,(len(q),1))
    for j in range(m.njnt):
        a=m.jnt_qposadr[j];b=new.joint(m.joint(j).name).qposadr[0];w=joint_width(m,j)
        qn[:,b:b+w]=q[:,a:a+w]
    d=mujoco.MjData(new);local=[]
    for i in range(len(q)):
        d.qpos[:]=qn[i];mujoco.mj_forward(new,d);rt=d.xmat[new.body('torso').id].reshape(3,3).T
        changes=[]
        for name,prefix in [('shoulder_jc','pitch_origin'),('glove_shoulder_jc','glove_origin')]:
            change=rt@(points[name][i]-d.site_xpos[new.site(name).id])
            for j in range(3):qn[i,new.joint(f'{prefix}_{j}').qposadr[0]]=change[j]
            changes.append(change)
        local.append(changes)
    return new,qn,np.array(local)

def evaluate(m,qp,dt,t,fs,ms,centers,obs,original):
    v,a,loads,ground=demands(m,qp,dt,t,fs,ms,centers)
    q=qp[2:-2];vv=v[2:-2];res=loads[:,:6]-ground
    pos,vel,info=integrate_inputs(m,q[0],vv[0],loads,dt,2,'foh')
    names,sites=body.positions(m,pos);_,ref=body.positions(m,q)
    ix=[names.index(n) for n in ['shoulder_jc','elbow_jc','wrist']]
    arm=sites[:,ix];truth=obs.points(t[2:-2])
    speed=float(np.linalg.norm(endpoint(m,pos[-1],vel[-1]))*3.6)
    extra=[int(m.joint(f'{prefix}_{j}').dofadr[0]) for prefix in ['pitch_origin','glove_origin'] for j in range(3)] if 'pitch_origin_0' in [m.joint(j).name for j in range(m.njnt)] else []
    row=dict(speed_kmh=speed,measured_speed_kmh=obs.ledger['recorded_speed_kmh'],
        signed_speed_error_kmh=speed-obs.ledger['recorded_speed_kmh'],
        arm_rmse_mm=rms((arm-truth).reshape(-1,3))*1000,
        shoulder_rmse_mm=rms(arm[:,0]-truth[:,0])*1000,
        wrist_rmse_mm=rms(arm[:,2]-truth[:,2])*1000,
        original_model_pose_rmse_mm=rms((sites-original).reshape(-1,3))*1000,
        numerical_replay_rmse_mm=rms((sites-ref).reshape(-1,3))*1000,
        root_force_rms_n=rms(res[:,:3]),root_torque_rms_nm=rms(res[:,3:]),
        internal_positive_work_j=float(np.maximum(loads[:,6:]*vv[:,6:],0).sum()*dt),
        root_residual_positive_work_j=float(np.maximum(np.sum(res*vv[:,:6],axis=1),0).sum()*dt),
        added_origin_positive_work_j=float(np.maximum(loads[:,extra]*vv[:,extra],0).sum()*dt) if extra else 0.,
        added_origin_peak_force_n=float(np.linalg.norm(loads[:,extra].reshape(len(q),2,3),axis=2).max()) if extra else 0.,
        state_resets=info['state_resets'])
    gp,gv=ground_replay(m,q,vv,loads,dt,t[2:-2],fs,ms,centers)
    _,gs=body.positions(m,gp)
    row.update(ground_only_speed_kmh=float(np.linalg.norm(endpoint(m,gp[-1],gv[-1]))*3.6),
        ground_only_observed_arm_rmse_mm=rms((gs[:,ix]-truth).reshape(-1,3))*1000)
    return row,sites

def run(pitches=None):
    OUT.mkdir(parents=True,exist_ok=True);records=[]
    for pitch in pitches or ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        rawobs,m,q,dt,t,fs,ms,centers=load(pitch)
        names,original=body.positions(m,q[2:-2]);states={};rounding=None
        for noise in ['original','rounding_bins']:
            obs=rawobs if noise=='original' else ObservedPitch(pitch,reconstruct_rounding=True)
            _,qp,dt,tp,points=body.build(1/2880,pitch,obs,2,OUT/'scratch',reconstruct_rounding=noise=='rounding_bins')
            rounding=obs.ledger['rounding_reconstruction'] or rounding
            for structure in ['fixed_origin','measured_mobile_origin']:
                mm=mujoco.MjModel.from_xml_path(str(MODEL));qq=qp
                if structure=='measured_mobile_origin':
                    mm,qq,local=mobile_model(mm,qp,points,OUT/f'{pitch}_{noise}.xml')
                row,sites=evaluate(mm,qq,dt,tp,fs,ms,centers,rawobs,original)
                row.update(pitch=pitch,noise=noise,structure=structure,
                    max_origin_motion_mm=float(np.linalg.norm(local,axis=2).max()*1000) if structure=='measured_mobile_origin' else 0.)
                records.append(row);states[noise+'__'+structure]=sites
                print(pitch,noise,structure,'speed',round(row['speed_kmh'],3),'arm',round(row['arm_rmse_mm'],3),'force',round(row['root_force_rms_n'],1),flush=True)
                if pitch=='3034_4':
                    np.savez_compressed(OUT/f'example_{noise}_{structure}.npz',q=qq,t=tp,dt=dt)
                    mujoco.mj_saveLastXML(str(OUT/f'example_{noise}_{structure}.xml'),mm)
        np.savez_compressed(OUT/f'{pitch}_overlay.npz',names=names,time=t[2:-2],observed_arm=rawobs.points(t[2:-2]),**states)
        (OUT/f'{pitch}_rounding.json').write_text(json.dumps(rounding,indent=2))
    report=dict(records=records,model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        pitch_ids=sorted(set(r['pitch'] for r in records)),ball_speed_used_in_fitting=False,
        interpretation='2x2 processing/structure sensitivity, NOT identified causal fractions of true error. Mobile origins follow observed joint centers with explicit translational actuation, not an anatomical scapulothoracic or muscle model. No mass/inertia/grip/release tuning. Same observed motion, not athlete improvement.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    import sys
    run(sys.argv[1:] or None)
