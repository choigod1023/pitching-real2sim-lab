"""Outcome-free first-pitch anatomy fit, reused-pitch reconstruction checks."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
from scipy.spatial.transform import Rotation as R
import coupled_body_search as body
from trace_error_audit import ObservedPitch,integrate_inputs,kinematics
from torque_arm import derivatives
from coupled_payload import model_for
from full_body_comparison import DATA
OUT=Path(__file__).resolve().parent/'artifacts/baseline-calibration'

def fit_attachment(m,q,observed):
    d=mujoco.MjData(m);local={n:[] for n in ['shoulder_jc','glove_shoulder_jc']}
    for i,p in enumerate(q):
        d.qpos[:]=p;mujoco.mj_forward(m,d);bid=m.body('torso').id;rotation=d.xmat[bid].reshape(3,3);origin=d.xpos[bid]
        for n in local:local[n].append(rotation.T@(observed[n][i]-origin))
    return {n:np.mean(v,axis=0).tolist() for n,v in local.items()}

def scale_mass(m,target):
    ids=[i for i in range(1,m.nbody) if i!=m.body('ball_payload').id]
    factor=target/m.body_mass[ids].sum();m.body_mass[ids]*=factor;m.body_inertia[ids]*=factor
    mujoco.mj_setConst(m,mujoco.MjData(m));return float(factor)

def endpoint(m,q,v,name='ball_center'):
    d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,m.site(name).id)
    return jp@v

def replay(m,qp,dt):
    vp,ap=derivatives(m,qp,dt);q=qp[2:-2];v=vp[2:-2];a=ap[2:-2];d=mujoco.MjData(m);loads=[]
    for qq,vv,aa in zip(q,v,a):
        d.qpos[:]=qq;d.qvel[:]=vv;d.qacc[:]=aa;mujoco.mj_inverse(m,d);loads.append(d.qfrc_inverse.copy())
    pos,vel,info=integrate_inputs(m,q[0],v[0],np.array(loads),dt,1,'foh')
    names,ref=body.positions(m,q);_,actual=body.positions(m,pos)
    rv=endpoint(m,q[-1],v[-1]);sv=endpoint(m,pos[-1],vel[-1])
    return dict(reference_speed_kmh=float(np.linalg.norm(rv)*3.6),simulated_speed_kmh=float(np.linalg.norm(sv)*3.6),
                integration_speed_abs_difference_kmh=float(abs(np.linalg.norm(rv)-np.linalg.norm(sv))*3.6),
                integration_rmse_mm=float(np.sqrt(np.mean(np.sum((actual-ref)**2,axis=2)))*1000),
                peak_external_force_n=float(np.linalg.norm(np.array(loads)[:,:3],axis=1).max()),
                peak_external_torque_nm=float(np.linalg.norm(np.array(loads)[:,3:6],axis=1).max()),state_resets=info['state_resets']),pos,vel,q,v,names,ref

def summarize(records):
    out={}
    for mode in ['nominal','mass_only','frozen_geometry_mass']:
        rows=[r for r in records if r['mode']==mode and r['hz']==2880]
        train=next(r for r in rows if r['split']=='calibration');test=[r for r in rows if r['split']=='reused_test']
        errors=[abs((r['simulated_speed_kmh']-train['simulated_speed_kmh'])-(r['measured_speed_kmh']-train['measured_speed_kmh'])) for r in test]
        out[mode]=dict(n_test=len(test),absolute_mae_kmh=float(np.mean([r['absolute_speed_error_kmh'] for r in test])),
                       change_from_first_pitch_mae_kmh=float(np.mean(errors)),
                       no_change_mae_kmh=float(np.mean([abs(r['measured_speed_kmh']-train['measured_speed_kmh']) for r in test])))
    return out

def run(frame_policy="segment_axes",reconstruct_rounding=False):
    OUT.mkdir(parents=True,exist_ok=True);obs={p:ObservedPitch(p,reconstruct_rounding=reconstruct_rounding) for p in ['3034_1','3034_2','3034_3','3034_4','3034_5']}
    metadata=pd.read_csv(DATA/'metadata.csv').set_index('session_pitch');target=float(metadata.loc['3034_1','session_mass_kg'])
    train,q,dt,t,p=body.build(observed_pitch=obs['3034_1'],pitch='3034_1',padding=2,output_dir=OUT/'train',frame_policy=frame_policy,reconstruct_rounding=reconstruct_rounding)
    offsets=fit_attachment(train,q[2:-2],{n:v[2:-2] for n,v in p.items()})
    train.body_pos[train.body('upper').id]=offsets['shoulder_jc'];train.body_pos[train.body('glove_upper').id]=offsets['glove_shoulder_jc']
    factor=scale_mass(train,target);mujoco.mj_saveLastXML(str(OUT/'calibrated.xml'),train)
    calibration=dict(training_pitch='3034_1',body_mass_kg=target,mass_factor=factor,attachments=offsets,speed_used=False,geometry_frozen_across_test_pitches=True)
    records=[]
    for pitch,o in obs.items():
        resolutions=[2880,5760] if pitch=='3034_4' else [2880]
        for hz in resolutions:
            raw,qp,dt,t,points=body.build(1/hz,pitch,o,2,OUT/'scratch',frame_policy=frame_policy,reconstruct_rounding=reconstruct_rounding)
            # Recreate separate models before calibration mutations.
            models={'nominal':raw,'mass_only':mujoco.MjModel.from_xml_path(str(OUT/'scratch/model.xml')),
                    'frozen_geometry_mass':mujoco.MjModel.from_xml_path(str(OUT/'calibrated.xml'))}
            scale_mass(models['mass_only'],target)
            for mode,m in models.items():
                r,pos,vel,q,v,names,ref=replay(m,qp,dt)
                r.update(pitch=pitch,mode=mode,hz=hz,split='calibration' if pitch=='3034_1' else 'reused_test',
                         measured_speed_kmh=o.ledger['recorded_speed_kmh'],mass_kg=float(m.body_mass.sum()-.145))
                r['absolute_speed_error_kmh']=abs(r['simulated_speed_kmh']-r['measured_speed_kmh'])
                r['fit_rmse_mm']={n:float(np.sqrt(np.mean(np.sum((ref[:,names.index(n)]-points[n][2:-2])**2,axis=1)))*1000) for n in names if n in points}
                # Free-arm replay reference uses EXACTLY the same lengths and arm world rotations.
                lengths=[-m.body_pos[m.body('forearm').id,2],-m.body_pos[m.body('hand').id,2]]
                am=model_for(lengths,dt);aq=o.sample(t);av,_=derivatives(am,aq,dt);_,ballv,_=kinematics(am,aq[-3:-2],av[-3:-2])
                r['free_arm_reference_speed_kmh']=float(np.linalg.norm(ballv[0,3])*3.6)
                r['free_arm_residual_to_measured_kmh']=r['measured_speed_kmh']-r['free_arm_reference_speed_kmh']
                r['full_body_reference_minus_free_arm_kmh']=r['reference_speed_kmh']-r['free_arm_reference_speed_kmh']
                records.append(r)
                if pitch=='3034_4' and hz==5760 and mode in ['nominal','frozen_geometry_mass']:
                    np.savez_compressed(OUT/f'{mode}_states.npz',qpos=pos,qvel=vel,time=t[2:-2],reference=q)
                    shutil.copyfile(OUT/('scratch/model.xml' if mode=='nominal' else 'calibrated.xml'),OUT/f'{mode}.xml')
    # Frame/unit/contact-point audit, NOT a contact-force implementation.
    force=pd.read_csv(DATA/'full_sig/force_plate.csv');force=force[force.session_pitch.isin(obs)]
    force_audit=dict(columns=force.columns.tolist(),has_cop=any('cop' in x.lower() for x in force.columns),has_moment=any('moment' in x.lower() for x in force.columns),
                     observation='Only rear/lead force xyz in this processed table. Contact points and free moments not supplied here; raw C3D channels must be audited next.')
    report=dict(rounding_reconstruction={p:o.ledger["rounding_reconstruction"] for p,o in obs.items()},frame_policy=frame_policy,calibration=calibration,records=records,summary=summarize(records),force_audit=force_audit,ready_for_gain_optimization=False,adopt_as_default=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for r in records:
        print(r['pitch'],r['hz'],r['mode'],'speed',round(r['simulated_speed_kmh'],4),'gap',round(r['absolute_speed_error_kmh'],4),'shoulder',round(r['fit_rmse_mm']['shoulder_jc'],2),'integration',round(r['integration_speed_abs_difference_kmh'],5),'free-arm',round(r['free_arm_reference_speed_kmh'],4))
    return report
if __name__=='__main__':run()
