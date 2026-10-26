"""Training-only launch-state estimation followed by actual free-ball RK4."""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco
from heldout_speed_comparison import fit_models, predict, evaluate

ROOT=Path(__file__).resolve().parent


def launch_states(times, ball, origin, rotation):
    fitted=fit_models(times,ball,origin,rotation)
    training_prediction=predict(fitted,times,origin,rotation)
    states={name:(positions[-1].copy(),(positions[-1]-positions[-2])/(times[-1]-times[-2])) for name,positions in training_prediction.items()}
    states['C']=(training_prediction['C'][-1].copy(),fitted['C'][1].copy())
    return states


def free_flight(position,velocity,gravity,frame_dt=1/360,substeps=1,nframes=3):
    gravity=np.asarray(gravity,dtype=float)
    xml=f'<mujoco><option timestep="{frame_dt/substeps:.17g}" integrator="RK4" gravity="{gravity[0]} {gravity[1]} {gravity[2]}"/><worldbody><body><freejoint/><geom type="sphere" size="0.0366" mass="0.145" contype="0" conaffinity="0"/></body></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
    data.qpos[:3]=position; data.qpos[3:]=[1,0,0,0]
    data.qvel[:3]=velocity
    mujoco.mj_forward(model,data)
    positions=[]; velocities=[]
    for _ in range(nframes):
        for _ in range(substeps):
            mujoco.mj_step(model,data)
        positions.append(data.qpos[:3].copy()); velocities.append(data.qvel[:3].copy())
    return np.array(positions),np.array(velocities)


def run():
    import ezc3d
    source=ROOT.parent/'hand-ball-link/artifacts/hand-ball/observation.json'
    observed=json.loads(source.read_text())
    body=Path(observed['source_sha256']['body']['path'])
    alignment_path=Path(observed['source_sha256']['alignment']['path'])
    alignment=json.loads(alignment_path.read_text()); q=np.asarray(alignment['rotation'])
    gravity=q@np.array([0.,0.,-9.81])
    c3d=ezc3d.c3d(str(body)); labels=c3d['parameters']['ROTATION']['LABELS']['value']
    rotation=c3d['data']['rotations'][:3,:3,labels.index('l_hand_4X4'),:]
    frames=np.arange(973,980); rate=observed['rate_hz']; times=(frames-973)/rate
    rows=[]
    for row in observed['conditions']:
        ball=np.array([p['ball_rig_m'] for p in row['positions']])
        origin=np.array([p['hand_origin_rig_m'] for p in row['positions']])
        rotations=np.einsum('ij,jkn->nik',q,rotation[:,:,frames[:4]+row['c3d_frame_offset']])
        states=launch_states(times[:4],ball[:4],origin[:4],rotations)
        conditions={}
        for name,(position,velocity) in states.items():
            coarse,vc=free_flight(position,velocity,gravity,1/rate,1)
            fine,vf=free_flight(position,velocity,gravity,1/rate,2)
            metrics=evaluate(fine,ball[4:],1/rate,62.7*1.609344)
            times_future=np.arange(1,4)/rate
            analytical=position+times_future[:,None]*velocity+.5*times_future[:,None]**2*gravity
            metrics.update({'launch_position_m':position.tolist(),'launch_velocity_m_s':velocity.tolist(),
                            'launch_speed_kmh':float(np.linalg.norm(velocity)*3.6),
                            'coarse_fine_max_position_difference_m':float(np.max(np.linalg.norm(coarse-fine,axis=1))),
                            'coarse_fine_max_velocity_difference_m_s':float(np.max(np.linalg.norm(vc-vf,axis=1))),
                            'analytic_max_position_difference_m':float(np.max(np.linalg.norm(analytical-fine,axis=1)))})
            conditions[name]=metrics
        rows.append({'threshold':row['threshold'],'offset':row['c3d_frame_offset'],'metrics':conditions,
                     'A_minus_B_speed_discrepancy_kmh':conditions['A']['mean_absolute_speed_discrepancy_kmh']-conditions['B']['mean_absolute_speed_discrepancy_kmh']})
    report={'scope':'Exploratory stage fixed after first kinematic results, before these outcomes. Training launch at976 is not verified release. Actual MuJoCo free sphere integration, not full athlete physics. Future hand excluded. No claim old3034_1 gap reduction.',
            'training_frames':frames[:4].tolist(),'evaluation_frames':frames[4:].tolist(),
            'gravity_rig_m_s2':gravity.tolist(),'lab_z_up_assumed':True,'mass_kg':.145,'frame_rate_hz':rate,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,body,alignment_path,ROOT/'COMPARISON_PROTOCOL.md',ROOT/'PHYSICAL_COMPARISON_PROTOCOL.md']},'conditions':rows}
    out=ROOT/'artifacts/physical-comparison';out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in rows:
        print(row['threshold'],row['offset'],json.dumps({k:round(v['mean_absolute_speed_discrepancy_kmh'],6) for k,v in row['metrics'].items()}))
    return report


if __name__=='__main__':run()
