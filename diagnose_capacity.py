"""Sensitivity of stock-model dynamics; hypothetical inertia is NOT a correction."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from muscle_model_check import load_arm
from muscle_capacity import capacity_for_trajectory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--vendor',type=Path,default=Path('artifacts/vendor'))
    parser.add_argument('--base',type=Path,default=Path('artifacts/pitch-fit/trajectory.npz'))
    parser.add_argument('--temporal',type=Path,default=Path('artifacts/pitch-fit-temporal/trajectory.npz'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/capacity-diagnosis.json'))
    args=parser.parse_args()
    model,_=load_arm(args.vendor)
    names=['clavphant_r','scapphant_r','humphant_r','humphant1_r']
    ids=[model.body(name).id for name in names]
    original=model.body_inertia.copy()
    bodies=[{'body':name,'mass_kg':float(model.body_mass[i]),'inertia_kg_m2':original[i].tolist()} for name,i in zip(names,ids)]
    rows=[]
    for label,path in [('framewise',args.base),('temporal_weight_10',args.temporal)]:
        with np.load(path) as t:
            q=t['qpos'];dt=float(np.diff(t['time'])[0])
            if 'model_joint_ranges' in t and not np.allclose(t['model_joint_ranges'],model.jnt_range):
                raise ValueError('Mismatched model limits')
        for hypothesis in ['stock','hypothetical_phantom_inertia_1e-6']:
            model.body_inertia[:]=original
            if hypothesis!='stock':model.body_inertia[ids]=1e-6
            mujoco.mj_setConst(model,mujoco.MjData(model))
            report=capacity_for_trajectory(model,q,dt)
            qvel=np.gradient(q,dt,axis=0,edge_order=2)
            rows.append({'trajectory':label,'inertia_hypothesis':hypothesis,
                         'max_abs_qvel_rad_s':float(np.abs(qvel).max()),
                         'max_abs_qacc_rad_s2':float(np.abs(np.gradient(qvel,dt,axis=0,edge_order=2)).max()),
                         'mean_frame_torque_rmse_nm':report['torque_residual_rmse_nm_mean'],
                         'max_torque_residual_nm':report['torque_residual_max_nm'],
                         'contacts':report['contact_frame_count']})
    report={'bodies':bodies,'comparisons':rows,'validated_inertia_correction':False,
            'scope':'In-memory numerical sensitivity only. Neither hypothetical inertia nor temporal weight is an anatomical calibration.',
            'next':'Check source OpenSim inertial definitions and anatomical rotation convention before candidate optimization.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
