"""Hand-only release observability; target-free grip perturbations, not coaching."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from coupled_payload import model_for, inverse_and_rollout, run as prepare
from torque_arm import derivatives

BASE = np.array([.15, 0., 0.])
RADIUS = .01  # engineering perturbation, not a measured physiological interval
CASES = [('nominal', BASE)] + [(f'{axis}{sign:+d}', BASE+sign*RADIUS*np.eye(3)[i])
                              for i, axis in enumerate('xyz') for sign in (-1, 1)]


def cross_matrix(w):
    x,y,z = w
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def velocity_map(lengths, dt, q, vel):
    model = model_for(lengths, dt, grip=(0.,0.,0.))
    d = mujoco.MjData(model); d.qpos[:] = q; d.qvel[:] = vel
    mujoco.mj_forward(model,d)
    jp,jr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,d,jp,jr,model.site('ball_center').id)
    rotation = d.xmat[model.body('hand').id].reshape(3,3).copy()
    omega = jr@vel
    return jp@vel, cross_matrix(omega)@rotation, rotation, omega


def audit_map(vw, a, rotation, omega):
    nominal = vw+a@BASE
    null = rotation.T@omega
    if np.linalg.norm(null)<1e-8: raise ValueError('No rotation for null-axis demonstration')
    null *= RADIUS/np.linalg.norm(null)
    alternatives = [{'name':name,'grip_m':r.tolist(),'speed_kmh':float(np.linalg.norm(vw+a@r)*3.6)}
                    for name,r in CASES]
    return {'scenarios':alternatives,
            'velocity_map_singular_values_per_s':np.linalg.svd(a,compute_uv=False).tolist(),
            'speed_gradient_kmh_per_m':(3.6*nominal@a/np.linalg.norm(nominal)).tolist(),
            'all_radius_ball_speed_upper_bound_kmh':float(3.6*(np.linalg.norm(nominal)+np.linalg.norm(a,2)*RADIUS)),
            'null_axis_grip_m':(BASE+null).tolist(),
            'null_axis_position_difference_mm':float(np.linalg.norm(rotation@null)*1000),
            'null_axis_velocity_difference_mps':float(np.linalg.norm(a@null)),
            'hand_observation_jacobian_wrt_grip':np.zeros((9,3)).tolist()}


def run(root=Path(__file__).resolve().parent.parent):
    out=Path('artifacts/grip-identifiability');out.mkdir(parents=True,exist_ok=True)
    data=root/'openbiomechanics/baseball_pitching/data'
    cohort=root/'cohort-baseline/artifacts/cohort-baseline'
    meta=pd.read_csv(data/'metadata.csv').set_index('session_pitch')
    selection=pd.read_csv(cohort/'selection.csv')
    chosen=selection[selection.user.eq(selection.user.min())]
    if chosen.session_pitch.tolist()!=['3034_1','3034_2','3034_3']:raise ValueError('Selection changed')
    records=[];rollouts=[]
    for pitch in chosen.session_pitch:
        for cutoff in (0,20):
            path=cohort/'cases'/pitch/str(cutoff)/'reference-0.npz'
            with np.load(path) as ref:
                q=ref['qpos'].copy();lengths=ref['lengths'].copy();dt=float(ref['dt'])
            model=model_for(lengths,dt);vel,_=derivatives(model,q,dt)
            vw,a,rotation,omega=velocity_map(lengths,dt,q[-1],vel[-1])
            record=audit_map(vw,a,rotation,omega)
            # Independent MuJoCo Jacobian check for all seven grip states.
            discrepancies=[];hand_positions=[]
            for _,grip in CASES:
                m=model_for(lengths,dt,grip=grip);d=mujoco.MjData(m)
                d.qpos[:]=q[-1];d.qvel[:]=vel[-1];mujoco.mj_forward(m,d)
                jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
                mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
                discrepancies.append(np.linalg.norm(jp@d.qvel-(vw+a@grip)))
                hand_positions.append(d.xpos[m.body('hand').id].copy())
            record.update(pitch=pitch,user=2,cutoff_hz=cutoff,
                          observed_kmh=float(meta.loc[pitch].pitch_speed_mph)*1.609344,
                          input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          max_direct_jacobian_disagreement_mps=float(max(discrepancies)),
                          hand_origin_difference_m=float(np.max(np.abs(np.array(hand_positions)-hand_positions[0]))))
            records.append(record)
            if cutoff==20:
                for name,grip in CASES[:3]:  # nominal, x-1cm, x+1cm; all declared, no speed selection
                    m=model_for(lengths,dt,grip=grip);m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
                    result,_=inverse_and_rollout(m,q,dt)
                    rollouts.append(dict(pitch=pitch,resolution='coarse',dt_s=dt,name=name,result=result))
    # One pitch refinement explicitly checks that the example is not an integration artifact.
    prepare(data,'3034_1',out/'fine-reference',cutoff_hz=20,steps=(1/11520,))
    with np.load(out/'fine-reference/reference-0.npz') as ref:
        q=ref['qpos'].copy();lengths=ref['lengths'].copy();dt=float(ref['dt'])
    for name,grip in CASES[:3]:
        m=model_for(lengths,dt,grip=grip);m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
        result,_=inverse_and_rollout(m,q,dt)
        rollouts.append(dict(pitch='3034_1',resolution='fine',dt_s=dt,name=name,result=result))
    stacked=[]
    for cutoff in (0,20):
        jac=np.array([r['speed_gradient_kmh_per_m'] for r in records if r['cutoff_hz']==cutoff])
        stacked.append({'cutoff_hz':cutoff,'rank':int(np.linalg.matrix_rank(jac)),
                        'singular_values':np.linalg.svd(jac,compute_uv=False).tolist()})
    report={'records':records,'rollouts':rollouts,'shared_grip_three_pitch_speed_jacobian':stacked,
            'scope':'Hand-only kinematic nonidentifiability. Scalar radar speed is a separate observed constraint. No grip fitted to radar. A full-rank shared-grip Jacobian can permit local identification under that additional assumption; not proof of a fitting solution.',
            'radius_m':RADIUS,'radius_scope':'Euclidean engineering perturbation, not biological bounds or measurement uncertainty.',
            'dynamics':'RK4 force rollout with inverse forces recomputed per grip. Same prescribed hand motion, not same forces, work, muscle capacity, or a motion intervention.',
            'null_scope':'Exact instantaneous null-axis change preserves full velocity but changes ball position. Does not establish multiple radar-matching solutions when nominal speed disagrees with radar.',
            'predicted_gain_kmh':None,'coaching_ready':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'examples':[{k:r[k] for k in ('pitch','cutoff_hz','observed_kmh','scenarios','all_radius_ball_speed_upper_bound_kmh')} for r in records], 'rollouts':rollouts,'shared_grip':stacked},indent=2))
    return report


if __name__=='__main__':run()
