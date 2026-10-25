"""No-impulse detachment from an integrated arm/hand/ball state."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from coupled_payload import model_for


def totals(model, data):
    mujoco.mj_forward(model, data)
    linear, angular, weighted_com = np.zeros(3), np.zeros(3), np.zeros(3)
    for body in range(1, model.nbody):
        mass = model.body_mass[body]
        if mass == 0:
            continue
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jp, jr, body)
        velocity, omega = jp@data.qvel, jr@data.qvel
        rotation = data.ximat[body].reshape(3, 3)
        inertia = rotation@np.diag(model.body_inertia[body])@rotation.T
        momentum = mass*velocity
        linear += momentum
        angular += np.cross(data.xipos[body], momentum)+inertia@omega
        weighted_com += mass*data.xipos[body]
    mujoco.mj_energyPos(model, data); mujoco.mj_energyVel(model, data)
    return {'mass_kg': float(model.body_mass.sum()), 'linear': linear, 'angular': angular,
            'com': weighted_com/model.body_mass.sum(), 'energy_j': float(data.energy.sum())}


def detach(lengths, qpos, qvel, dt):
    attached = model_for(lengths, dt)
    before = mujoco.MjData(attached)
    before.qpos[:] = qpos; before.qvel[:] = qvel
    mujoco.mj_forward(attached, before)
    sid = attached.site('ball_center').id
    jp, jr = np.zeros((3, attached.nv)), np.zeros((3, attached.nv))
    mujoco.mj_jacSite(attached, before, jp, jr, sid)
    velocity, omega_world = jp@before.qvel, jr@before.qvel
    hand = attached.body('hand').id
    rotation = before.xmat[hand].reshape(3, 3)
    split = model_for(lengths, dt, free_ball=True)
    # RK4 only for passive post-release diagnostic, no new arm controls.
    split.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    after = mujoco.MjData(split)
    after.qpos[:attached.nq] = qpos; after.qvel[:attached.nv] = qvel
    address = split.jnt_qposadr[split.joint('ball_free').id]
    dof = split.jnt_dofadr[split.joint('ball_free').id]
    after.qpos[address:address+3] = before.site_xpos[sid]
    after.qpos[address+3:address+7] = before.xquat[hand]
    after.qvel[dof:dof+3] = velocity
    # MuJoCo free-joint angular velocity is body-local; jacobian omega is world.
    after.qvel[dof+3:dof+6] = rotation.T@omega_world
    a, b = totals(attached, before), totals(split, after)
    report = {'mass_jump_kg': b['mass_kg']-a['mass_kg'],
              'energy_jump_j': b['energy_j']-a['energy_j'],
              'linear_momentum_jump_kg_mps': float(np.linalg.norm(b['linear']-a['linear'])),
              'angular_momentum_jump_kg_m2ps': float(np.linalg.norm(b['angular']-a['angular'])),
              'position_jump_m': float(np.linalg.norm(after.qpos[address:address+3]-before.site_xpos[sid])),
              'velocity_jump_mps': float(np.linalg.norm(after.qvel[dof:dof+3]-velocity)),
              'conditional_release_speed_kmh': float(np.linalg.norm(velocity)*3.6),
              'angular_velocity_world_rad_s': omega_world.tolist()}
    return split, after, report


def passive_check(model, data, duration=.05):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Positive finite duration required')
    initial = totals(model, data)
    address = model.jnt_qposadr[model.joint('ball_free').id]
    dof = model.jnt_dofadr[model.joint('ball_free').id]
    position = data.qpos[address:address+3].copy(); velocity = data.qvel[dof:dof+3].copy()
    steps = int(np.ceil(duration/model.opt.timestep)); model.opt.timestep = duration/steps
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if data.warning.number.any() or not np.isfinite(data.qpos).all():
            raise RuntimeError('Unstable detached simulation')
    final = totals(model, data)
    gravity = model.opt.gravity
    force = initial['mass_kg']*gravity
    expected_p = initial['linear']+force*duration
    expected_l = initial['angular']+np.cross(initial['com'], force)*duration
    expected_l += .5*np.cross(initial['linear']/initial['mass_kg'], force)*duration**2
    expected_ball = position+velocity*duration+.5*gravity*duration**2
    return {'duration_s': float(data.time), 'gravity_corrected_linear_residual_kg_mps': float(np.linalg.norm(final['linear']-expected_p)),
            'gravity_corrected_angular_residual_kg_m2ps': float(np.linalg.norm(final['angular']-expected_l)),
            'mechanical_energy_drift_j': final['energy_j']-initial['energy_j'],
            'ball_ballistic_error_m': float(np.linalg.norm(data.qpos[address:address+3]-expected_ball)),
            'post_release_controls': 'All zero; passive numerical audit, not realistic follow-through.'}


def run(source, out):
    source, out = Path(source), Path(out)
    with np.load(source) as state:
        model, data, report = detach(state['lengths'], state['qpos'], state['qvel'], float(state['dt']))
        fine_model, fine_data, _ = detach(state['lengths'], state['qpos'], state['qvel'], float(state['dt'])/2)
    report['passive_50ms'] = passive_check(model, data)
    report['passive_50ms_half_step'] = passive_check(fine_model, fine_data)
    report['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    report['input_path'] = str(source.resolve())
    report['scope'] = 'No-impulse state transfer from actual forward-integrated attached model. Generic hand, assumed grip, no air forces, no measured ball state or physiological validation.'
    report['predicted_gain_kmh'] = None
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('artifacts/release-conservation'))
    a = p.parse_args(); print(json.dumps(run(a.source,a.out), indent=2))
