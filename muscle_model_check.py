"""Exercise a pinned public muscle model with mj_step, not recorded-qpos playback."""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings
import mujoco
import numpy as np


def load_arm(vendor):
    if vendor.is_dir():
        sys.path.insert(0, str(vendor.resolve()))
    import myo_sim
    if myo_sim.__version__ != '0.2.3':
        raise RuntimeError('This audit is pinned to myo-sim 0.2.3')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = myo_sim.load_model('myoarm_r')
    return model, [str(w.message) for w in caught]


def rollout(model, control, seconds=1.):
    data = mujoco.MjData(model)
    start = data.qpos.copy()
    for _ in range(round(seconds/model.opt.timestep)):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError('Nonfinite muscle simulation')
        if data.warning.number.any():
            raise RuntimeError(f'MuJoCo runtime warning: {data.warning.number.tolist()}')
    if abs(data.time-seconds) > model.opt.timestep:
        raise RuntimeError('Unexpected simulation time reset')
    return data, {'seconds': data.time, 'uniform_excitation': control,
                  'max_abs_qpos_change_rad': float(np.max(np.abs(data.qpos-start))),
                  'runtime_warning_counts': data.warning.number.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor', type=Path, default=Path(__file__).resolve().parent/'artifacts/vendor')
    parser.add_argument('--output', type=Path, default=Path('artifacts/muscle-model'))
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    model, compile_warnings = load_arm(args.vendor)
    passive, a = rollout(model, 0.)
    active, b = rollout(model, .05)
    difference = float(np.linalg.norm(active.qpos-passive.qpos))
    if difference < 1e-8:
        raise RuntimeError('Excitation did not change the simulated trajectory')
    report = {'model': 'myoarm_r', 'myo_sim_version': '0.2.3', 'mujoco_version': mujoco.__version__,
              'source': 'https://github.com/MyoHub/myo_sim', 'license': 'Apache-2.0',
              'dofs': model.nv, 'actuators': model.nu, 'compile_warnings': compile_warnings,
              'passive': a, 'excited': b, 'control_response_qpos_l2_rad': difference,
              'recorded_pitch_tracking': False, 'ball_model': False,
              'sports_prediction_validated': False,
              'scope': 'Numerical smoke check of public muscle dynamics, not pitching or maximum strength validation.',
              'joints': [{'name': model.joint(i).name, 'limited': bool(model.jnt_limited[i]),
                          'range_rad': model.jnt_range[i].tolist()} for i in range(model.njnt)]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.render:
        from PIL import Image
        display = mujoco.MjData(model)
        mujoco.mj_forward(model, display)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = display.xpos[model.body('humerus_r').id]
        camera.distance = 1.8
        camera.azimuth = 135
        camera.elevation = -15
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(display, camera=camera)
            Image.fromarray(renderer.render()).save(args.output/'public-arm.png')
    print(json.dumps({k: v for k,v in report.items() if k != 'joints'}, indent=2))
    if args.gui:
        from mujoco import viewer as mjviewer
        display = mujoco.MjData(model)
        with mjviewer.launch_passive(model, display) as viewer:
            while viewer.is_running():
                started = time.monotonic()
                display.ctrl[:] = .05
                mujoco.mj_step(model, display)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep-(time.monotonic()-started)))


if __name__ == '__main__':
    main()
