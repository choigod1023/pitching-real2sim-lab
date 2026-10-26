"""Independent saved-state and RK4 replay check for the joint demand experiment."""
import json
from pathlib import Path
import mujoco
import numpy as np
from coupled_payload import model_for


def audit(folder=Path('artifacts/joint-intervention')):
    report=json.loads((folder/'report.json').read_text())
    rows=[]
    for run in report['runs']:
        s=np.load(folder/f"state-{run['case']}-sub{run['substeps']}.npz")
        m=model_for(s['lengths'],float(s['dt'])/run['substeps'])
        m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
        d=mujoco.MjData(m)
        d.qpos[:]=s['qpos'][0]; d.qvel[:]=s['qvel'][0]
        mujoco.mj_forward(m,d); mujoco.mj_energyPos(m,d); mujoco.mj_energyVel(m,d)
        initial_energy=float(d.energy.sum())
        for force in s['controls'][:-1]:
            d.qfrc_applied[:]=force
            for _ in range(run['substeps']): mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
        speed=float(np.linalg.norm(jp@d.qvel)*3.6)
        force_peak=np.max(np.abs(s['controls'][:-1]),axis=0)
        rows.append(dict(case=run['case'],substeps=run['substeps'],
            replay_qpos_max_difference=float(np.max(np.abs(d.qpos-s['qpos'][-1]))),
            replay_qvel_max_difference=float(np.max(np.abs(d.qvel-s['qvel'][-1]))),
            endpoint_speed_difference_kmh=speed-run['speed_kmh'],
            initial_energy_difference_j=initial_energy-run['initial_energy_j'],
            peak_force_difference=float(np.max(np.abs(force_peak-run['peak_abs_generalized_force_per_dof'])))))
    keys=[k for k in rows[0] if k not in ('case','substeps')]
    maxima={k:max(abs(r[k]) for r in rows) for k in keys}
    assert all(v<1e-9 for v in maxima.values()),maxima
    result={'independent_replays':len(rows),'terminal_control_excluded':True,
            'all_maxima':maxima,'rows':rows,
            'scope':'Reproduces dynamics and recorded quantities; does not establish biological validity or intervention accuracy.'}
    out=Path('artifacts/joint-review');out.mkdir(parents=True,exist_ok=True)
    (out/'independent_replay.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'independent_replays':len(rows),'maxima':maxima}))


if __name__=='__main__': audit()
