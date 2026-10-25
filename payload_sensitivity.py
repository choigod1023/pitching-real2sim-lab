"""Predeclared engineering probes; never select a setting by observed speed."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from coupled_payload import model_for, inverse_and_rollout, run as prepare

# Numeric stress probes, not anatomical or statistical uncertainty bounds.
SCENARIOS = [('nominal',(.15,0.,0.),1.), ('grip_x13',(.13,0.,0.),1.),
             ('grip_x17',(.17,0.,0.),1.), ('grip_y_minus',(.15,-.02,0.),1.),
             ('grip_y_plus',(.15,.02,0.),1.), ('grip_z_minus',(.15,0.,-.02),1.),
             ('grip_z_plus',(.15,0.,.02),1.), ('hand_mass_0.8',(.15,0.,0.),.8),
             ('hand_mass_1.2',(.15,0.,0.),1.2)]
FILTERS = [0.,10.,20.,30.]


def run(data, pitch, out):
    data, out = Path(data), Path(out)
    prepare(data,pitch,out/'reference20',steps=(1/5760,))
    with np.load(out/'reference20/reference-0.npz') as r:
        q, lengths, dt, nominal = r['qpos'].copy(),r['lengths'].copy(),float(r['dt']),r['required_loads'].copy()
    cases=[]
    for name, grip, mass_scale in SCENARIOS:
        model = model_for(lengths,dt,grip=grip,hand_mass_scale=mass_scale)
        required,_ = inverse_and_rollout(model,q,dt)
        # Euler showed large resolution-dependent changes in perturbed rollouts.
        # Retain it in the diagnostic below; use RK4 for reported fixed inputs.
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        fixed,_ = inverse_and_rollout(model,q,dt,applied_loads=nominal)
        cases.append({'name':name,'grip_m':list(grip),'hand_mass_scale':mass_scale,
                      'recomputed_demand':required,'fixed_nominal_loads':fixed})
    processing=[]
    for cutoff in FILTERS:
        report=prepare(data,pitch,out/f'filter-{cutoff:g}',cutoff_hz=cutoff,steps=(1/5760,))
        processing.append({'cutoff_hz':cutoff,'result':report['runs'][0]['with_ball_mass']})
    prepare(data,pitch,out/'fine20',steps=(1/11520,))
    refinements=[]
    for folder in ['reference20','fine20']:
        with np.load(out/folder/'reference-0.npz') as r:
            q2,lengths2,dt2=r['qpos'].copy(),r['lengths'].copy(),float(r['dt'])
        t=np.arange(len(q))*dt;u=np.arange(len(q2))*dt2
        loads=np.column_stack([np.interp(u,t,nominal[:,j]) for j in range(nominal.shape[1])])
        for name,grip in [('nominal',(.15,0.,0.)),('grip_z_plus',(.15,0.,.02)),('grip_y_minus',(.15,-.02,0.))]:
            for integrator in [mujoco.mjtIntegrator.mjINT_EULER,mujoco.mjtIntegrator.mjINT_RK4]:
                m=model_for(lengths2,dt2,grip=grip);m.opt.integrator=integrator
                value,_=inverse_and_rollout(m,q2,dt2,applied_loads=loads)
                refinements.append({'resolution':folder,'scenario':name,'integrator':int(integrator),'result':value})
    report={'pitch':pitch,'cases':cases,'processing':processing,'refinement':refinements,
            'solver_policy':'Recomputed-reference cases retain Euler; fixed-load cases use RK4. Euler and RK4 coarse/fine diagnostics are retained for three selected scenarios. No physiological validity implied.',
            'scope':'Exploratory engineering sensitivity on previously inspected trials, not biological bounds or candidate motion optimization.',
            'fixed_loads':'Same initial arm state and nominal generalized force time series. Changed grip/mass can change mechanical work; equal force input is not an equal-work or muscle-capacity guarantee.',
            'filter_scope':'Changes hand and RFRM markers only; processed joint centers remain fixed. Initial states and required inputs differ between filters. Not an equal-input comparison.',
            'selection':'All declared settings reported; observed ball speed not used to select any setting.',
            'predicted_gain_kmh':None,'coaching_ready':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    p.add_argument('--pitch',default='3034_1');p.add_argument('--out',type=Path,default=Path('artifacts/payload-sensitivity'))
    a=p.parse_args();r=run(a.data,a.pitch,a.out)
    print(json.dumps({'pitch':r['pitch'],'scenarios':len(r['cases']),'filters':len(r['processing'])}))
