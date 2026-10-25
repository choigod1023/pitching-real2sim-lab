"""Predeclared synthetic contact/convergence audit, not a pitching experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
from finger_contact import ContactParams,contact,rk4_step,totals

ROOT=Path(__file__).resolve().parent


def simulate(mu,dt):
    p=ContactParams(friction=mu)
    s=np.zeros((2,9));s[1,0]=-sum(p.radii)+.0002;s[1,3:6]=[1.,2.,0.]
    first=totals(s,p);records=[];states=[];last_energy=first['energy_j']
    max_rise=0.;max_p=0.;max_l=0.;max_violation=0.;min_dissipation=0.
    for i in range(round(.02/dt)+1):
        c=contact(s,p);a=totals(s,p)
        max_p=max(max_p,float(np.linalg.norm(a['linear_momentum']-first['linear_momentum'])))
        max_l=max(max_l,float(np.linalg.norm(a['angular_momentum']-first['angular_momentum'])))
        max_rise=max(max_rise,a['energy_j']-last_energy);last_energy=a['energy_j']
        max_violation=max(max_violation,float(np.linalg.norm(c['tangent_force_n'])-mu*c['normal_force_n']))
        min_dissipation=min(min_dissipation,c['dissipation_w'])
        assert c['normal_force_n']>=0
        records.append([i*dt,c['normal_force_n'],c['penetration_m'],a['energy_j'],np.linalg.norm(s[0,3:6])*3.6])
        states.append(s.copy())
        if i<round(.02/dt):s=rk4_step(s,dt,p)
    history=np.array(records)
    active=np.flatnonzero(history[:,1]>0)
    force_free=int(active[-1]+1)
    assert force_free<len(history) and history[-1,2]==0
    geometric_free=int(np.flatnonzero(history[:,2]>0)[-1]+1)
    report={'mu':mu,'dt_s':dt,'ball_final_speed_kmh':float(history[-1,4]),
            'ball_final_spin_rad_s':float(np.linalg.norm(s[0,6:9])),
            'last_force_to_zero_interval_s':[float(history[force_free-1,0]),float(history[force_free,0])],
            'last_overlap_to_zero_interval_s':[float(history[geometric_free-1,0]),float(history[geometric_free,0])],
            'max_penetration_mm':float(history[:,2].max()*1000),
            'energy_change_j':float(history[-1,3]-first['energy_j']),
            'max_step_energy_increase_j':max_rise,'max_linear_momentum_residual':max_p,
            'max_angular_momentum_residual':max_l,'max_friction_bound_violation_n':max_violation,
            'minimum_dissipation_w':min_dissipation}
    return report,history,np.array(states)


def run():
    out=ROOT/'artifacts/finger-contact';out.mkdir(parents=True,exist_ok=True)
    results=[];convergence=[]
    for mu in (0.,.4,.8):
        speeds=[]
        for dt in (1e-4,5e-5,2.5e-5,1.25e-5):
            r,h,s=simulate(mu,dt);results.append(r);speeds.append(r['ball_final_speed_kmh'])
            np.savez(out/f'mu-{mu}-dt-{dt}.npz',history=h,states=s)
            assert r['max_linear_momentum_residual']<1e-10
            assert r['max_angular_momentum_residual']<1e-9
            assert r['max_friction_bound_violation_n']<1e-12 and r['minimum_dissipation_w']>=-1e-12
        convergence.append({'mu':mu,'successive_resolution_speed_difference_kmh':np.abs(np.diff(speeds)).tolist(),
                            'finest_pair_difference_below_0_01_kmh':bool(abs(speeds[-1]-speeds[-2])<.01)})
    report={'results':results,'convergence':convergence,'protocol_sha256':hashlib.sha256((ROOT/'CONTACT_PROTOCOL.md').read_bytes()).hexdigest(),
            'scope':'Two free spheres; no athlete input or observed ball trajectory. Numerical verification only.',
            'baseball_speed_prediction_mae_kmh':None,'validated_gain_kmh':None,'connected_to_arm_model':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'finest_cases':[r for r in results if r['dt_s']==1.25e-5],'convergence':convergence},indent=2))


if __name__=='__main__':run()
