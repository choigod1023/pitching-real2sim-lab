"""Finite-grid release-level tradeoffs, not body-to-spin or athlete gain predictions."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from spin_flight import (MASS,RADIUS,INERTIA,G,release_energy,energy_speed,velocities,
                         spin_vectors,flight_batch,aim)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/performance-frontier'
E0=float(release_energy(130.,2400.,1.8))


def states(rows,angles,noise):
    """Same standardized stress and same energy in every candidate for each draw.

    Columns: energy1J, rpm50, az/el .15deg, spin rotation xyz2deg,
    release x/y/z5mm. Synthetic, not measured athlete distributions.
    """
    n,s=len(rows),len(noise);z=np.broadcast_to(np.asarray(noise,float),(n,s,10))
    def col(k):return rows[k].to_numpy(float)[:,None]
    p=np.stack((z[:,:,7]*.005,col('forward_m')+z[:,:,8]*.005,col('height_m')+z[:,:,9]*.005),axis=-1)
    rpm=col('rpm')+z[:,:,1]*50;energy=E0+z[:,:,0]
    speed=energy_speed(energy,rpm,p[:,:,2])
    az=angles[:,0,None]+z[:,:,2]*.15;el=angles[:,1,None]+z[:,:,3]*.15
    v=velocities(speed,az,el)
    nominal_direction=velocities(np.ones((n,s))*130,angles[:,0,None],angles[:,1,None])
    w=spin_vectors(nominal_direction.reshape(-1,3),rpm.ravel(),
                   np.broadcast_to(col('efficiency'),(n,s)).ravel(),np.broadcast_to(col('tilt_deg'),(n,s)).ravel())
    w=Rotation.from_rotvec((z[:,:,4:7]*np.deg2rad(2)).reshape(-1,3)).apply(w)
    return p.reshape(-1,3),v.reshape(-1,3),w,energy


def evaluate(rows,angles,cd=.35,lift_scale=1.,dt=.002,noise=None):
    linear=noise is None
    if linear:
        noise=np.zeros((21,10))
        for i in range(10):noise[1+2*i,i]=1;noise[2+2*i,i]=-1
    p,v,w,energies=states(rows,angles,noise);n,s=len(rows),len(noise)
    result=flight_batch(p,v,w,cd=cd,lift_scale=lift_scale,dt=dt)
    end=result['state'].reshape(n,s,6);times=result['time_s'].reshape(n,s);valid=result['valid'].reshape(n,s)
    if not valid.all():raise ValueError('Excluded flight in this predeclared grid/stress; do not hide failures')
    endpoint=end[:,:,[0,2]];target=np.array([0,.9])
    if not linear:
        errors=np.linalg.norm(endpoint-target,axis=2)*100
        return dict(rms_target_cm=np.sqrt(np.mean(errors**2,axis=1)),p90_target_cm=np.percentile(errors,90,axis=1),
                    endpoints=endpoint,failed=np.sum(~valid,axis=1),energy_sample_j=energies)
    jac=np.stack([(endpoint[:,1+2*i]-endpoint[:,2+2*i])/2 for i in range(10)],axis=-1)
    variance=np.sum(jac**2,axis=(1,2));bias=np.sum((endpoint[:,0]-target)**2,axis=1)
    pn,vn,wn,_=states(rows,angles,np.zeros((1,10)))
    no_lift=flight_batch(pn,vn,wn,cd=cd,lift_scale=0,dt=dt)
    if not no_lift['valid'].all():raise ValueError('Counterfactual ground impact; movement not defined')
    movement=(end[:,0][:,[0,2]]-no_lift['state'][:,[0,2]])*100
    return dict(speed_kmh=np.linalg.norm(vn,axis=1)*3.6,plate_speed_kmh=np.linalg.norm(end[:,0,3:],axis=1)*3.6,
                flight_ms=times[:,0]*1000,ivb_proxy_cm=movement[:,1],hb_proxy_cm=movement[:,0],
                rms_target_cm=np.sqrt(variance+bias)*100,nominal_target_error_cm=np.sqrt(bias)*100,
                vaa_deg=np.degrees(np.arctan2(end[:,0,5],np.linalg.norm(end[:,0,3:5],axis=1))))


def improvement(metrics,base):
    speed=metrics['speed_kmh']-base['speed_kmh'];shape=metrics['ivb_proxy_cm']-base['ivb_proxy_cm']
    precision=base['rms_target_cm']-metrics['rms_target_cm'];time=base['flight_ms']-metrics['flight_ms']
    return dict(speed_delta_kmh=speed,ivb_delta_cm=shape,rms_reduction_cm=precision,flight_reduction_ms=time,
                all_three_strict=bool(speed>1e-6 and shape>1e-6 and precision>1e-6),
                all_four_nonworse=bool(speed>=-1e-6 and shape>=-1e-6 and precision>=-1e-6 and time>=-1e-6))


def pareto_mask(objectives):
    """All columns maximized; finite-grid dominance, not a global optimum."""
    x=np.asarray(objectives,float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite objective matrix required')
    dominates=np.all(x[None,:,:]>=x[:,None,:]-1e-12,axis=2)&np.any(x[None,:,:]>x[:,None,:]+1e-12,axis=2)
    return ~dominates.any(axis=1)


def choose_speed_then_movement(table):
    """Ignore roundoff-only speed differences before secondary shape ranking."""
    tied=table[table.speed_kmh>=table.speed_kmh.max()-1e-8]
    return int(tied.sort_values(['ivb_proxy_cm','rms_target_cm'],ascending=[False,True]).index[0])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=pd.DataFrame([dict(rpm=r,efficiency=e,tilt_deg=t,height_m=h,forward_m=f)
                       for r,e,t,h,f in product((1600,1800,2000,2200,2400,2600),(.6,.75,.9,1.),(-10,0,10),(1.78,1.8,1.82),(0,.05,.10))])
    base_id=int(np.flatnonzero((rows.rpm==2400)&(rows.efficiency==.75)&(rows.tilt_deg==0)&(rows.height_m==1.8)&(rows.forward_m==0))[0])
    p=np.c_[np.zeros(len(rows)),rows.forward_m,rows.height_m]
    speed=energy_speed(E0,rows.rpm,rows.height_m)
    angles=aim(p,speed,rows.rpm,rows.efficiency,rows.tilt_deg)
    nominal=evaluate(rows,angles);table=rows.copy()
    for k,v in nominal.items():table[k]=v
    table['azimuth_deg']=angles[:,0];table['elevation_deg']=angles[:,1]
    base={k:float(v[base_id]) for k,v in nominal.items()}
    changes=[improvement({k:float(v[i]) for k,v in nominal.items()},base) for i in range(len(rows))]
    for k in changes[0]:table[k]=[x[k] for x in changes]
    table['aim_change_deg']=np.linalg.norm(angles-angles[base_id],axis=1)
    table['fixed_release_position']=np.isclose(table.height_m,1.8)&np.isclose(table.forward_m,0)
    table['release_energy_j']=E0
    feasible=table[table.aim_change_deg<=1]
    table['pareto_grid_member']=False
    table.loc[feasible.index,'pareto_grid_member']=pareto_mask(np.c_[feasible.speed_kmh,feasible.ivb_proxy_cm,-feasible.rms_target_cm,-feasible.flight_ms])
    strict=feasible[feasible.all_three_strict & feasible.all_four_nonworse]
    fixed=strict[strict.fixed_release_position]
    selected=dict(baseline=base_id,speed_focus=choose_speed_then_movement(feasible),
                  movement_focus=int(feasible.ivb_proxy_cm.idxmax()),precision_focus=int(feasible.rms_target_cm.idxmin()))
    if len(strict):selected['joint_release_candidate']=choose_speed_then_movement(strict)
    if len(fixed):selected['fixed_position_joint_candidate']=choose_speed_then_movement(fixed)
    # Search all nominally joint-improving rows, not only the most attractive one.
    robust_counts={int(i):0 for i in strict.index}
    if len(strict):
        audit_ids=[base_id,*[int(i) for i in strict.index if i!=base_id]]
        audit_rows=rows.loc[audit_ids].reset_index(drop=True)
        for cd,scale in product((.30,.35,.40),(.9,1.,1.1)):
            audit=evaluate(audit_rows,angles[audit_ids],cd,scale)
            audit_base={k:float(v[0]) for k,v in audit.items()}
            for j,idx in enumerate(audit_ids[1:],1):
                robust_counts[idx]+=int(improvement({k:float(v[j]) for k,v in audit.items()},audit_base)['all_four_nonworse'])
        robust=[i for i,count in robust_counts.items() if count==9]
        if robust:selected['robust_joint_release_candidate']=choose_speed_then_movement(table.loc[robust])
    indices=list(dict.fromkeys(selected.values()));small=rows.loc[indices].reset_index(drop=True);smallangles=angles[indices]
    mc=evaluate(small,smallangles,noise=np.random.default_rng(20261029).normal(size=(256,10)))
    mc2=evaluate(small,smallangles,noise=np.random.default_rng(20261030).normal(size=(256,10)))
    sensitivity=[]
    for cd,scale in product((.30,.35,.40),(.9,1.,1.1)):
        values=evaluate(small,smallangles,cd,scale)
        baseline={k:float(v[0]) for k,v in values.items()}
        for i,idx in enumerate(indices):
            entry=dict(index=idx,cd=cd,lift_scale=scale,**improvement({k:float(v[i]) for k,v in values.items()},baseline))
            sensitivity.append(entry)
    fine=evaluate(small,smallangles,dt=.001);finer=evaluate(small,smallangles,dt=.0005)
    records=[]
    for label,idx in selected.items():
        i=indices.index(idx);row=table.loc[idx].to_dict()
        row.update(label=label,index=idx,mc_rms_target_cm=float(mc['rms_target_cm'][i]),mc_p90_target_cm=float(mc['p90_target_cm'][i]),
                   mc_second_seed_rms_cm=float(mc2['rms_target_cm'][i]),
                   numerical_rms_difference_cm=float(abs(fine['rms_target_cm'][i]-finer['rms_target_cm'][i])),
                   numerical_ivb_difference_cm=float(abs(fine['ivb_proxy_cm'][i]-finer['ivb_proxy_cm'][i])),
                   sensitivity_all_four_nonworse=int(sum(r['all_four_nonworse'] for r in sensitivity if r['index']==idx)),
                   athlete_force_work_evaluated=False,athlete_candidate_validated=False)
        records.append(row)
    horizontal=pd.DataFrame([dict(rpm=2400,efficiency=1.,tilt_deg=90,height_m=1.8,forward_m=0.)])
    hp=np.array([[0.,0.,1.8]]);hs=energy_speed(E0,horizontal.rpm,horizontal.height_m)
    ha=aim(hp,hs,horizontal.rpm,horizontal.efficiency,horizontal.tilt_deg)
    hm=evaluate(horizontal,ha)
    horizontal_example=dict(**horizontal.iloc[0].to_dict(),**{k:float(v[0]) for k,v in hm.items()},
                            azimuth_deg=float(ha[0,0]),elevation_deg=float(ha[0,1]),
                            scope='Alternative horizontal-movement shape, not better fastball or verified slider/sinker')
    np.savez_compressed(OUT/'plot_inputs.npz',indices=indices,angles=smallangles,mc_endpoints=mc['endpoints'],
                        **{f'{k}_nominal':v for k,v in nominal.items()})
    p,v,w,_=states(small,smallangles,np.zeros((1,10)));paths=flight_batch(p,v,w,dt=.001,save_path=True)
    np.savez_compressed(OUT/'flight_paths.npz',paths=paths['path'],flight_times_s=paths['time_s'],indices=indices)
    table.to_csv(OUT/'all_cases.csv',index=False)
    report=dict(scope='Synthetic release-level fastball scenario, not measured athlete or joint/muscle intervention prediction',
                baseline_scenario=dict(speed_kmh=130,rpm=2400,transverse_fraction=.75,height_m=1.8,release_to_plate_m=16.8,target_xz_m=[0,.9]),
                fixed_release_mechanical_energy_j=E0,ball_mass_kg=MASS,ball_radius_m=RADIUS,inertia_kg_m2=INERTIA,
                grid_cases=len(rows),nominal_strict_joint_cases=len(strict),fixed_position_strict_joint_cases=len(fixed),
                pareto_grid_cases=int(table.pareto_grid_member.sum()),
                robust_joint_cases=sum(n==9 for n in robust_counts.values()),
                all_nominal_joint_sensitivity_pass_counts=robust_counts,
                selection='finite grid, nominal linearized RMS, <=1deg aim change; selected joint row maximizes speed then vertical movement among joint-pass rows',
                command_scope='re-aimed nominal means; same synthetic stress, not learned command; no perturbation variance improvement imposed',
                noise_scope='common energy1J, rpm50, az/el.15deg, axis xyz2deg, position xyz5mm; not athlete distribution or CI',
                aero_scope='bilinear Cl transverse-spin extension; constant Cd; no seams/wind/spin decay',
                sensitivity_scope='9 engineering air-coefficient cases, frozen nominal aiming angles',
                tested_metrics_are_game_performance=False,body_to_release_model_validated=False,
                release_speed_is_energy_allocation_assumption=True,induced_movement_is_proxy=True,
                nominal_aim_is_retargeted=True,perturbations_are_assumed=True,
                same_athlete_force_work_verified=False,full_hypothesis_confirmed=False,records=records,sensitivity=sensitivity)
    report['horizontal_shape_example']=horizontal_example
    report['zero_spin_energy_conversion_upper_bound_gain_kmh']=float(energy_speed(E0,0,1.8)-130)
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('grid',len(rows),'strict joint',len(strict),'fixed position joint',len(fixed),'E',E0)
    for r in records:print(r['label'],{k:round(r[k],5) for k in ['speed_kmh','ivb_proxy_cm','rms_target_cm','mc_rms_target_cm','flight_ms','height_m','forward_m']},'stress_pass',r['sensitivity_all_four_nonworse'])
    return report


if __name__=='__main__':main()
