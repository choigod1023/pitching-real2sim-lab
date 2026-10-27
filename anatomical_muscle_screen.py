"""Constrained generalized-force screen using source muscle moment arms.

Source Fmax and source eccentric active-force envelopes are diagnostics.
Millard equilibrium endpoints are evaluated separately, but no continuous
tendon/activation dynamics, ball payload or glenohumeral stability gate is solved.
"""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import opensim as o
from scipy.linalg import null_space
from scipy.optimize import lsq_linear
from anatomical_shoulder_fit import OUT,arr

def ov(x):
    v=o.Vector(len(x),0.)
    for i,a in enumerate(x):v.set(i,float(a))
    return v

def equilibrium_tendon_forces(model,state,muscles,activation):
    """OpenSim equilibrateMuscles skips disabled muscles: enable temporarily.

    Inverse dynamics intentionally disables forces, but equilibrium must not
    inherit that mode. Restore flags even on solver exceptions.
    """
    flags=[mus.appliesForce(state) for mus in muscles]
    try:
        for mus in muscles:
            mus.setAppliesForce(state,True);mus.setActivation(state,activation)
        model.equilibrateMuscles(state);model.realizeDynamics(state)
        return np.array([mus.getTendonForce(state) for mus in muscles]),np.array([mus.getNormalizedFiberLength(state) for mus in muscles])
    finally:
        for mus,flag in zip(muscles,flags):mus.setAppliesForce(state,flag)

def main(output_dir=None):
    global OUT
    if output_dir is not None:OUT=Path(output_dir)
    o.Logger.setLevelString('error');m=o.Model(str(OUT/'scaled_source.osim'));s=m.initSystem();matter=m.getMatterSubsystem();cs=m.getCoordinateSet()
    source_caps=[float(x.text) for x in ET.parse(OUT/'scaled_source.osim').findall('.//max_eccentric_velocity_force_multiplier')]
    if len(source_caps)!=33 or not np.allclose(source_caps,1.4):raise ValueError('Unexpected source active force envelope')
    muscles=[m.getMuscles().get(i) for i in range(m.getMuscles().getSize())]
    for i in range(m.getForceSet().getSize()):m.getForceSet().get(i).setAppliesForce(s,False)
    solver=o.InverseDynamicsSolver(m);nu=s.getNU();reports=[]
    def set_coordinates(x):
        for j,v in enumerate(x):cs.get(j).setValue(s,float(v),False)
        m.realizePosition(s)
    # Identify prescribed ground/thorax mobility directions without assuming
    # coordinate qdot == u; scapulothoracic mobilizer has quaternion coordinates.
    root=[]
    for j in range(6):
        s.setU(ov(np.zeros(nu)));cs.get(j).setSpeedValue(s,1.);root.append(arr(s.getU()))
    root=np.array(root)
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        z=np.load(OUT/f'{pitch}_ik.npz');time=z['time'];Q=[]
        for x in z['q']:set_coordinates(x);Q.append(arr(s.getQ()))
        Q=np.array(Q);qdot=np.gradient(Q,time,axis=0,edge_order=2);U=[]
        for i,x in enumerate(z['q']):
            set_coordinates(x);u=o.Vector(nu,0.);matter.multiplyByNInv(s,False,ov(qdot[i]),u);U.append(arr(u))
        U=np.array(U);udot=np.gradient(U,time,axis=0,edge_order=2);rows=[]
        for i in range(2,len(time)-2):
            set_coordinates(z['q'][i]);s.setU(ov(U[i]));m.realizeVelocity(s)
            tau=arr(solver.solve(s,ov(udot[i])))
            G=[];R=[]
            for k in range(nu):
                basis=np.eye(nu)[k];g=o.Vector();matter.multiplyByG(s,ov(basis),g);G.append(arr(g))
            S=null_space(np.vstack([np.array(G).T,root]))
            for k in range(nu):
                s.setU(ov(np.eye(nu)[k]));m.realizeVelocity(s)
                R.append([-mus.getGeometryPath().getLengtheningSpeed(s) for mus in muscles])
            R=np.array(R);s.setU(ov(U[i]));m.realizeVelocity(s)
            target=S.T@tau;fmax=np.array([mus.getMaxIsometricForce() for mus in muscles]);envelopes={}
            for cap in [1.,1.4]:
                # Only original Millard muscles have the verified 1.4 envelope.
                # Imported Schutte wrist muscles retain a 1.0 Fmax diagnostic;
                # their source equilibrium curves are evaluated below.
                caps=np.array([cap if mus.getConcreteClassName()=='Millard2012EquilibriumMuscle' else 1. for mus in muscles])
                A=(S.T@R)*(fmax*caps)[None,:]
                fit=lsq_linear(A,target,bounds=(0,1),tol=1e-8,max_iter=200)
                residual=A@fit.x-target
                envelopes[str(cap)]=dict(residual_norm_nm=float(np.linalg.norm(residual)),
                    demand_norm_nm=float(np.linalg.norm(target)),max_fraction=float(fit.x.max()),
                    muscle_tendon_positive_power_w=float(np.maximum((R.T@U[i])*(fit.x*fmax*caps),0).sum()))
            # Actual Millard equilibrium tendon-force endpoints, with current
            # model lengths and velocities. Still an instantaneous relaxation:
            # no shared tendon/activation state trajectory is integrated.
            hill={}
            try:
                ends=[]
                for activation in [0.,1.]:
                    forces,fiber_lengths=equilibrium_tendon_forces(m,s,muscles,activation)
                    ends.append(forces)
                low=np.minimum(*ends);high=np.maximum(*ends)
                if not np.isfinite(low).all() or not np.isfinite(high).all():raise ValueError('Nonfinite equilibrium force')
                A=(S.T@R)*(high-low)[None,:];offset=S.T@R@low
                fit=lsq_linear(A,target-offset,bounds=(0,1),tol=1e-8,max_iter=200)
                hill=dict(status='success',residual_norm_nm=float(np.linalg.norm(A@fit.x+offset-target)),
                    endpoint_force_max_n=float(high.max()),largest_force_muscle=muscles[int(high.argmax())].getName(),
                    wrist_endpoint_force_max_n=float(max((high[j] for j,mus in enumerate(muscles) if mus.getName() in ['ECRL','ECRB','ECU','FCR','FCU','PL']),default=0.)),
                    normalized_fiber_length_max=float(fiber_lengths.max()))
            except Exception as exc:hill=dict(status='failed',error=str(exc)[:300])
            rows.append(dict(sample=i,free_internal_dofs=S.shape[1],envelopes=envelopes,hill_equilibrium_interval=hill))
        result=dict(pitch=pitch,frames=len(rows),
            projected_demand_rms_nm=float(np.sqrt(np.mean([r['envelopes']['1.0']['demand_norm_nm']**2 for r in rows]))),
            hill_success_frames=sum(r['hill_equilibrium_interval']['status']=='success' for r in rows),
            hill_residual_rms_nm=float(np.sqrt(np.mean([r['hill_equilibrium_interval']['residual_norm_nm']**2 for r in rows if r['hill_equilibrium_interval']['status']=='success']))) if any(r['hill_equilibrium_interval']['status']=='success' for r in rows) else None,
            envelopes={str(cap):dict(residual_rms_nm=float(np.sqrt(np.mean([r['envelopes'][str(cap)]['residual_norm_nm']**2 for r in rows]))),
                frames_residual_over_1nm=sum(r['envelopes'][str(cap)]['residual_norm_nm']>1 for r in rows)) for cap in [1.,1.4]},rows=rows)
        reports.append(result);print({k:v for k,v in result.items() if k!='rows'},flush=True)
    (OUT/'muscle_screen.json').write_text(json.dumps(dict(reports=reports,scope=__doc__),indent=2))

if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else None)
