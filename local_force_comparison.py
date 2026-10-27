"""One-state diagnostic; never extrapolate the local path to a full pitch."""
import json
from pathlib import Path
import numpy as np
import opensim as o
from scipy.linalg import null_space
from scipy.optimize import lsq_linear
from anatomical_shoulder_fit import arr
from anatomical_muscle_screen import ov,equilibrium_tendon_forces
from scholz_wrapping_trial import ROOT,OUT,SOURCE
from applied_path_forces import unit_path_force,applied_path_force_matrix

def require_domain(q,low,high):
    q,low,high=map(lambda x:np.asarray(x,float),(q,low,high))
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in [q,low,high]) or np.any(high<=low):raise ValueError('Finite three-coordinate local box required')
    if np.any(q<low-1e-12) or np.any(q>high+1e-12):raise ValueError('Outside validated local path box; full-pitch use forbidden')

def evaluate(path):
    m=o.Model(str(path));s=m.initSystem();cs=m.getCoordinateSet();mat=m.getMatterSubsystem();nu=s.getNU();z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');t=z['time'];sample=28;Q=[]
    def put(q):
        for j,v in enumerate(q):cs.get(j).setValue(s,float(v),False)
        m.realizePosition(s)
    for q in z['q']:put(q);Q.append(arr(s.getQ()))
    qdot=np.gradient(Q,t,axis=0,edge_order=2);U=[]
    for i,q in enumerate(z['q']):
        put(q);u=o.Vector(nu,0);mat.multiplyByNInv(s,False,ov(qdot[i]),u);U.append(arr(u))
    U=np.array(U);ud=np.gradient(U,t,axis=0,edge_order=2);put(z['q'][sample]);root=[]
    for j in range(6):s.setU(ov(np.zeros(nu)));cs.get(j).setSpeedValue(s,1.);root.append(arr(s.getU()))
    mus=[m.getMuscles().get(i) for i in range(m.getMuscles().getSize())]
    for i in range(m.getForceSet().getSize()):m.getForceSet().get(i).setAppliesForce(s,False)
    s.setU(ov(U[sample]));m.realizeVelocity(s);tau=arr(o.InverseDynamicsSolver(m).solve(s,ov(ud[sample])));G=[];R=[]
    for k in range(nu):
        g=o.Vector();mat.multiplyByG(s,ov(np.eye(nu)[k]),g);G.append(arr(g));s.setU(ov(np.eye(nu)[k]));m.realizeVelocity(s);R.append([-x.getPath().getLengtheningSpeed(s) for x in mus])
    S=null_space(np.vstack([np.array(G).T,root]));R=np.array(R);s.setU(ov(U[sample]));m.realizeVelocity(s);R=applied_path_force_matrix(m,s,mus);ends=[equilibrium_tendon_forces(m,s,mus,a)[0] for a in [0.,1.]];low=np.minimum(*ends);high=np.maximum(*ends)
    A=(S.T@R)*(high-low)[None,:];offset=S.T@R@low;target=S.T@tau;fit=lsq_linear(A,target-offset,bounds=(0,1),max_iter=300,tol=1e-9)
    if not fit.success:raise ValueError('Force solve not converged')
    s.setU(ov(U[sample]));m.realizeVelocity(s)
    pathobj=m.getMuscles().get('Subscapularis_I').getPath();f=unit_path_force(m,s,pathobj);power=float(f@U[sample]);speed=float(pathobj.getLengtheningSpeed(s))
    return {'model':path.name,'pitch':'3034_4','sample':sample,'residual_nm':float(np.linalg.norm(A@fit.x+offset-target)),'subscapularis_unit_tension_power_w':power,'subscapularis_path_speed_m_s':speed,'unit_force_power_balance_error_w':abs(power+speed),'scope':'One already-seen state. Other wrapped muscles remain unvalidated; NOT pitch-level RMS or speed gain.'}

def main():
    o.Logger.setLevelString('error');metadata=json.loads((OUT/'local_0p002_length_based_checks.json').read_text());z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');names=list(z['coordinate_names']);q=z['q'][28,[names.index(n) for n in ['plane_elv','shoulder_elv','axial_rot']]]
    if not metadata['geometry_pass']:raise ValueError('Local source geometry check failed')
    require_domain(q,metadata['domain_low_rad'],metadata['domain_high_rad']);rows=[evaluate(path) for path in [SOURCE,OUT/'local_0p002_length_based_trial.osim']]
    (OUT/'local_force_comparison.json').write_text(json.dumps({'rows':rows,'full_pitch_eligible':False,'reason':'Local domain only; other muscles fail independent path checks.'},indent=2));print(rows)

if __name__=='__main__':main()
