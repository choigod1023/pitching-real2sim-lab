"""Separate torso cut loads, internal torque demands and muscle feasibility.

All relaxed-force cases are diagnostics, never claimed as athlete capability.
"""
import json
from pathlib import Path
import numpy as np
import opensim as o
from scipy.linalg import null_space
from scipy.optimize import lsq_linear,nnls
from anatomical_muscle_screen import ov,equilibrium_tendon_forces
from anatomical_shoulder_fit import arr
from applied_path_forces import applied_path_force_matrix
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/wrist-muscle'
MODEL=ROOT/'artifacts/elbow-moment-validation/matched/registered_plus_elbow.osim'
OUT=ROOT/'artifacts/shoulder-load-audit'
GROUPS={'girdle':['clav_prot','clav_elev','scapula_abduction','scapula_elevation','scapula_upward_rot','scapula_winging'],
        'shoulder':['plane_elv','shoulder_elv','axial_rot'],
        'elbow_forearm':['elbow_flexion','pro_sup'],
        'wrist':['deviation','flexion','wrist_hand_r1','wrist_hand_r3']}

def solve_interval(R,target,low,high):
    A=R*(high-low)[None,:];offset=R@low
    f=lsq_linear(A,target-offset,bounds=(0,1),tol=1e-9,max_iter=300)
    if not f.success:raise RuntimeError(f'Bounded force solve did not converge: {f.message}')
    force=low+(high-low)*f.x
    return force,R@force-target

def frame_jacobian(model,state,body,nu):
    """Ground spatial velocity Jacobian, torque first and force second."""
    out=[]
    for k in range(nu):
        state.setU(ov(np.eye(nu)[k]));model.realizeVelocity(state)
        vel=body.getVelocityInGround(state)
        out.append(np.r_[arr(vel.get(0)),arr(vel.get(1))])
    return np.array(out).T

def run(pitch,force_mapping='applied_force'):
    m=o.Model(str(MODEL));s=m.initSystem();matter=m.getMatterSubsystem();cs=m.getCoordinateSet();nu=s.getNU();nc=cs.getSize();names=[cs.get(i).getName() for i in range(nc)]
    muscles=[m.getMuscles().get(i) for i in range(m.getMuscles().getSize())]
    for i in range(m.getForceSet().getSize()):m.getForceSet().get(i).setAppliesForce(s,False)
    solver=o.InverseDynamicsSolver(m);z=np.load(BASE/f'{pitch}_ik.npz');t=z['time']
    def put(x):
        for j,v in enumerate(x):cs.get(j).setValue(s,float(v),False)
        m.realizePosition(s)
    Q=[]
    for q in z['q']:put(q);Q.append(arr(s.getQ()))
    qd=np.gradient(Q,t,axis=0,edge_order=2);U=[]
    for i,q in enumerate(z['q']):
        put(q);v=o.Vector(nu,0);matter.multiplyByNInv(s,False,ov(qd[i]),v);U.append(arr(v))
    U=np.array(U);ud=np.gradient(U,t,axis=0,edge_order=2);rows=[]
    root=[]
    for j in range(6):
        s.setU(ov(np.zeros(nu)));cs.get(j).setSpeedValue(s,1);root.append(arr(s.getU()))
    root=np.array(root)
    for i in range(2,len(t)-2):
        put(z['q'][i]);s.setU(ov(U[i]));m.realizeVelocity(s);tau=arr(solver.solve(s,ov(ud[i])))
        velocityzeroacc=arr(solver.solve(s,ov(np.zeros(nu))))
        s.setU(ov(np.zeros(nu)));m.realizeVelocity(s);gravity=arr(solver.solve(s,ov(np.zeros(nu))))
        inertial=tau-velocityzeroacc;velocity=velocityzeroacc-gravity
        G=[];R=[];E=[]
        for k in range(nu):
            g=o.Vector();matter.multiplyByG(s,ov(np.eye(nu)[k]),g);G.append(arr(g))
            s.setU(ov(np.eye(nu)[k]));m.realizeVelocity(s)
            R.append([-mus.getPath().getLengtheningSpeed(s) for mus in muscles])
            E.append([cs.get(j).getSpeedValue(s) for j in range(nc)])
        G=np.array(G).T;R=np.array(R);E=np.array(E).T;S=null_space(np.vstack([G,root]));M=S.T@R;target=S.T@tau
        s.setU(ov(U[i]));m.realizeVelocity(s)
        if force_mapping=='applied_force':R=applied_path_force_matrix(m,s,muscles);M=S.T@R
        elif force_mapping!='legacy_speed':raise ValueError('Unknown force mapping')
        endpoints=[];lengths=[]
        for a in [0.,1.]:
            forces,l=equilibrium_tendon_forces(m,s,muscles,a);endpoints.append(forces);lengths.append(l)
        low=np.minimum(*endpoints);high=np.maximum(*endpoints)
        forces,res=solve_interval(M,target,low,high)
        _,nopassive=solve_interval(M,target,np.zeros(len(muscles)),high)
        fmax=np.array([mus.getMaxIsometricForce() for mus in muscles]);_,static=solve_interval(M,target,np.zeros(len(muscles)),fmax)
        positive,positive_residual=nnls(M,target,maxiter=10000)
        signed=np.linalg.lstsq(M,target,rcond=None)[0]
        actual_native=S@res;groups={};bases=[]
        for group,members in GROUPS.items():
            other=[j for j,n in enumerate(names) if j>=6 and n not in members]
            C=null_space(np.vstack([G,root,E[other]]));bases.append(C)
            groups[group]={'dofs':C.shape[1],'residual':float(np.linalg.norm(C.T@actual_native)),
                           'demand':float(np.linalg.norm(C.T@tau)),
                           'acceleration':float(np.linalg.norm(C.T@inertial)),
                           'velocity':float(np.linalg.norm(C.T@velocity)),
                           'gravity':float(np.linalg.norm(C.T@gravity))}
        B=frame_jacobian(m,s,m.getBodySet().get('thorax'),nu).T
        # Any torso wrench has no virtual work for internal motion with torso fixed.
        leak=float(np.linalg.norm(S.T@B))
        cut=np.linalg.lstsq(np.column_stack([B,G.T]),tau-R@forces,rcond=None)[0][:6]
        matrix=np.concatenate(bases,axis=1);orth=float(np.linalg.norm(matrix.T@matrix-np.eye(matrix.shape[1])))
        rows.append(dict(sample=i,residual=float(np.linalg.norm(res)),no_passive_bound=float(np.linalg.norm(nopassive)),
            static_fmax=float(np.linalg.norm(static)),positive_unbounded=float(positive_residual),
            signed_unbounded=float(np.linalg.norm(M@signed-target)),rank=int(np.linalg.matrix_rank(M)),internal_dofs=S.shape[1],
            groups=groups,group_orthogonality=orth,group_dofs=matrix.shape[1],torso_wrench_internal_leak=leak,
            torso_required_torque_nm=cut[:3].tolist(),torso_required_force_n=cut[3:].tolist(),
            torque_reconstruction_error=float(np.linalg.norm(tau-inertial-velocity-gravity)),
            max_normalized_fiber_length=float(np.max(lengths)),max_endpoint_force=float(high.max()),
            positive_unbounded_max_force=float(positive.max()),positive_unbounded_solver='nnls_active_set'))
    result={'pitch':pitch,'force_mapping':force_mapping,'rows':rows,'mean_scope':'RMS within pitch, then arithmetic mean across pitches'}
    metrics=['residual','no_passive_bound','static_fmax','positive_unbounded','signed_unbounded']
    result['rms']={k:float(np.sqrt(np.mean([r[k]**2 for r in rows]))) for k in metrics}
    result['groups']={g:{k:float(np.sqrt(np.mean([r['groups'][g][k]**2 for r in rows]))) for k in ['residual','demand','acceleration','velocity','gravity']} for g in GROUPS}
    print({k:v for k,v in result.items() if k!='rows'},flush=True);return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);o.Logger.setLevelString('error');reports=[]
    for p in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        reports.append(run(p));(OUT/'results.json').write_text(json.dumps({'reports':reports},indent=2))

if __name__=='__main__':main()
