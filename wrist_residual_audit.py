"""Paired structural and derivative diagnostics. No radar fitting or strength tuning."""
import json,copy
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import opensim as o
from scipy.linalg import null_space
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear
from anatomical_muscle_screen import ov,equilibrium_tendon_forces
from anatomical_shoulder_fit import arr,model_points
from applied_path_forces import applied_path_force_matrix
from directional_shoulder_fit import orientations
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'artifacts/wrist-muscle'
OUT=ROOT/'artifacts/wrist-residual-audit'

def project(v,G,rhs=None):
    """Minimum norm correction satisfying G v = rhs."""
    rhs=np.zeros(G.shape[0]) if rhs is None else rhs
    return v-G.T@np.linalg.lstsq(G@G.T,G@v-rhs,rcond=None)[0]

def run(model_file,pitch,mode,force_mapping='applied_force'):
    m=o.Model(str(model_file));s=m.initSystem();matter=m.getMatterSubsystem();cs=m.getCoordinateSet();nu=s.getNU()
    muscles=[m.getMuscles().get(i) for i in range(m.getMuscles().getSize())]
    for i in range(m.getForceSet().getSize()):m.getForceSet().get(i).setAppliesForce(s,False)
    solver=o.InverseDynamicsSolver(m);z=np.load(INPUT/f'{pitch}_ik.npz');t=z['time'];q=z['q'].copy();nc=q.shape[1]
    names=list(z['coordinate_names']);bounds=np.array([[cs.get(j).getRangeMin(),cs.get(j).getRangeMax()] for j in range(nc)])
    if mode.startswith('smooth'):
        window=int(mode[6:]);q=savgol_filter(q,window,3,axis=0)
        q=np.clip(q,bounds[:,0],bounds[:,1])
        for conname in ['wrist_hand_r1_con','wrist_hand_r3_con']:
            con=o.CoordinateCouplerConstraint.safeDownCast(m.getConstraintSet().get(conname))
            src=names.index(con.getIndependentCoordinateNames().get(0));dst=names.index(con.getDependentCoordinateName())
            q[:,dst]=[con.getFunction().calcValue(ov([v])) for v in q[:,src]]
    def put(x):
        for j,v in enumerate(x):cs.get(j).setValue(s,float(v),False)
        m.realizePosition(s)
    root=[]
    for j in range(6):
        s.setU(ov(np.zeros(nu)));cs.get(j).setSpeedValue(s,1);root.append(arr(s.getU()))
    Q=[];Gs=[];positions=[];angles=[];closures=[]
    for i,x in enumerate(q):
        put(x);Q.append(arr(s.getQ()));positions.append(model_points(m,s));closures.append(np.max(abs(arr(s.getQErr()))))
        angles.append(Rotation.from_matrix(z['orientation_targets'][i,2].T@orientations(m,s)[2]).magnitude())
        cols=[]
        for k in range(nu):
            g=o.Vector();matter.multiplyByG(s,ov(np.eye(nu)[k]),g);cols.append(arr(g))
        Gs.append(np.array(cols).T)
    Gs=np.array(Gs);qd=np.gradient(Q,t,axis=0,edge_order=2);U=[]
    for i,x in enumerate(q):
        put(x);u=o.Vector(nu,0);matter.multiplyByNInv(s,False,ov(qd[i]),u);U.append(arr(u))
    U=np.array(U);rawU=U.copy()
    if mode!='original':U=np.array([project(u,g) for u,g in zip(U,Gs)])
    ud=np.gradient(U,t,axis=0,edge_order=2);gd=np.gradient(Gs,t,axis=0,edge_order=2);rawud=ud.copy()
    if mode!='original':ud=np.array([project(a,g,-dg@u) for a,g,dg,u in zip(ud,Gs,gd,U)])
    rows=[]
    for i in range(2,len(t)-2):
        put(q[i]);s.setU(ov(U[i]));m.realizeVelocity(s);tau=arr(solver.solve(s,ov(ud[i])))
        S=null_space(np.vstack([Gs[i],root]));R=[]
        for k in range(nu):
            s.setU(ov(np.eye(nu)[k]));m.realizeVelocity(s);R.append([-mus.getPath().getLengtheningSpeed(s) for mus in muscles])
        R=np.array(R);s.setU(ov(U[i]));m.realizeVelocity(s)
        if force_mapping=='applied_force':R=applied_path_force_matrix(m,s,muscles)
        elif force_mapping!='legacy_speed':raise ValueError('Unknown force mapping')
        ends=[equilibrium_tendon_forces(m,s,muscles,a)[0] for a in [0.,1.]];low=np.minimum(*ends);high=np.maximum(*ends)
        A=(S.T@R)*(high-low)[None,:];target=S.T@tau;offset=S.T@R@low
        fit=lsq_linear(A,target-offset,bounds=(0,1),tol=1e-8,max_iter=200)
        res=S@(A@fit.x+offset-target)
        relaxed=(S.T@R)*high[None,:]
        no_passive=lsq_linear(relaxed,target,bounds=(0,1),tol=1e-8,max_iter=200)
        # Fixed-wrist virtual motions inside SAME model/basis: diagnostic only.
        wrist=[]
        for name in ['deviation','flexion','wrist_hand_r1','wrist_hand_r3']:
            s.setU(ov(np.zeros(nu)));cs.get(names.index(name)).setSpeedValue(s,1);wrist.append(arr(s.getU()))
        C=null_space(np.vstack([Gs[i],root,wrist]));common=C.T@res
        commonA=(C.T@R)*(high-low)[None,:];commontarget=C.T@(tau-R@low)
        cf=lsq_linear(commonA,commontarget,bounds=(0,1),tol=1e-8,max_iter=200)
        # Gravity/velocity demand with acceleration zero, same muscles and U.
        s.setU(ov(U[i]));m.realizeVelocity(s);tau0=arr(solver.solve(s,ov(np.zeros(nu))))
        f0=lsq_linear(A,S.T@tau0-offset,bounds=(0,1),tol=1e-8,max_iter=200)
        rows.append(dict(sample=i,residual=float(np.linalg.norm(res)),common_component=float(np.linalg.norm(common)),wrist_complement=float(np.sqrt(max(0,res@res-common@common))),common_only_fit=float(np.linalg.norm(commonA@cf.x-commontarget)),zero_accel_diagnostic=float(np.linalg.norm(A@f0.x+offset-S.T@tau0)),acceleration_demand=float(np.linalg.norm(S.T@(tau-tau0))),max_endpoint_force=float(high.max()),u_constraint=float(np.linalg.norm(Gs[i]@U[i])),a_constraint=float(np.linalg.norm(Gs[i]@ud[i]+gd[i]@U[i]))))
        rows[-1]['zero_passive_bound_diagnostic']=float(np.linalg.norm(relaxed@no_passive.x-target))
    keys=['residual','common_component','wrist_complement','common_only_fit','zero_accel_diagnostic','acceleration_demand','u_constraint','a_constraint','zero_passive_bound_diagnostic']
    report=dict(pitch=pitch,mode=mode,force_mapping=force_mapping,model=Path(model_file).stem,frames=len(rows),rms={k:float(np.sqrt(np.mean([r[k]**2 for r in rows]))) for k in keys},peak_force=max(r['max_endpoint_force'] for r in rows),arm_rmse_mm=float(np.sqrt(np.mean(np.sum((np.array(positions)[:,3:]-z['target'][:,3:])**2,axis=2)))*1000),hand_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.array(angles)**2)))),constraint_position_max=float(max(closures)),rows=rows)
    np.savez_compressed(OUT/f'{Path(model_file).stem}_{pitch}_{mode}.npz',q=q,U=U,udot=ud,time=t,predicted=positions)
    print({k:v for k,v in report.items() if k!='rows'},flush=True);return report

def main():
    OUT.mkdir(parents=True,exist_ok=True);o.Logger.setLevelString('error')
    # Structural ablation: collapse source wrist-hand translation only. This
    # is NOT an anatomically validated correction or a candidate for adoption.
    tree=E.parse(INPUT/'scaled_source.osim');j=tree.find(".//JointSet/objects/*[@name='wrist_hand']")
    j.find('frames/PhysicalOffsetFrame/translation').text='0 0 0';tree.write(OUT/'collapsed_offset.osim',encoding='utf-8',xml_declaration=True)
    rows=[]
    for p in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        for model,mode in [(INPUT/'scaled_source.osim','original'),(INPUT/'scaled_source.osim','consistent'),(INPUT/'scaled_source.osim','smooth7'),(INPUT/'scaled_source.osim','smooth11'),(OUT/'collapsed_offset.osim','original'),(OUT/'collapsed_offset.osim','smooth7')]:
            rows.append(run(model,p,mode));(OUT/'results.json').write_text(json.dumps(dict(rows=rows),indent=2))

if __name__=='__main__':main()
