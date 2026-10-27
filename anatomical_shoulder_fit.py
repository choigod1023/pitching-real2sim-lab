"""Execute the published OpenSim shoulder model, retaining source constraints.

Registration/IK feasibility before muscle or counterfactual claims. Source model
is a generic shoulder, not a calibrated pitcher; no range or Fmax increases.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import opensim as osim
from scipy.optimize import least_squares
OUT=Path(__file__).resolve().parent/'artifacts/anatomical-shoulder'

def arr(v):return np.array([v.get(i) for i in range(v.size())])
def vec3(v):return np.array([v.get(i) for i in range(3)])

def model_points(m,s):
    return np.array([vec3(m.getMarkerSet().get(n).getLocationInGround(s)) for n in ['ij','c7','px']]+[vec3(m.getJointSet().get(n).getChildFrame().getPositionInGround(s)) for n in ['GlenoHumeral','elbow','rc']])

def register(source,target):
    a=source-source.mean(0);b=target-target.mean(0);u,_,vt=np.linalg.svd(a.T@b)
    sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vt);rot=u@sign@vt
    return rot,target.mean(0)-source.mean(0)@rot

def setup():
    osim.Logger.setLevelString('error')
    path=OUT/'TSM_subject_noWeight.osim';m=osim.Model(str(path));s=m.initSystem();m.realizePosition(s)
    initial=model_points(m,s);train=np.load(OUT/'3034_1_input.npz')
    upper=float(np.median(np.linalg.norm(train['arm'][:,1]-train['arm'][:,0],axis=1))/np.linalg.norm(initial[4]-initial[3]))
    fore=float(np.median(np.linalg.norm(train['arm'][:,2]-train['arm'][:,1],axis=1))/np.linalg.norm(initial[5]-initial[4]))
    torso=float(np.median(np.linalg.norm(train['torso'][:,0]-train['torso'][:,2],axis=1))/np.linalg.norm(initial[0]-initial[2]))
    ss=osim.ScaleSet()
    for i in range(m.getBodySet().getSize()):
        name=m.getBodySet().get(i).getName();f=upper if name=='humerus' else fore if name in ['ulna','radius','hand'] else torso
        sc=osim.Scale();sc.setSegmentName(name);sc.setScaleFactors(osim.Vec3(f));sc.setApply(True);ss.cloneAndAppend(sc)
    force0=[m.getMuscles().get(i).getMaxIsometricForce() for i in range(m.getMuscles().getSize())]
    m.scale(s,ss,False);s=m.initSystem();m.realizePosition(s)
    force1=[m.getMuscles().get(i).getMaxIsometricForce() for i in range(m.getMuscles().getSize())]
    if not np.array_equal(force0,force1):raise ValueError('Scaling changed Fmax')
    source=model_points(m,s);r,tr=register(train['torso'][0],source[:3])
    m.printToXML(str(OUT/'scaled_source.osim'))
    manifest=dict(source_url='https://github.com/ComputationalBiomechanicsLab/rmr-solver',
        source_commit='d40ebd8ba658633993e531e408c0d96df30ff367',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        runtime=osim.GetVersionAndDate(),muscles=len(force0),constraints=m.getConstraintSet().getSize(),
        upper_scale=upper,forearm_scale=fore,torso_scale=torso,fmax_unchanged=True,
        marker_correspondence={'CLAV':'ij','C7':'c7','STRN':'px'},
        limitation='Marker correspondences assume compatible placements and are not independently verified. Source bounds are model bounds, not athlete-specific ROM. Wrist is welded in source, no ball/contact yet.',
        coordinates=[dict(name=m.getCoordinateSet().get(i).getName(),range=[m.getCoordinateSet().get(i).getRangeMin(),m.getCoordinateSet().get(i).getRangeMax()]) for i in range(m.getNumCoordinates())])
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2));return m,s,r,tr

def fit(pitches,temporal_weight=10.):
    m,s,rotation,translation=setup();cs=m.getCoordinateSet();nc=cs.getSize()
    coords=[cs.get(i) for i in range(nc)];lower=np.array([c.getRangeMin() for c in coords]);upper=np.array([c.getRangeMax() for c in coords]);default=np.array([c.getValue(s) for c in coords]);reports=[]
    for pitch in pitches:
        data=np.load(OUT/f'{pitch}_input.npz');target=np.concatenate([data['torso'],data['arm']],axis=1)@rotation+translation
        q=default.copy();qs=[];preds=[];closures=[];at_bounds=[];nfevs=[]
        for index,truth in enumerate(target):
            prior=q.copy() if len(qs)<2 else 2*qs[-1]-qs[-2]
            def residual(x):
                for c,v in zip(coords,x):c.setValue(s,float(v),False)
                m.realizePosition(s)
                return np.r_[((model_points(m,s)-truth)/.005).ravel(),arr(s.getQErr())/.0001,(x-prior)*temporal_weight]
            result=least_squares(residual,np.clip(q,lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=70,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            q=result.x;residual(q);qs.append(q.copy());preds.append(model_points(m,s));closures.append(float(np.linalg.norm(arr(s.getQErr()))));nfevs.append(result.nfev)
            at_bounds.append(((q-lower<1e-4)|(upper-q<1e-4)).tolist())
        preds=np.array(preds);errors=np.linalg.norm(preds-target,axis=2)*1000
        row=dict(pitch=pitch,arm_rmse_mm=float(np.sqrt(np.mean(errors[:,3:]**2))),torso_rmse_mm=float(np.sqrt(np.mean(errors[:,:3]**2))),
            shoulder_rmse_mm=float(np.sqrt(np.mean(errors[:,3]**2))),wrist_rmse_mm=float(np.sqrt(np.mean(errors[:,5]**2))),
            ac_closure_max_mm=max(closures)*1000,frames=len(qs),solver_max_evaluations=max(nfevs),
            bound_hit_frames={c.getName():int(np.array(at_bounds)[:,i].sum()) for i,c in enumerate(coords) if np.any(np.array(at_bounds)[:,i])},
            wrist_speed_at_br_kmh=float(np.linalg.norm(np.gradient(preds[:,5],data['time'],axis=0,edge_order=2)[-1])*3.6),
            ball_speed_predicted=False,temporal_weight=temporal_weight,
            maximum_coordinate_step_rad_or_m=float(np.max(abs(np.diff(np.array(qs),axis=0)))))
        np.savez_compressed(OUT/f'{pitch}_ik.npz',q=qs,time=data['time'],target=target,predicted=preds,ac_closure_m=closures,coordinate_names=[c.getName() for c in coords])
        reports.append(row);print(json.dumps(row),flush=True)
    (OUT/'ik_report.json').write_text(json.dumps(dict(rows=reports,scope='Published constrained shoulder IK; wrist proxy is not ball speed. No athlete capacity inference or counterfactual optimization.'),indent=2))

if __name__=='__main__':
    import sys
    fit(sys.argv[1:] or ['3034_1','3034_2','3034_3','3034_4','3034_5'])
