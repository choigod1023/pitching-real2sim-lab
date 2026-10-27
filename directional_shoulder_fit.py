"""Add frozen-training skin-frame orientation targets to the source shoulder IK."""
import json,shutil
from pathlib import Path
import numpy as np
import opensim as o
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from anatomical_shoulder_fit import OUT as BASE,model_points,arr,register
OUT=BASE.parent/'shoulder-marker-stability'
BODIES=['humerus','radius','hand']

def orientations(m,s):
    result=[]
    for name in BODIES:
        mat=m.getBodySet().get(name).getTransformInGround(s).R().asMat33()
        result.append(np.array([[mat.get(i,j) for j in range(3)] for i in range(3)]))
    return np.array(result)

def batch_refine(m,s,coords,qs,target,targets,lo,hi,max_nfev=30):
    """Same data/weights; symmetric second-difference regularity across frames."""
    n,nc=np.shape(qs);m.realizePosition(s);nerr=len(arr(s.getQErr()));nr=18+nerr+9
    sparsity=lil_matrix((n*nr+(n-2)*nc,n*nc),dtype=int)
    for i in range(n):sparsity[i*nr:(i+1)*nr,i*nc:(i+1)*nc]=1
    for i in range(n-2):sparsity[n*nr+i*nc:n*nr+(i+1)*nc,i*nc:(i+3)*nc]=1
    def objective(flat):
        q=flat.reshape(n,nc);parts=[]
        for i,x in enumerate(q):
            for c,v in zip(coords,x):c.setValue(s,float(v),False)
            m.realizePosition(s)
            delta=Rotation.from_matrix(np.einsum('bji,bjk->bik',targets[i],orientations(m,s))).as_rotvec()
            parts.append(np.r_[((model_points(m,s)-target[i])/.005).ravel(),arr(s.getQErr())/.0001,(delta/np.deg2rad(10.)).ravel()])
        return np.r_[np.concatenate(parts),np.diff(q,n=2,axis=0).ravel()*10.]
    q0=np.clip(qs,lo+1e-8,hi-1e-8).ravel()
    fit=least_squares(objective,q0,bounds=(np.tile(lo,n),np.tile(hi,n)),jac_sparsity=sparsity.tocsr(),
        max_nfev=max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5,x_scale='jac',tr_solver='lsmr')
    return fit.x.reshape(n,nc),dict(nfev=fit.nfev,status=int(fit.status),cost=float(fit.cost),initial_cost=float(np.sum(objective(q0)**2)/2))

def main():
    o.Logger.setLevelString('error');m=o.Model(str(BASE/'scaled_source.osim'));s=m.initSystem();cs=m.getCoordinateSet();coords=[cs.get(i) for i in range(cs.getSize())]
    lo=np.array([c.getRangeMin() for c in coords]);hi=np.array([c.getRangeMax() for c in coords]);default=np.array([c.getValue(s) for c in coords]);m.realizePosition(s)
    train=np.load(BASE/'3034_1_input.npz');rotation,translation=register(train['torso'][0],model_points(m,s)[:3])
    def put(q):
        for c,x in zip(coords,q):c.setValue(s,float(x),False)
        m.realizePosition(s)
    trainq=np.load(BASE/'3034_1_ik.npz')['q'][0];put(trainq)
    trainframes=np.load(OUT/'3034_1_directions.npz')['frames'][0]
    registered=np.einsum('ij,bjk->bik',rotation.T,trainframes)
    offsets=np.einsum('bji,bjk->bik',registered,orientations(m,s))
    rows=[]
    shutil.copyfile(BASE/'scaled_source.osim',OUT/'scaled_source.osim')
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        data=np.load(BASE/f'{pitch}_input.npz');frames=np.load(OUT/f'{pitch}_directions.npz')['frames']
        target=np.concatenate([data['torso'],data['arm']],axis=1)@rotation+translation
        targets=np.einsum('ij,nbjk,bkl->nbil',rotation.T,frames,offsets)
        q=default.copy();qs=[];positions=[];angles=[];closures=[];bounds=[];iterations=[]
        for i,truth in enumerate(target):
            prior=q.copy() if len(qs)<2 else 2*qs[-1]-qs[-2]
            def residual(x):
                put(x);delta=Rotation.from_matrix(np.einsum('bji,bjk->bik',targets[i],orientations(m,s))).as_rotvec()
                return np.r_[((model_points(m,s)-truth)/.005).ravel(),arr(s.getQErr())/.0001,(x-prior)*10.,(delta/np.deg2rad(10.)).ravel()]
            fit=least_squares(residual,np.clip(q,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=70,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            q=fit.x;put(q);qs.append(q.copy());positions.append(model_points(m,s));iterations.append(fit.nfev)
            angles.append(Rotation.from_matrix(np.einsum('bji,bjk->bik',targets[i],orientations(m,s))).magnitude())
            closures.append(np.linalg.norm(arr(s.getQErr())));bounds.append((q-lo<1e-4)|(hi-q<1e-4))
        qs,batch_info=batch_refine(m,s,coords,qs,target,targets,lo,hi)
        positions=[];angles=[];closures=[];bounds=[]
        for i,x in enumerate(qs):
            put(x);positions.append(model_points(m,s));angles.append(Rotation.from_matrix(np.einsum('bji,bjk->bik',targets[i],orientations(m,s))).magnitude())
            closures.append(np.linalg.norm(arr(s.getQErr())));bounds.append((x-lo<1e-4)|(hi-x<1e-4))
        positions=np.array(positions);angles=np.array(angles);error=np.linalg.norm(positions-target,axis=2)*1000
        row=dict(pitch=pitch,arm_rmse_mm=float(np.sqrt(np.mean(error[:,3:]**2))),wrist_rmse_mm=float(np.sqrt(np.mean(error[:,5]**2))),
            torso_rmse_mm=float(np.sqrt(np.mean(error[:,:3]**2))),ac_closure_max_mm=float(max(closures)*1000),
            orientation_rmse_deg={n:float(np.rad2deg(np.sqrt(np.mean(angles[:,j]**2)))) for j,n in enumerate(BODIES)},
            max_coordinate_step_rad_or_m=float(np.max(abs(np.diff(qs,axis=0)))),
            bound_hit_frames={c.getName():int(np.array(bounds)[:,j].sum()) for j,c in enumerate(coords) if np.any(np.array(bounds)[:,j])},
            frames=len(qs),max_nfev=max(iterations),batch_optimizer=batch_info)
        # Compare old fit against EXACT same frozen orientation targets.
        old=np.load(BASE/f'{pitch}_ik.npz');oldangles=[]
        for i,x in enumerate(old['q']):
            put(x);oldangles.append(Rotation.from_matrix(np.einsum('bji,bjk->bik',targets[i],orientations(m,s))).magnitude())
        row['before_orientation_rmse_deg']={n:float(np.rad2deg(np.sqrt(np.mean(np.array(oldangles)[:,j]**2)))) for j,n in enumerate(BODIES)}
        np.savez_compressed(OUT/f'{pitch}_ik.npz',q=qs,time=data['time'],target=target,predicted=positions,orientation_targets=targets,coordinate_names=[c.getName() for c in coords])
        rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'ik_report.json').write_text(json.dumps(dict(rows=rows,orientation_weight_deg=10.,training_registration_pitch='3034_1',training_registration_sample=0,
        scope='Skin-frame temporal orientation transfer, not independent anatomical calibration. Source wrist remains welded; all original ranges/Fmax/geometry/muscle lengths retained.'),indent=2))

if __name__=='__main__':main()
