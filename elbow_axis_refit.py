"""Sensitivity IK using a skin-marker axis proxy, not anatomical calibration."""
import json
import numpy as np
import opensim as o
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from anatomical_shoulder_fit import model_points,arr,register
from directional_shoulder_fit import orientations
from elbow_registration_audit import INPUT,OUT
import wrist_residual_audit as audit

def main():
    o.Logger.setLevelString('error');m=o.Model(str(OUT/'registered_wrist.osim'));s=m.initSystem();m.realizePosition(s)
    train=np.load(INPUT.parent/'anatomical-shoulder/3034_1_input.npz');rotation,_=register(train['torso'][0],model_points(m,s)[:3])
    cs=m.getCoordinateSet();coords=[cs.get(i) for i in range(cs.getSize())];lo=np.array([c.getRangeMin() for c in coords]);hi=np.array([c.getRangeMax() for c in coords]);axis=np.array([.04940001,.03660001,.99810825]);axis/=np.linalg.norm(axis)
    reports=[];audit.OUT=OUT;audit.INPUT=OUT
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        z=np.load(INPUT/f'{pitch}_ik.npz');p=np.load(OUT/f'{pitch}_epicondyles.npz');ob=(p['lateral']-p['medial'])@rotation;ob/=np.linalg.norm(ob,axis=1)[:,None]
        n,nc=z['q'].shape;neq=len(arr(s.getQErr()));nr=18+neq+9+3
        sparse=lil_matrix((n*nr+(n-2)*nc,n*nc),dtype=int)
        for i in range(n):sparse[i*nr:(i+1)*nr,i*nc:(i+1)*nc]=1
        for i in range(n-2):sparse[n*nr+i*nc:n*nr+(i+1)*nc,i*nc:(i+3)*nc]=1
        def evaluate(flat,save=False):
            q=flat.reshape(n,nc);parts=[];pred=[];angs=[];ori=[]
            for i,x in enumerate(q):
                for c,v in zip(coords,x):c.setValue(s,float(v),False)
                m.realizePosition(s);R=m.getJointSet().get('elbow').getParentFrame().getTransformInGround(s).R().asMat33();world=np.array([[R.get(j,k) for k in range(3)] for j in range(3)])@axis
                pos=model_points(m,s);rot=orientations(m,s);delta=Rotation.from_matrix(np.einsum('bji,bjk->bik',z['orientation_targets'][i],rot)).as_rotvec()
                # Cross product treats a line as unsigned and preserves prior
                # target weights; 10° is a sensitivity setting, not accuracy.
                parts.append(np.r_[((pos-z['target'][i])/.005).ravel(),arr(s.getQErr())/.0001,(delta/np.deg2rad(10)).ravel(),np.cross(world,ob[i])/np.deg2rad(10)])
                if save:pred.append(pos);ori.append(rot);angs.append(np.rad2deg(np.arccos(np.clip(abs(world@ob[i]),0,1))))
            if save:return np.array(pred),np.array(ori),np.array(angs)
            return np.r_[np.concatenate(parts),np.diff(q,n=2,axis=0).ravel()*10.]
        q0=np.clip(z['q'],lo+1e-8,hi-1e-8).ravel();fit=least_squares(evaluate,q0,bounds=(np.tile(lo,n),np.tile(hi,n)),jac_sparsity=sparse.tocsr(),max_nfev=30,ftol=1e-5,xtol=1e-5,gtol=1e-5,x_scale='jac',tr_solver='lsmr')
        q=fit.x.reshape(n,nc);pos,ori,angles=evaluate(fit.x,True)
        np.savez_compressed(OUT/f'{pitch}_ik.npz',q=q,time=z['time'],target=z['target'],predicted=pos,orientation_targets=z['orientation_targets'],coordinate_names=z['coordinate_names'])
        report=audit.run(OUT/'registered_wrist.osim',pitch,'original');report['axis_proxy_rms_deg']=float(np.sqrt(np.mean(angles**2)));report['optimizer']={'nfev':fit.nfev,'status':int(fit.status),'cost':float(fit.cost),'initial_cost':float(np.sum(evaluate(q0)**2)/2)}
        reports.append(report);(OUT/'axis_refit_results.json').write_text(json.dumps({'rows':reports},indent=2));print(pitch,report['axis_proxy_rms_deg'],report['rms']['residual'],report['optimizer'],flush=True)

if __name__=='__main__':main()
