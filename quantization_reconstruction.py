"""Smooth only within known decimal rounding bins; no speed or force fitting."""
import numpy as np
from scipy.optimize import minimize

def smooth_rounding_bins(values,step=1e-4,maxiter=15000):
    y=np.asarray(values,float)
    if y.ndim==1:y=y[:,None]
    if y.ndim!=2 or len(y)<5 or not np.isfinite(y).all() or step<=0:raise ValueError('Invalid sampled positions')
    base=np.diff(y/step,n=3,axis=0);reg=1e-7
    def loss(flat):
        d=flat.reshape(y.shape);r=base+np.diff(d,n=3,axis=0);g=reg*d.copy()
        for i,c in enumerate([-1,3,-3,1]):g[i:i+len(r)]+=c*r
        return .5*np.sum(r*r)+.5*reg*np.sum(d*d),g.ravel()
    opt=minimize(loss,np.zeros(y.size),jac=True,method='L-BFGS-B',bounds=[(-.5,.5)]*y.size,options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-7,'maxcor':20})
    d=opt.x.reshape(y.shape)*step;out=y+d
    if np.max(np.abs(d))>step*.50000001:raise RuntimeError('Rounding interval violated')
    return out,dict(converged=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),max_adjustment_mm=float(np.max(abs(d))*1000),objective_before=float(loss(np.zeros(y.size))[0]),objective_after=float(opt.fun),rounding_step_m=step)


_CACHE={}
def reconstruct_landmarks(frame):
    import hashlib
    columns=[c for c in frame if c.endswith(('_x','_y','_z'))]
    values=frame[columns].to_numpy(float)
    if np.max(np.abs(values/1e-4-np.round(values/1e-4)))>1e-6:raise ValueError('Expected four-decimal position serialization')
    key=hashlib.sha256(values.tobytes()).hexdigest()
    if key not in _CACHE:
        result,info=smooth_rounding_bins(values)
        if not info['converged']:raise RuntimeError('Rounding reconstruction did not converge')
        _CACHE[key]=(result,info)
    result,info=_CACHE[key];out=frame.copy();out[columns]=result
    return out,dict(info)
