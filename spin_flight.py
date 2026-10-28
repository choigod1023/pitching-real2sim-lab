"""Vectorized baseball flight with explicit rpm, transverse spin and ball-energy accounting.

x lateral, y forward, z up. Bilinear Cl(S) from Sawicki et al. 2003 Eq29;
using transverse S for oblique spin is an explicit engineering extension.
No seam-shifted wake, spin decay, wind, athlete constraints or game-value model.
"""
import numpy as np

MASS=.145
RADIUS=.0366
INERTIA=.4*MASS*RADIUS**2
G=9.80665
RHO=1.225
K=.5*RHO*np.pi*RADIUS**2/MASS


def release_energy(speed_kmh,rpm,height):
    return .5*MASS*(np.asarray(speed_kmh)/3.6)**2+.5*INERTIA*(np.asarray(rpm)*2*np.pi/60)**2+MASS*G*np.asarray(height)


def energy_speed(energy,rpm,height):
    energy,rpm,height=np.broadcast_arrays(np.asarray(energy,float),np.asarray(rpm,float),np.asarray(height,float))
    transl=energy-.5*INERTIA*(rpm*2*np.pi/60)**2-MASS*G*height
    if not np.isfinite([energy,rpm,height]).all() or np.any(rpm<0) or np.any(transl<=0):raise ValueError('Impossible/nonfinite release energy')
    return np.sqrt(2*transl/MASS)*3.6


def velocities(speed_kmh,azimuth_deg,elevation_deg):
    speed,az,el=np.broadcast_arrays(np.asarray(speed_kmh,float)/3.6,np.radians(azimuth_deg),np.radians(elevation_deg))
    return np.stack((speed*np.cos(el)*np.sin(az),speed*np.cos(el)*np.cos(az),speed*np.sin(el)),axis=-1)


def spin_vectors(velocity,rpm,efficiency,tilt_deg):
    v=np.atleast_2d(np.asarray(velocity,float));rpm,eff,tilt=np.broadcast_arrays(np.asarray(rpm,float),np.asarray(efficiency,float),np.radians(tilt_deg))
    rpm,eff,tilt=[np.broadcast_to(a,(len(v),)) for a in (rpm,eff,tilt)]
    norm=np.linalg.norm(v,axis=1)
    if not np.isfinite(np.r_[v.ravel(),rpm,eff,tilt]).all() or np.any(norm<=0) or np.any(rpm<0) or np.any((eff<0)|(eff>1)):raise ValueError('Invalid spin inputs')
    forward=v/norm[:,None];back=np.tile([1.,0,0],(len(v),1));back-=np.sum(back*forward,axis=1)[:,None]*forward
    if np.any(np.linalg.norm(back,axis=1)<1e-8):raise ValueError('Degenerate forward direction')
    back/=np.linalg.norm(back,axis=1)[:,None]
    side=np.cross(forward,back)
    transverse=np.cos(tilt)[:,None]*back+np.sin(tilt)[:,None]*side
    axis=eff[:,None]*transverse+np.sqrt(1-eff**2)[:,None]*forward
    return axis*(rpm*2*np.pi/60)[:,None]


def lift_coefficient(spin_parameter):
    s=np.asarray(spin_parameter,float)
    if not np.isfinite(s).all() or np.any(s<0):raise ValueError('Invalid spin parameter')
    return np.where(s<=.1,1.5*s,.09+.6*s)


def accelerations(v,omega,cd=.35,lift_scale=1.,gravity=G):
    v,omega=np.asarray(v,float),np.asarray(omega,float)
    speed=np.linalg.norm(v,axis=-1);cross=np.cross(omega,v);magnitude=np.linalg.norm(cross,axis=-1)
    s=RADIUS*magnitude/np.maximum(speed**2,1e-16)
    direction=np.divide(cross,magnitude[...,None],out=np.zeros_like(cross),where=magnitude[...,None]>1e-15)
    lift=K*lift_scale*lift_coefficient(s)[...,None]*speed[...,None]**2*direction
    drag=-K*cd*speed[...,None]*v
    total=drag+lift;total[...,2]-=gravity
    return total


def flight_batch(position,velocity,omega,plate_y=16.8,cd=.35,lift_scale=1.,dt=.002,gravity=G,save_path=False):
    p,v,w=map(lambda a:np.atleast_2d(np.asarray(a,float)),(position,velocity,omega))
    if p.shape!=v.shape or w.shape!=v.shape or p.shape[1]!=3 or not np.isfinite(np.r_[p.ravel(),v.ravel(),w.ravel()]).all():raise ValueError('Need matching finite Nx3 states')
    if not np.isfinite([plate_y,cd,lift_scale,dt,gravity]).all() or dt<=0 or dt>.02 or cd<0 or lift_scale<0 or gravity<0 or np.any(p[:,1]>=plate_y) or np.any(p[:,2]<0) or np.any(v[:,1]<=0):raise ValueError('Invalid flight settings')
    state=np.c_[p,v];active=np.ones(len(p),bool);valid=np.zeros(len(p),bool);elapsed=np.zeros(len(p));path=[state.copy()] if save_path else None
    def rhs(s):return np.c_[s[:,3:],accelerations(s[:,3:],w,cd,lift_scale,gravity)]
    for _ in range(int(2/dt)):
        if not active.any():break
        a=rhs(state);b=rhs(state+dt*a/2);c=rhs(state+dt*b/2);d=rhs(state+dt*c)
        nxt=state+dt*(a+2*b+2*c+d)/6
        crossed=active&(nxt[:,1]>=plate_y)
        frac=np.divide(plate_y-state[:,1],nxt[:,1]-state[:,1],out=np.zeros(len(p)),where=np.abs(nxt[:,1]-state[:,1])>1e-15)
        arrived=state+frac[:,None]*(nxt-state)
        state[crossed]=arrived[crossed];elapsed[crossed]+=frac[crossed]*dt
        valid[crossed]=state[crossed,2]>=0
        moving=active&~crossed;state[moving]=nxt[moving];elapsed[moving]+=dt
        active &= ~crossed & (state[:,2]>=0) & np.isfinite(state).all(axis=1)
        if save_path:path.append(state.copy())
    return dict(state=state,time_s=elapsed,valid=valid,path=np.array(path) if save_path else None)


def aim(position,speed,rpm,efficiency,tilt,cd=.35,lift_scale=1.,target=(0,.9),dt=.002):
    """Re-aim each nominal scenario. This is assumed control, NOT learned command."""
    p=np.atleast_2d(np.asarray(position,float));n=len(p);angles=np.zeros((n,2));angles[:,1]=-1.5
    for _ in range(6):
        def end(a):
            v=velocities(speed,a[:,0],a[:,1]);w=spin_vectors(v,rpm,efficiency,tilt)
            out=flight_batch(p,v,w,cd=cd,lift_scale=lift_scale,dt=dt)
            if not out['valid'].all():raise ValueError('Aim iterate does not reach plate')
            return out['state'][:,[0,2]]
        base=end(angles);residual=base-target
        if np.max(np.linalg.norm(residual,axis=1))<1e-8:break
        step=.01;cols=[]
        for axis in (0,1):
            shifted=angles.copy();shifted[:,axis]+=step;cols.append((end(shifted)-base)/step)
        jac=np.stack(cols,axis=-1)
        correction=np.linalg.solve(jac,residual[...,None])[...,0]
        angles-=np.clip(correction,-1,1)
    return angles
