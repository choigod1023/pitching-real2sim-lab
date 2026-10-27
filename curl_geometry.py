import numpy as np
import mujoco
from coupled_payload import model_for
from grip_bench import DIR


def load_frame(source):
    dt=float(source['dt']);n=round(.03/dt);start=len(source['qpos'])-1-n
    q=np.asarray(source['qpos']);v=np.asarray(source['qvel']);m=model_for(source['lengths'],dt,ball_mass=0.);d=mujoco.MjData(m)
    jp=np.zeros((3,m.nv));jr=jp.copy();vel=[];rot=None
    for index in [start,len(q)-1]:
        d.qpos[:]=q[index];d.qvel[:]=v[index];mujoco.mj_forward(m,d);mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
        vel.append(jp@d.qvel)
        if rot is None:rot=d.xmat[m.body('hand').id].reshape(3,3).copy()
    load_world=(vel[1]-vel[0])/(n*dt)-m.opt.gravity
    if np.linalg.norm(load_world)<1e-9:raise ValueError('Undefined mean loading direction')
    unit_world=load_world/np.linalg.norm(load_world);u=-rot.T@unit_world
    seed=np.eye(3)[np.argmin(abs(u))];v=seed-u*(seed@u);v/=np.linalg.norm(v)
    angles=np.arange(3)*2*np.pi/3
    dirs=np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
    return dirs,np.cross(u,v),unit_world,load_world


def path(t,dirs,axis,final_deg):
    u=np.clip(t/.020,0,1);angle=np.deg2rad(final_deg)*u*u*(3-2*u)
    rate=np.deg2rad(final_deg)*6*u*(1-u)/.020 if 0<t<.020 else 0.
    axis=np.asarray(axis);K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    rotation=np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
    new=dirs@rotation.T
    return new,np.cross(rate*axis,new)
