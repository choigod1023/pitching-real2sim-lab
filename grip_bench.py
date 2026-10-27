"""Three prescribed radial finger surrogates, with explicit actuator work."""
import json
from pathlib import Path
import numpy as np
from friction_contact import contact_force,RADIUS,PAD_RADIUS

MASS=.145
INERTIA=.4*MASS*RADIUS**2
DIR=np.array([[np.cos(a),np.sin(a),0] for a in np.arange(3)*2*np.pi/3])
DEPTH=.001
SMOOTH=.002


def opening(t,start,end):
    if t<=start:return 0.,0.
    if t>=end:return .012,0.
    u=(t-start)/(end-start)
    return .012*u*u*(3-2*u),.072*u*(1-u)/(end-start)


def hand(t,kind):
    center=np.array([0.,0.,.5]);v=np.zeros(3);w=np.zeros(3);angle=0.
    if kind=='motion':
        f=2*np.pi;center[0]=.02*np.sin(f*t);v[0]=.02*f*np.cos(f*t)
        angle=np.deg2rad(10)*np.sin(f*t);w[2]=np.deg2rad(10)*f*np.cos(f*t)
    r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    return center,v,w,r


def contacts(x,v,w,center,hv,hw,rotation,amount,rate):
    directions=DIR@rotation.T;pads=center+(RADIUS+PAD_RADIUS-DEPTH+amount)*directions
    force=np.zeros(3);torque=np.zeros(3);power=np.zeros(3);radialpower=0.;elastic=loss=depth=0.;forces=[];active=0
    for j,(pad,direction) in enumerate(zip(pads,directions)):
        delta=x-pad;dist=np.linalg.norm(delta);penetration=RADIUS+PAD_RADIUS-dist
        if dist<1e-10:raise ValueError('Coincident centers')
        normal=delta/dist;point=x-RADIUS*normal
        padv=hv+np.cross(hw,point-center)+rate*direction
        ballv=v+np.cross(w,point-x)
        f,e,l=contact_force(normal,penetration,ballv-padv,10000/3,2/3,.6,SMOOTH)
        force+=f;torque+=np.cross(point-x,f);power[j]=f@padv;radialpower+=f@(rate*direction)
        elastic+=e;loss+=l;depth=max(depth,penetration);forces.append(np.linalg.norm(f));active+=int(np.linalg.norm(f)>1e-7)
    return force,torque,power,elastic,loss,depth,np.array(forces),active,radialpower,pads


def simulate(kind,dt):
    duration=.22 if kind=='release' else .2;n=round(duration/dt);dt=duration/n
    y=np.zeros(16);c,hv,hw,r=hand(0,kind);y[:3]=c;y[3:6]=hv;y[6:9]=hw
    def rhs(t,y):
        c,hv,hw,r=hand(t,kind);a,ad=opening(t,.12,.13) if kind=='release' else (0.,0.)
        f,tau,p,e,l,depth,forces,active,rp,pads=contacts(y[:3],y[3:6],y[6:9],c,hv,hw,r,a,ad)
        dy=np.r_[y[3:6],f/MASS+np.array([0,0,-9.81]),tau/INERTIA,np.maximum(p,0),np.maximum(-p,0),l]
        return dy,(e,depth,forces,active,pads)
    def energy(y,elastic):return MASS*9.81*y[2]+.5*MASS*(y[3:6]@y[3:6])+.5*INERTIA*(y[6:9]@y[6:9])+elastic
    initial=energy(y,rhs(0,y)[1][0]);initial_elastic=rhs(0,y)[1][0]
    maxoffset=maxdepth=0.;maxforce=np.zeros(3);last=None;lost_before_open=False;afterforce=0.;states=[]
    for i in range(n):
        t=i*dt;k1,_=rhs(t,y);k2,_=rhs(t+dt/2,y+dt*k1/2);k3,_=rhs(t+dt/2,y+dt*k2/2);k4,_=rhs(t+dt,y+dt*k3)
        y+=dt*(k1+2*k2+2*k3+k4)/6;t=(i+1)*dt;_,meta=rhs(t,y)
        c=hand(t,kind)[0];offset=np.linalg.norm(y[:3]-c)
        if kind!='release' or t<=.12:maxoffset=max(maxoffset,offset)
        maxdepth=max(maxdepth,meta[1]);maxforce=np.maximum(maxforce,meta[2])
        if meta[3]:last=t
        elif kind!='release' or t<.12:lost_before_open=True
        if t>=.135:afterforce=max(afterforce,float(max(meta[2])))
        if i%20==0 or i==n-1:states.append(np.r_[t,y[:9],meta[4].ravel(),meta[2]])
    final=energy(y,meta[0]);residual=final-initial-y[9:12].sum()+y[12:15].sum()+y[15]
    result=dict(kind=kind,dt=dt,max_relative_offset_mm=maxoffset*1000,final_speed_m_s=float(np.linalg.norm(y[3:6])),
                last_contact_s=last,lost_before_open=lost_before_open,force_after_135ms_n=afterforce,
                peak_force_per_pad_n=maxforce.tolist(),max_penetration_mm=maxdepth*1000,
                work_positive_per_pad_j=y[9:12].tolist(),work_negative_per_pad_j=y[12:15].tolist(),loss_j=float(y[15]),
                initial_elastic_j=initial_elastic,energy_residual_j=float(residual),final_position_m=y[:3].tolist())
    return result,np.array(states)


def run():
    root=Path(__file__).resolve().parent;out=root/'artifacts/grip-bench';out.mkdir(parents=True,exist_ok=True);rows=[]
    for kind in ['static','motion','release']:
        a,_=simulate(kind,2e-5);b,states=simulate(kind,1e-5)
        pdiff=np.linalg.norm(np.array(a['final_position_m'])-b['final_position_m'])*1000;vdiff=abs(a['final_speed_m_s']-b['final_speed_m_s'])*3.6
        gate=not b['lost_before_open'] and abs(b['energy_residual_j'])<1e-4 and pdiff<.05 and vdiff<.01
        if kind=='static':gate=gate and b['max_relative_offset_mm']<.5 and b['final_speed_m_s']<.005
        if kind=='motion':gate=gate and b['max_relative_offset_mm']<2
        if kind=='release':
            free=states[states[:,0]>.135]
            accel=np.diff(free[:,4:7],axis=0)/np.diff(free[:,0])[:,None]
            b['freefall_acceleration_error_m_s2']=float(np.max(abs(accel-[0,0,-9.81])))
            gate=gate and b['last_contact_s'] is not None and .12<=b['last_contact_s']<=.135 and b['force_after_135ms_n']<1e-6 and b['freefall_acceleration_error_m_s2']<1e-7
        rows.append(dict(coarse=a,fine=b,position_resolution_mm=float(pdiff),speed_resolution_kmh=float(vdiff),passed=bool(gate)))
        np.savez_compressed(out/f'{kind}.npz',states=states);print(kind,gate,b,flush=True)
    result=dict(rows=rows,passed=all(r['passed'] for r in rows))
    (root/'grip_bench_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__':run()
