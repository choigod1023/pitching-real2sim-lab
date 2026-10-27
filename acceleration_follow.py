import json,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from grip_bench import MASS,INERTIA
from follow_path import follow_offset
from follow_contacts import contacts


def simulate(axis,acc,follow,dt=2e-5):
    duration=.03;n=round(duration/dt);dt=duration/n;y=np.zeros(22);y[2]=.5
    acceleration=np.eye(3)[axis]*acc
    def rhs(t,y):
        center=np.array([0,0,.5])+.5*acceleration*t*t;hv=acceleration*t
        shift,rate=follow_offset(y[:3]-center,y[3:6]-hv,follow)
        f,tau,p,e,l,dep,forces,active,_,pads,vectors=contacts(y[:3],y[3:6],y[6:9],center+shift,hv+rate,np.zeros(3),np.eye(3),0,0)
        fp=vectors@rate
        dy=np.r_[y[3:6],f/MASS+[0,0,-9.81],tau/INERTIA,np.maximum(p,0),np.maximum(-p,0),l,np.maximum(fp,0),np.maximum(-fp,0)]
        return dy,(e,dep,forces,active,pads,shift,center)
    def energy(y,e):return MASS*9.81*y[2]+.5*MASS*(y[3:6]@y[3:6])+.5*INERTIA*(y[6:9]@y[6:9])+e
    initial=energy(y,rhs(0,y)[1][0]);first=None;last=None;count=0;peak=np.zeros(3);depth=offset=0.;states=[]
    for i in range(n):
        t=i*dt;k1,_=rhs(t,y);k2,_=rhs(t+dt/2,y+dt*k1/2);k3,_=rhs(t+dt/2,y+dt*k2/2);k4,_=rhs(t+dt,y+dt*k3)
        y+=dt*(k1+2*k2+2*k3+k4)/6;t=(i+1)*dt;_,meta=rhs(t,y)
        peak=np.maximum(peak,meta[2]);depth=max(depth,meta[1]);offset=max(offset,np.linalg.norm(y[:3]-meta[6]))
        if meta[3]:count+=1;last=t
        elif first is None:first=t
        if i%5==0 or i==n-1:states.append(np.r_[t,y[:9],meta[4].ravel(),meta[2],meta[5]])
    residual=energy(y,meta[0])-initial-y[9:12].sum()+y[12:15].sum()+y[15]
    r=dict(axis='xz'[axis//2],acceleration_m_s2=acc,follow=follow,dt=dt,speed_kmh=float(np.linalg.norm(y[3:6])*3.6),
        first_loss_ms=None if first is None else first*1000,last_contact_ms=None if last is None else last*1000,contact_fraction=count/n,
        peak_pad_force_n=peak.tolist(),max_penetration_mm=float(depth*1000),max_ball_hand_offset_mm=float(offset*1000),
        work_positive_j=y[9:12].tolist(),work_negative_j=y[12:15].tolist(),follow_positive_j=y[16:19].tolist(),follow_negative_j=y[19:22].tolist(),
        energy_residual_j=float(residual),final_position_m=y[:3].tolist())
    return r,np.array(states)


def job(args):
    r,states=simulate(*args);out=Path(__file__).resolve().parent/'artifacts/grip-follow';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f"acc-{r['axis']}-{int(args[1])}-{int(args[2])}.npz",states=states)
    return r


def run():
    args=list(itertools.product([0,2],[0.,25.,50.,100.,250.,500.,1000.],[False,True]))
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[]
        for i,r in enumerate(pool.map(job,args)):rows.append(r);print('done',i+1,flush=True)
    root=Path(__file__).resolve().parent
    (root/'acceleration_follow_results.json').write_text(json.dumps(dict(rows=rows),indent=2),encoding='utf8')


if __name__=='__main__':run()
