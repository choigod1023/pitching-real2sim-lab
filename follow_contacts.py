import numpy as np
from grip_bench import DIR,DEPTH,SMOOTH,RADIUS,PAD_RADIUS,contact_force

def contacts(x,v,w,center,hv,hw,rotation,amount,rate):
    directions=DIR@rotation.T;pads=center+(RADIUS+PAD_RADIUS-DEPTH+amount)*directions
    force=np.zeros(3);torque=np.zeros(3);power=np.zeros(3);radialpower=0.;elastic=loss=depth=0.;forces=[];vectors=[];active=0
    for j,(pad,direction) in enumerate(zip(pads,directions)):
        delta=x-pad;dist=np.linalg.norm(delta);penetration=RADIUS+PAD_RADIUS-dist
        if dist<1e-10:raise ValueError('Coincident centers')
        normal=delta/dist;point=x-RADIUS*normal
        padv=hv+np.cross(hw,point-center)+rate*direction
        ballv=v+np.cross(w,point-x)
        f,e,l=contact_force(normal,penetration,ballv-padv,10000/3,2/3,.6,SMOOTH)
        vectors.append(f);force+=f;torque+=np.cross(point-x,f);power[j]=f@padv;radialpower+=f@(rate*direction)
        elastic+=e;loss+=l;depth=max(depth,penetration);forces.append(np.linalg.norm(f));active+=int(np.linalg.norm(f)>1e-7)
    return force,torque,power,elastic,loss,depth,np.array(forces),active,radialpower,pads,np.array(vectors)
