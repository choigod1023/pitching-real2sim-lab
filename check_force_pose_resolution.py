"""Recheck the example fitted trajectory at half the integration step."""
import json
import numpy as np
from force_pose_calibration import OUT,load,demands,ground_replay,integrate_inputs,endpoint,body,rms

def main():
    obs,m,_,dt,t,fs,ms,centers=load('3034_4')
    z=np.load(OUT/'3034_4_fit.npz');rows=[]
    for mode in ['before','after']:
        q=z['q_'+mode];v,_,loads,_=demands(m,q,dt,t,fs,ms,centers)
        for policy in ['residual_assisted','ground_only']:
            if policy=='residual_assisted':p,w,_=integrate_inputs(m,q[2],v[2],loads,dt,4,'foh')
            else:p,w=ground_replay(m,q[2:-2],v[2:-2],loads,dt,t[2:-2],fs,ms,centers,4)
            _,sites=body.positions(m,p);_,original=body.positions(m,z['q_before'][2:-2])
            rows.append(dict(mode=mode,policy=policy,substeps=4,
                speed_kmh=float(np.linalg.norm(endpoint(m,p[-1],w[-1]))*3.6),
                original_pose_rmse_mm=rms((sites-original).reshape(-1,3))*1000))
    (OUT/'resolution.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
