import json
import numpy as np
import mujoco
from noise_structure_audit import OUT
from force_pose_calibration import load,demands,integrate_inputs,endpoint,rms,body

def main():
    obs,_,_,_,_,fs,ms,centers=load('3034_4');rows=[]
    for noise in ['original','rounding_bins']:
        for structure in ['fixed_origin','measured_mobile_origin']:
            stem=f'example_{noise}_{structure}'
            z=np.load(OUT/(stem+'.npz'));m=mujoco.MjModel.from_xml_path(str(OUT/(stem+'.xml')))
            q=z['q'];t=z['t'];dt=float(z['dt'])
            v,_,loads,_=demands(m,q,dt,t,fs,ms,centers)
            p,w,_=integrate_inputs(m,q[2],v[2],loads,dt,4,'foh')
            names,sites=body.positions(m,p);ix=[names.index(n) for n in ['shoulder_jc','elbow_jc','wrist']]
            rows.append(dict(noise=noise,structure=structure,substeps=4,
                speed_kmh=float(np.linalg.norm(endpoint(m,p[-1],w[-1]))*3.6),
                arm_rmse_mm=rms((sites[:,ix]-obs.points(t[2:-2])).reshape(-1,3))*1000))
    (OUT/'resolution.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
