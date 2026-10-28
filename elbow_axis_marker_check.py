"""Compare model hinge axis with lateral-to-medial skin-marker proxy."""
import json
import numpy as np
import opensim as o
from anatomical_shoulder_fit import register,model_points
from elbow_registration_audit import INPUT,OUT

def main():
    o.Logger.setLevelString('error');m=o.Model(str(INPUT/'scaled_source.osim'));s=m.initSystem();m.realizePosition(s)
    train=np.load(INPUT.parent/'anatomical-shoulder/3034_1_input.npz')
    rotation,_=register(train['torso'][0],model_points(m,s)[:3]);cs=m.getCoordinateSet();axis=np.array([.04940001,.03660001,.99810825]);axis/=np.linalg.norm(axis)
    rows=[]
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        z=np.load(INPUT/f'{pitch}_ik.npz');p=np.load(OUT/f'{pitch}_epicondyles.npz');vec=(p['lateral']-p['medial'])@rotation
        separation=np.linalg.norm(vec,axis=1);vec/=separation[:,None];errors=[]
        for q,ob in zip(z['q'],vec):
            for j,v in enumerate(q):cs.get(j).setValue(s,float(v),False)
            m.realizePosition(s);R=m.getJointSet().get('elbow').getParentFrame().getTransformInGround(s).R().asMat33()
            world=np.array([[R.get(i,j) for j in range(3)] for i in range(3)])@axis
            errors.append(np.rad2deg(np.arccos(np.clip(abs(world@ob),0,1))))
        row=dict(pitch=pitch,angle_rmse_deg=float(np.sqrt(np.mean(np.array(errors)**2))),angle_median_deg=float(np.median(errors)),angle_max_deg=float(max(errors)),marker_separation_mm=float(np.mean(separation)*1000))
        rows.append(row);print(row)
    (OUT/'elbow_axis_marker_check.json').write_text(json.dumps({'rows':rows,'scope':'Unsigned angle to filtered skin-marker epicondyle line; not measured functional hinge axis. Soft tissue, marker placement, model registration and bone-frame alignment contribute.'},indent=2))

if __name__=='__main__':main()
