"""Export reconstructed poses/orientations; this is IK, not muscle rollout."""
import numpy as np
import opensim as o
from anatomical_shoulder_fit import OUT as BASE
from directional_shoulder_fit import OUT,orientations

def main():
    o.Logger.setLevelString('error');m=o.Model(str(BASE/'scaled_source.osim'));s=m.initSystem();cs=m.getCoordinateSet()
    before=np.load(BASE/'3034_4_ik.npz');after=np.load(OUT/'3034_4_ik.npz');rots=[]
    for q in [before['q'],after['q']]:
        rows=[]
        for x in q:
            for i,v in enumerate(x):cs.get(i).setValue(s,float(v),False)
            m.realizePosition(s);rows.append(orientations(m,s))
        rots.append(rows)
    np.savez_compressed(OUT/'overlay.npz',before=before['predicted'],after=after['predicted'],observed=after['target'],
        before_rotation=rots[0],after_rotation=rots[1],target_rotation=after['orientation_targets'],time=after['time'])

if __name__=='__main__':main()
