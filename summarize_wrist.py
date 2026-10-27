"""Summarize comparison and export same-target overlay data."""
import json
import numpy as np
import opensim as o
from wrist_muscle_integration import OUT,BASE
from directional_shoulder_fit import orientations

def main():
    before=json.loads((BASE/'ik_report.json').read_text())['rows'];after=json.loads((OUT/'ik_report.json').read_text())['rows']
    bs=json.loads((BASE/'muscle_screen.json').read_text())['reports'];ns=json.loads((OUT/'muscle_screen.json').read_text())['reports']
    def screen(r):
        valid=[v['hill_equilibrium_interval'] for v in r['rows'] if v['hill_equilibrium_interval']['status']=='success']
        return dict(success_frames=r['hill_success_frames'],frames=r['frames'],residual_nm=r['hill_residual_rms_nm'],max_endpoint_n=max((v['endpoint_force_max_n'] for v in valid),default=None))
    rows=[dict(pitch=a['pitch'],before=b,after=a,before_muscle=screen(x),after_muscle=screen(y)) for b,a,x,y in zip(before,after,bs,ns)]
    report=dict(rows=rows,candidate_search_executed=False,scope='Marker IK and instantaneous equilibrium interval diagnostic. No muscle forward rollout, ball speed prediction, or athlete performance validation.')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    o.Logger.setLevelString('error');m=o.Model(str(BASE/'scaled_source.osim'));s=m.initSystem();cs=m.getCoordinateSet()
    old=np.load(BASE/'3034_4_ik.npz');new=np.load(OUT/'3034_4_ik.npz');rots=[]
    for q in old['q']:
        for j,v in enumerate(q):cs.get(j).setValue(s,float(v),False)
        m.realizePosition(s);rots.append(orientations(m,s))
    np.savez_compressed(OUT/'overlay.npz',before=old['predicted'],after=new['predicted'],observed=new['target'],before_rotation=rots,after_rotation=new['rotations'],target_rotation=new['orientation_targets'],time=new['time'])

if __name__=='__main__':main()
