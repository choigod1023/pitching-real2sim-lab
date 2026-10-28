"""Frozen-pose repeatability after nearby perturbations and angle slices."""
import json
from pathlib import Path
import numpy as np
import opensim as o
from scholz_wrapping_trial import SOURCE,OUT,ROOT
COORDS=['plane_elv','shoulder_elv','axial_rot']

def main():
    o.Logger.setLevelString('error');m=o.Model(str(SOURCE));s=m.initSystem();cs=m.getCoordinateSet();z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');q=z['q'][28];names=list(z['coordinate_names']);indices=[names.index(n) for n in COORDS];mus=m.getMuscles().get('Subscapularis_I')
    def value(values):
        for j,v in enumerate(values):cs.get(j).setValue(s,float(v),False)
        m.realizePosition(s);return mus.getLength(s)
    reference=value(q);rng=np.random.default_rng(20261027);repeat=[]
    for i in range(64):
        trial=q.copy()
        for j in indices:trial[j]=np.clip(trial[j]+rng.uniform(-.02,.02),cs.get(j).getRangeMin(),cs.get(j).getRangeMax())
        value(trial);repeat.append(value(q))
    slices=[]
    for name,j in zip(COORDS,indices):
        coordinates=np.linspace(max(q[j]-.02,cs.get(j).getRangeMin()),min(q[j]+.02,cs.get(j).getRangeMax()),401);lengths=[]
        for v in coordinates:
            trial=q.copy();trial[j]=v;lengths.append(value(trial))
        slices.append({'coordinate':name,'q_rad':coordinates.tolist(),'length_m':lengths,'maximum_neighbor_length_change_mm':float(np.max(abs(np.diff(lengths)))*1000),'angle_step_deg':float(np.rad2deg(coordinates[1]-coordinates[0]))})
    report={'reference_length_m':reference,'same_pose_after_nearby_history_length_m':repeat,'same_pose_history_range_mm':float(np.ptp(repeat)*1000),'same_pose_max_deviation_mm':float(np.max(abs(np.array(repeat)-reference))*1000),'slices':slices}
    (OUT/'history_checks.json').write_text(json.dumps(report,indent=2));print('history range mm',report['same_pose_history_range_mm'],'slices',[(x['coordinate'],x['maximum_neighbor_length_change_mm'],x['angle_step_deg']) for x in slices])

if __name__=='__main__':main()
