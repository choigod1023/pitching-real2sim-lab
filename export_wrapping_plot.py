"""Only query the local path inside its explicitly tested angle box."""
import json
import numpy as np
import opensim as o
from scholz_wrapping_trial import ROOT,OUT
from local_force_comparison import require_domain

def main():
    o.Logger.setLevelString('error');history=json.loads((OUT/'history_checks.json').read_text());curve=history['slices'][0];meta=json.loads((OUT/'local_0p002_length_based_checks.json').read_text());z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');q=z['q'][28];names=list(z['coordinate_names']);indices=[names.index(n) for n in ['plane_elv','shoulder_elv','axial_rot']]
    m=o.Model(str(OUT/'local_0p002_length_based_trial.osim'));s=m.initSystem();cs=m.getCoordinateSet();lengths=[]
    for v in curve['q_rad']:
        x=q.copy();x[indices[0]]=v
        try:require_domain(x[indices],meta['domain_low_rad'],meta['domain_high_rad'])
        except ValueError:lengths.append(float('nan'));continue
        for j,value in enumerate(x):cs.get(j).setValue(s,float(value),False)
        m.realizePosition(s);lengths.append(m.getMuscles().get('Subscapularis_I').getLength(s))
    np.savez_compressed(OUT/'wrapping_plot.npz',q_deg=np.rad2deg(curve['q_rad']),original_mm=np.array(curve['length_m'])*1000,local_mm=np.array(lengths)*1000,seed_deg=float(np.rad2deg(q[indices[0]])))

if __name__=='__main__':main()
