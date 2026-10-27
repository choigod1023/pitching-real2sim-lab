"""Same model, kinematics, force bounds; only torque-map calculation differs."""
import json,shutil
import numpy as np
import anatomical_muscle_screen as screen
from scholz_wrapping_trial import ROOT,OUT,SOURCE

def main():
    results=[]
    for mapping in ['legacy_speed','applied_force']:
        directory=OUT/mapping;directory.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE,directory/'scaled_source.osim')
        for i in range(1,6):shutil.copyfile(ROOT/f'artifacts/wrist-muscle/3034_{i}_ik.npz',directory/f'3034_{i}_ik.npz')
        screen.main(directory,mapping);r=json.loads((directory/'muscle_screen.json').read_text())['reports'];results.append({'mapping':mapping,'mean_residual_nm':float(np.mean([x['hill_residual_rms_nm'] for x in r])),'pitches':[{k:v for k,v in x.items() if k!='rows'} for x in r]})
    (OUT/'force_mapping_comparison.json').write_text(json.dumps({'results':results,'scope':'Native force-map correction only. Wrapped length/velocity consistency and continuous muscle dynamics remain unresolved; no athlete performance inference.'},indent=2));print(results,flush=True)

if __name__=='__main__':main()
