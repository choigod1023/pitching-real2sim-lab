"""Small control-timing diagnostic; never a calibrated muscle intervention."""
import itertools
import json
from pathlib import Path
import numpy as np
from compliant_timing import rollout,budget
from fine_compliant_timing import geometry

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/performance-restart'


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    source=np.load(ROOT.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz')
    grid=[(0.,0.,0.)]+[s for s in itertools.product([-.25,0.,.25],repeat=3) if s!=(0.,0.,0.)]
    bases={};rows=[]
    for sub in [2,4]:
        for shifts in grid:
            r,q,_,_,_,_=rollout(source,shifts,sub)
            if shifts==(0.,0.,0.):bases[sub]=(r,q)
            base,bq=bases[sub]
            angle,position=geometry(q,bq,source['lengths'],float(source['dt']))
            r.update(gain_kmh=r['speed_kmh']-base['speed_kmh'],violations=budget(base,r),
                     max_joint_rotation_deg=angle,max_wrist_change_mm=position,small_motion=angle<=3 and position<=10)
            rows.append(r)
        print('completed integration subdivision',sub,flush=True)
    accepted=[]
    for a,b in zip(rows[:27],rows[27:]):
        bound=abs(a['speed_kmh']-b['speed_kmh'])+abs(bases[2][0]['speed_kmh']-bases[4][0]['speed_kmh'])
        b['numerical_difference_bound_kmh']=bound
        if b['shifts_ms']!=[0.,0.,0.] and not a['violations'] and not b['violations'] and a['small_motion'] and b['small_motion'] and b['gain_kmh']>bound:
            accepted.append(b)
    best=max(rows[28:],key=lambda r:r['gain_kmh'])
    result=dict(nonzero_conditions=26,accepted_under_existing_engineering_budgets=len(accepted),
        validated_athlete_candidates=0,muscles_validated=False,spin_validated=False,baseline=bases[4][0],
        fastest_diagnostic=best,rows=rows,
        scope='Existing uncalibrated compliant hand-ball model; same force amplitudes and per-DOF budgets. Smaller timing grid is a diagnostic only. Numerical subdivision does not validate sampled controls or physiology. No video optical error correction transferred from Prime to3034.')
    (OUT/'joint_screen.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','baseline']},indent=2))
    return result


if __name__=='__main__':run()
