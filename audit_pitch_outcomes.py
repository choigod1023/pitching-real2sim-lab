"""Predeclared 0/10/20/30 degree diagnostic scenarios; no best-angle selection."""
import json
from pathlib import Path
import numpy as np
from pitch_outcomes import evaluate


def run():
    out=Path('artifacts/outcome-simulator')
    data=json.loads((out/'states.json').read_text())
    summaries=[]
    for angle in (0,10,20,30):
        result=evaluate(data['states'],elevation_offset_deg=angle)
        fine=evaluate(data['states'],elevation_offset_deg=angle,dt=.001,samples=2)
        for r,f in zip(result['records'],fine['records']):
            r['plate_step_difference_mm']=float(np.linalg.norm(np.array(r['plate_xz_m'])-f['plate_xz_m'])*1000) if r['status']==f['status']=='ok' else None
        result['provenance']={k:v for k,v in data.items() if k!='states'}
        (out/f'diagnostic-{angle}deg.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        summaries.append(dict(angle=angle,records=[{k:r.get(k) for k in ['name','status','release_speed_kmh','speed_change_kmh',
            'target_error_cm','target_error_reduction_cm','perturbation_spread_cm','work_change_j','sampled_budget_pass','plate_step_difference_mm']} for r in result['records']]))
    (out/'diagnostic-summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':run()
