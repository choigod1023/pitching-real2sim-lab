"""Post-selection numerical and assumption stress checks, not confidence bounds."""
import json
from pathlib import Path
import numpy as np
from compliant_timing import rollout,budget

def run():
    root=Path(__file__).resolve().parent;source=np.load(root.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz')
    selected=json.loads((root/'fine_timing_results.json').read_text())['selected']['shifts_ms']
    numerical=[]
    for sub in (4,8,16):
        a=rollout(source,(0,0,0),sub)[0];b=rollout(source,selected,sub)[0]
        numerical.append(dict(substeps=sub,baseline=a,candidate=b,gain_kmh=b['speed_kmh']-a['speed_kmh'],violations=budget(a,b)))
    sensitivity=[]
    for k in (5000.,10000.,20000.):
        for t in (.001,.002,.004):
            a=rollout(source,(0,0,0),4,k,t)[0];b=rollout(source,selected,4,k,t)[0]
            sensitivity.append(dict(stiffness=k,release_ms=t*1000,baseline=a,candidate=b,gain_kmh=b['speed_kmh']-a['speed_kmh'],violations=budget(a,b)))
    original=rollout(source,(-.5,.5,0),4)[0]
    result=dict(numerical=numerical,sensitivity=sensitivity,original_sub4=original,scope='Fixed post-selection checks; engineering stress ranges, not statistical confidence interval or radar error')
    (root/'fine_uncertainty_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(numerical=[(r['substeps'],r['gain_kmh'],r['violations']) for r in numerical],gain_range=[min(x['gain_kmh'] for x in sensitivity),max(x['gain_kmh'] for x in sensitivity)])))

if __name__=='__main__':run()
