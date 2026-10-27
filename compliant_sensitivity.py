import json
from pathlib import Path
import numpy as np
from compliant_timing import rollout,budget


def run():
    root=Path(__file__).resolve().parent
    src=np.load(root.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz')
    rows=[]
    for k in [5000.,10000.,20000.]:
        for release in [.001,.002,.004]:
            a=rollout(src,(0,0,0),2,k,release)[0];b=rollout(src,(.5,.5,-.5),2,k,release)[0]
            rows.append(dict(stiffness=k,release_ms=release*1000,baseline=a,candidate=b,gain_kmh=b['speed_kmh']-a['speed_kmh'],violations=budget(a,b)))
    (root/'compliant_sensitivity_results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print(json.dumps([dict(k=r['stiffness'],release_ms=r['release_ms'],baseline=r['baseline']['speed_kmh'],gain=r['gain_kmh'],budget_pass=not r['violations']) for r in rows]))


if __name__=='__main__':run()
