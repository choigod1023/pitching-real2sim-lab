import hashlib,json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from paired_motion_baseline import FEATURES
from personal_calibration import fit_population,predict


def run():
    root=Path(__file__).resolve().parent
    frame=pd.read_csv(root.parent/'temporal-motion-gate/artifacts/temporal-motion/predictions.csv')
    groups={}
    for u,g in frame.groupby('user'):
        g=g.sort_values('pitch')
        groups[int(u)]=(np.vstack([np.zeros(len(FEATURES)),g[FEATURES].to_numpy()]),np.r_[0.,g.actual.to_numpy()])
    original=json.loads((root/'personal_results.json').read_text());rows=[];per=[]
    for fold in original['folds']:
        u=fold['user'];x,y=groups[u];choice=fold['personal_choice']
        model=fit_population([v for k,v in groups.items() if k!=u],choice['alpha'])
        local=[]
        for combo in combinations(range(len(y)),2):
            cal=np.array(combo);test=np.array([i for i in range(len(y)) if i not in combo])
            pred=predict(model,x[cal],y[cal],x[test],choice['strength'])
            r=dict(user=u,calibration_indices=list(combo),personal_mae=float(np.mean(abs(pred-y[test]))),mean_mae=float(np.mean(abs(y[cal].mean()-y[test]))))
            rows.append(r);local.append(r)
        errors=np.array([[r['personal_mae'],r['mean_mae']] for r in local])
        per.append(dict(user=u,combinations=len(local),personal_mean=float(errors[:,0].mean()),baseline_mean=float(errors[:,1].mean()),personal_min=float(errors[:,0].min()),personal_max=float(errors[:,0].max()),wins=int(sum(errors[:,0]<errors[:,1]))))
    result=dict(scope='Post-result exploratory overlapping calibration split sensitivity; original tuning frozen.',
        combinations=len(rows),athletes=len(per),personal_mae=float(np.mean([p['personal_mean'] for p in per])),baseline_mae=float(np.mean([p['baseline_mean'] for p in per])),
        athletes_winning_every_split=sum(p['wins']==p['combinations'] for p in per),athletes_with_mixed_wins=sum(0<p['wins']<p['combinations'] for p in per),per_athlete=per,rows=rows,
        protocol_sha256=hashlib.sha256((root/'CALIBRATION_SENSITIVITY_PROTOCOL.md').read_bytes()).hexdigest())
    (root/'calibration_sensitivity_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','per_athlete']}))


if __name__=='__main__':run()
