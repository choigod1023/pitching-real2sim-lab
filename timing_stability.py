"""Report both common and independently shifted release-time sign stability."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def stability(data):
    results=[]
    for method,g in data.groupby('method'):
        pairs=[]
        for user,a in g.groupby('user'):
            wide=a.pivot(index='pitch',columns='offset_frames',values='predicted_kmh').sort_index()
            if wide.isna().any().any() or list(wide.columns)!=[-2,-1,0,1,2]:raise ValueError('Incomplete timing grid')
            base=wide.iloc[0].to_numpy()
            for j in range(1,len(wide)):
                other=wide.iloc[j].to_numpy()
                common=other-base
                independent=(other[:,None]-base[None,:]).ravel()
                pairs.append({'user':int(user),'base':str(wide.index[0]),'pitch':str(wide.index[j]),
                              'common_sign_flip':bool(common.min()<0<common.max()),
                              'independent_sign_flip':bool(independent.min()<0<independent.max()),
                              'common_delta_span_kmh':float(np.ptp(common)),
                              'independent_delta_span_kmh':float(np.ptp(independent))})
        results.append({'method':method,'pairs':pairs,'pair_count':len(pairs),
                        'common_sign_flip_count':sum(p['common_sign_flip'] for p in pairs),
                        'independent_sign_flip_count':sum(p['independent_sign_flip'] for p in pairs)})
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result=stability(pd.read_csv(a.directory/'predictions.csv'))
    (a.directory/'stability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='pairs'} for r in result],indent=2))
