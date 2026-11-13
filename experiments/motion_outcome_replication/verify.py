"""Independent scalar checks on stored results, with no fit re-selection."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np


def verify(folder):
    result=json.loads((folder/'results.private.json').read_text(encoding='utf-8'))
    tracks=np.load(folder/'mapped-tracks.private.npz',allow_pickle=False)
    angle_diffs=[];prediction_diffs=[]
    for key in result['clips']:
        pts=tracks[key+'_points']
        for name in ('lead_knee','throwing_elbow','trunk_tilt','arm_direction'):
            for index in np.flatnonzero(tracks[key+'_'+name+'_mask']):
                p=pts[index]
                if name in ('lead_knee','throwing_elbow'):
                    a,b,c=(23,25,27) if name=='lead_knee' else (12,14,16)
                    u=[float(p[a,j]-p[b,j]) for j in range(2)];v=[float(p[c,j]-p[b,j]) for j in range(2)]
                    angle=math.degrees(math.atan2(abs(u[0]*v[1]-u[1]*v[0]),u[0]*v[0]+u[1]*v[1]))
                else:
                    if name=='trunk_tilt':
                        x=(float(p[11,0])+float(p[12,0])-float(p[23,0])-float(p[24,0]))/2
                        y=(float(p[11,1])+float(p[12,1])-float(p[23,1])-float(p[24,1]))/2
                    else:x,y=[float(p[16,j]-p[12,j]) for j in range(2)]
                    angle=math.degrees(math.atan2(x,-y))
                angle_diffs.append(abs(angle-float(tracks[key+'_'+name][index])))
    for cohort,summary in result['cohorts'].items():
        pairs=[p for p in result['pairs'] if p['cohort']==cohort]
        for model in summary['models']:
            feature,target=model['feature'],model['target']
            for fold in model['folds']:
                train=[p for p in pairs if p['game']!=fold['game'] and p['features'][feature]['nominal'] is not None]
                counts=Counter(p['game'] for p in train)
                if len(train)>=2 and len(counts)>=2:
                    xs=sorted(abs(p['features'][feature]['nominal']) for p in train)
                    m=len(xs)//2;scale=max(1.,xs[m] if len(xs)%2 else (xs[m-1]+xs[m])/2)
                    num=den=0.
                    for p in train:
                        x=p['features'][feature]['nominal'];y=p['delta'][target];w=1/counts[p['game']]/len(counts)
                        num+=w*x*y;den+=w*x*x
                    slope=num/(den+scale**2)
                    prediction_diffs.append(abs(slope-fold['slope']))
                else:slope=None
                for pred in fold['predictions']:
                    p=next(p for p in pairs if p['id']==pred['pair'])
                    assert p['game']==fold['game'] and p['game'] not in fold['train_games']
                    x=p['features'][feature]['nominal'];estimate=0. if x is None or slope is None else x*slope
                    prediction_diffs.append(abs(estimate-pred['predicted']))
    report={'scalar_angles_checked':len(angle_diffs),'max_angle_difference_deg':max(angle_diffs),
            'scalar_slope_prediction_checks':len(prediction_diffs),'max_prediction_difference':max(prediction_diffs),
            'unique_source_count':len({c['source_sha256'] for c in result['clips'].values()}),
            'all_nominal_folds_game_disjoint':True}
    assert report['max_angle_difference_deg']<1e-9 and report['max_prediction_difference']<1e-9
    (folder/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder)
