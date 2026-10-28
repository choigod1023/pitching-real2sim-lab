"""Outcome-blind motion matching on a fixed ID-parity internal evaluation cohort."""
import json
import hashlib
from itertools import combinations,product
from pathlib import Path

import numpy as np
import pandas as pd

from load_matched_pitching import read_selected,clustered_summary
from within_athlete_candidates import extract,PEAKS
from full_body_comparison import peak
from pilot import regular_time

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'openbiomechanics/baseball_pitching/data'
TARGET=np.array([1.8573,-1.948,2.0313,1000/360*2])
FACTORS={'A':[0],'B':[1,2],'C':[3]}
MASKS=[''.join(x for x,b in zip('ABC',bits) if b) for bits in product((0,1),repeat=3) if any(bits)]
FEATURES=['knee_extension','lateral_tilt','arm_slot','rear_peak_ms']


def evaluation_membership(users):
    """A partition that cannot depend on speed/load or membership in successful pairs."""
    return np.asarray(users,dtype=int)%2==1


def motion_direction(delta,mask):
    """Only motion values enter selection/orientation; speed and loads are inaccessible."""
    d=np.asarray(delta,float)
    if d.shape!=(4,) or not np.isfinite(d).all() or mask not in MASKS:raise ValueError('Invalid motion/mask')
    ids=[i for f in mask for i in FACTORS[f]]
    ratios=d/TARGET
    for direction in (1,-1):
        r=ratios[ids]*direction
        if np.all((r>=.25)&(r<=1.5)) and ('C' not in mask or abs(d[3])>1000/360+1e-6):
            return direction
    return 0


def main():
    out=ROOT/'artifacts/hypothesis-test';out.mkdir(parents=True,exist_ok=True)
    discovery=json.loads((ROOT/'load_matched_summary.json').read_text(encoding='utf-8'))
    discovery_users={p['user'] for p in discovery['selected_pair_records']}
    meta=pd.read_csv(DATA/'metadata.csv');poi=pd.read_csv(DATA/'poi/poi_metrics.csv')
    if meta.session_pitch.duplicated().any() or poi.session_pitch.duplicated().any():raise ValueError('Duplicate ID')
    cols=list(dict.fromkeys(['session_pitch','time','fp_10_time','BR_time',*PEAKS.values(),
                            'wrist_velo_x','wrist_velo_y','wrist_velo_z','rear_knee_velo_x']))
    signals=read_selected(DATA/'full_sig/joint_velos.csv',set(meta.session_pitch),cols)
    timing=[];excluded=[]
    for pid,g in signals.groupby('session_pitch'):
        try:
            g=g.sort_values('time').reset_index(drop=True)
            extract(g);t=regular_time(g);fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0])
            pt,_=peak(t,g.rear_knee_velo_x.to_numpy(float),fp,br)
            window=(t>=fp-.000051)&(t<=br+.000051)
            if not np.isfinite(g.loc[window,'rear_knee_velo_x']).all():raise ValueError('rear knee missing')
            timing.append(dict(session_pitch=pid,rear_peak_ms=(pt-br)*1000))
        except ValueError as err:excluded.append(dict(pitch=pid,reason=str(err)))
    d=meta.merge(poi.drop(columns=['pitch_speed_mph']),on=['session_pitch','session'],validate='one_to_one',how='left',indicator=True)
    if not d['_merge'].eq('both').all():raise ValueError('Missing POI identity join')
    d=d.merge(pd.DataFrame(timing),on='session_pitch',validate='one_to_one')
    d['knee_extension']=d.lead_knee_extension_from_fp_to_br;d['lateral_tilt']=d.torso_lateral_tilt_br
    d['arm_generation_j']=d.shoulder_generation_fp_br+d.elbow_generation_fp_br
    d['speed_kmh']=d.pitch_speed_mph*1.609344
    needed=[*FEATURES,'speed_kmh','elbow_varus_moment','shoulder_internal_rotation_moment','arm_generation_j']
    if not np.isfinite(d[needed].to_numpy(float)).all():raise ValueError('Missing features')
    # Do NOT exclude all 22 outcome-selected athletes: that would remove every
    # qualifying speed/load success by construction. Parity depends on IDs only.
    primary=d[evaluation_membership(d.user)]
    rows=[]
    for key,g in primary.groupby(['user','session','pitch_type']):
        for a,b in combinations(g.sort_values('session_pitch').to_dict('records'),2):
            delta=np.array([b[f]-a[f] for f in FEATURES])
            for mask in MASKS:
                direction=motion_direction(delta,mask)
                if not direction:continue
                target,reference=(b,a) if direction==1 else (a,b)
                r=dict(mask=mask,user=int(key[0]),reference=reference['session_pitch'],target=target['session_pitch'])
                r.update({k:target[k]-reference[k] for k in needed})
                active=[i for f in mask for i in FACTORS[f]];inactive=[i for i in range(4) if i not in active]
                r['inactive_scaled_change_max']=float(np.max(np.abs(delta[inactive]/TARGET[inactive]))) if inactive else 0.
                r['joint_observational_success']=(r['speed_kmh']>=1 and r['elbow_varus_moment']<=0 and r['shoulder_internal_rotation_moment']<=0 and r['arm_generation_j']<=0)
                rows.append(r)
    matched=pd.DataFrame(rows,columns=['mask','user','reference','target',*needed,'inactive_scaled_change_max','joint_observational_success']);results=[]
    endpoints=['speed_kmh','elbow_varus_moment','shoulder_internal_rotation_moment','arm_generation_j']
    for mask in MASKS:
        p=matched[matched['mask'].eq(mask)]
        results.append(dict(mask=mask,pairs=len(p),athletes=p.user.nunique(),
                            deltas=clustered_summary(p,endpoints) if len(p) else [],
                            joint_observational_success_pairs=int(p.joint_observational_success.sum()),
                            joint_success_athlete_balanced_fraction=float(p.groupby('user').joint_observational_success.mean().mean()) if len(p) else None,
                            inactive_scaled_change_median=float(p.inactive_scaled_change_max.median()) if len(p) else None))
    report=dict(scope='Internal exploratory association/falsification, not randomized intervention or pristine external validation',
                hypothesis='Small athlete-specific motion change can improve speed >=1 km/h without increased arm-load proxies',
                cohort_rule='odd athlete user IDs; representative user2 is even; reused internal data',
                previously_analyzed_30pair_users=sorted(discovery_users),primary_athletes=primary.user.nunique(),
                primary_pitches=len(primary),valid_total=len(d),signal_exclusions=excluded,
                target_motion_delta=dict(zip(FEATURES,TARGET.tolist())),
                matching_range_in_target_units=[.25,1.5],orientation_uses_outcomes=False,
                inactive_factors_controlled=False,pure_factorial_effect_identified=False,
                input_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [DATA/'metadata.csv',DATA/'poi/poi_metrics.csv',DATA/'full_sig/joint_velos.csv',ROOT/'load_matched_summary.json']},
                new_physical_velocity_optimization_executed=False,causal_hypothesis_confirmed=False,
                results=results)
    (out/'motion_matching.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    matched.to_csv(out/'motion_matched_pairs.csv',index=False)
    print('primary',primary.user.nunique(),'athletes',len(primary),'pitches')
    for r in results:
        means={x['feature']:round(x['mean_delta'],3) for x in r['deltas']}
        print(r['mask'],r['pairs'],r['athletes'],means,'joint_success',r['joint_observational_success_pairs'])


if __name__=='__main__':main()
