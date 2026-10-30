"""Summarize all registered recipients without hiding infeasible gains."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .study import COHORT


def compact(report,method):
    """Keep actual gains, gate failures and uncovered costs together."""
    pair=report['refit']['selected'] if method=='refit' else report['transfers'][method]
    r=pair['fine']
    row={'pitch':report['pitch'],'method':method,
         'baseline_kmh':report['baseline']['fine']['speed_kmh'],'candidate_kmh':r['speed_kmh'],
         'gain_kmh':r['paired_gain_kmh'],'transfer_pass':pair['judgement']['pass'],
         'A_pass':r['budget_gate']['strict_model_budget_pass'],'pose_pass':r['pose_screen_pass'],
         'violations':r['budget_gate']['violations'],'B_pass':pair['extra']['B_gate']['policy_budget_pass'],
         'net_work_change_j':pair['extra']['net_work_change_j'],
         'root_force_norm_peak_change_n':pair['extra']['root_force_norm_peak_change_n'],
         'positive_component_work_change_j':pair['extra']['positive_component_work_change_j'],
         'negative_component_work_change_j':pair['extra']['negative_component_work_change_j'],
         'rotation_deg':r['max_rotation_change_deg'],'ball_position_mm':r['max_ball_position_change_mm'],
         'direction_deg':r['release_direction_change_deg'],
         'resolution_gain_difference_kmh':pair['judgement']['gain_resolution_difference_kmh'],
         'verified':pair['verification']}
    if method=='refit':
        opt=report['refit']['optimization']
        row.update(paired_evaluations=opt['paired_evaluations'],optimizer_messages=opt['messages'],
                   baseline_fallback=opt['baseline_fallback'])
    return row


def summarize(reports):
    """Require the complete cohort; refits never count as frozen transfer."""
    lookup={r['pitch']:r for r in reports}
    if len(lookup)!=len(reports) or set(lookup)!=set(COHORT):
        raise ValueError('Complete unique preregistered cohort required')
    rows=[compact(lookup[p],m) for p in COHORT for m in ('normalized','absolute')]
    rows.extend(compact(lookup[p],'refit') for p in COHORT[1:5] if 'refit' in lookup[p])
    counts={}
    for label,pitches in (('same_athlete',COHORT[1:5]),('other_athletes',COHORT[5:])):
        counts[label]={m:{'passed':sum(r['transfer_pass'] for r in rows if r['pitch'] in pitches and r['method']==m),
                          'denominator':len(pitches)} for m in ('normalized','absolute')}
    candidates=[r for r in rows if r['pitch']!='3034_1' and r['transfer_pass']]
    return {'rows':rows,'frozen_transfer_counts':counts,
            'video_candidate':max(candidates,key=lambda r:r['gain_kmh']) if candidates else None,
            'same_athlete_refit_count':sum(r['method']=='refit' for r in rows),
            'real_effect_verified':False,'effect_95ci':None,'independent_external_validation':False,
            'scope':'Model-conditional applied-input feasibility; not an anatomical coaching prescription or counterfactual accuracy validation.'}


def archive(source,out):
    """Save deterministic compact archives with raw and compressed hashes."""
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    reports=[];manifest={}
    for pitch in COHORT:
        path=source/pitch/'report.json';raw=path.read_bytes();reports.append(json.loads(raw))
        compressed=gzip.compress(raw,mtime=0);(out/f'{pitch}.json.gz').write_bytes(compressed)
        manifest[pitch]={'original_path':str(path.resolve()),'raw_sha256':hashlib.sha256(raw).hexdigest(),
                         'gzip_sha256':hashlib.sha256(compressed).hexdigest()}
    result=summarize(reports);result['archives']=manifest
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=archive(a.source,a.out);print(json.dumps(r['frozen_transfer_counts']))
