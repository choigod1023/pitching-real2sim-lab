"""Score frozen visible tip regions; preserve missing and ambiguous identity."""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

BONES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
       (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
       (13,17),(0,17),(17,18),(18,19),(19,20)]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze_regions(source,destination):
    """Read the historical literal without executing the source script."""
    source=Path(source);destination=Path(destination)
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    values=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='samples' for target in node.targets)]
    if len(values)!=1:raise ValueError('Expected one literal samples table')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        old=load(destination)
        if old['source_sha256']!=digest:raise ValueError('Refusing to overwrite frozen regions')
        return
    payload={'source_sha256':digest,'samples':values[0],
             'interpretation':'Visible image-left/right tips; anatomical index/middle identity unverified; not direct contact or 3D'}
    destination.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')


def selected(row):
    index=row.get('selected_index')
    return None if index is None else row['candidates'][index]


def region_distance(point,box):
    x0,x1,y0,y1=box
    delta=np.maximum([x0-point[0],y0-point[1]],0)+np.minimum([x1-point[0],y1-point[1]],0)
    return float(np.linalg.norm(delta))


def match_tips(points,regions):
    """Optimistic unordered assignment, not verified index/middle identity."""
    points=np.asarray(points,float)
    if not regions:return []
    if len(regions)>len(points):raise ValueError('Insufficient candidate fingertips')
    assignments=list(itertools.permutations(range(len(points)),len(regions)))
    best=min(assignments,key=lambda p:sum(region_distance(points[j],b) for j,b in zip(p,regions)))
    return [dict(candidate_tip=(8,12)[j],region=list(b),
                 outside_region_px=region_distance(points[j],b),
                 center_distance_px=float(np.linalg.norm(points[j]-[(b[0]+b[1])/2,(b[2]+b[3])/2])))
            for j,b in zip(best,regions)]


def longest_run(frames):
    best=current=0;last=None
    for frame in sorted(set(frames)):
        current=current+1 if last is not None and frame==last+1 else 1
        best=max(best,current);last=frame
    return best


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def bone_variation(rows):
    lengths=[]
    for row in rows:
        hand=selected(row)
        if hand is None:continue
        world=np.asarray(hand['world21'],float)
        lengths.append([np.linalg.norm(world[a]-world[b]) for a,b in BONES])
    if len(lengths)<2:return None
    lengths=np.asarray(lengths)
    cv=np.std(lengths,axis=0)/np.maximum(np.mean(lengths,axis=0),1e-12)
    return dict(samples=len(lengths),median_segment_cv_percent=float(100*np.median(cv)),
                max_segment_cv_percent=float(100*np.max(cv)),
                interpretation='model-relative shape stability, not anatomical accuracy')


def evaluate(inference,manifest,manual):
    inputs={c['clip_id']:c for c in manifest['clips']};clips=[]
    for clip in inference['clips']:
        meta=inputs[clip['clip_id']]
        stem=Path(meta.get('path',meta.get('video_path',''))).stem
        samples={int(s['frame']):s for s in meta['samples']}
        truth=manual['samples'].get(stem,{})
        result={'clip_id':clip['clip_id'],'domain':clip['domain'],'methods':{}}
        for mode,rows in clip['methods'].items():
            target=[r for r in rows if samples[r['frame']]['role']=='target']
            controls=[r for r in rows if samples[r['frame']]['role']!='target']
            valid=[r['frame'] for r in target if selected(r) is not None]
            scoring=[];total_regions=0;missing_regions=0
            by_frame={r['frame']:r for r in rows}
            for key,ref in truth.items():
                boxes=[b for b in ref[1:] if b is not None]
                total_regions+=len(boxes)
                hand=selected(by_frame[int(key)]) if int(key) in by_frame else None
                if hand is None:
                    missing_regions+=len(boxes)
                    scoring.append({'frame':int(key),'missing':True,'region_count':len(boxes)})
                    continue
                points=np.asarray(hand['pixel21'],float)[[8,12]]
                scoring.append({'frame':int(key),'missing':False,'matches':match_tips(points,boxes)})
            matches=[m for row in scoring for m in row.get('matches',[])]
            result['methods'][mode]=dict(target_frames=len(target),detected_frames=len(valid),
                missing_frames=len(target)-len(valid),longest_detected_run=longest_run(valid),
                control_roi_samples=len(controls),control_unique_frames=len({r['frame'] for r in controls}),
                control_any_hand=sum(bool(r['candidates']) for r in controls),
                control_selected_hand=sum(selected(r) is not None for r in controls),
                frozen_tip_regions=total_regions,missing_tip_regions=missing_regions,
                scored_tip_regions=len(matches),
                tip_outside_region_mean_px=mean_or_none([m['outside_region_px'] for m in matches]),
                tip_region_center_mean_px=mean_or_none([m['center_distance_px'] for m in matches]),
                tip_within_3px_of_region=sum(m['outside_region_px']<=3 for m in matches),
                optimistic_tip_identity=True,relative_world_shape=bone_variation(target),region_rows=scoring)
        modes=list(clip['methods'])
        if len(modes)==2:
            left={r['frame']:selected(r) for r in clip['methods'][modes[0]]}
            right={r['frame']:selected(r) for r in clip['methods'][modes[1]]}
            agreement=[]
            for f in sorted(left.keys()&right.keys()):
                if samples[f]['role']!='target' or left[f] is None or right[f] is None:continue
                d=np.linalg.norm(np.array(left[f]['pixel21'])-np.array(right[f]['pixel21']),axis=1)
                agreement.append({'frame':f,'median21_px':float(np.median(d)),
                                  'max21_px':float(max(d)),'fingertips8_12_px':d[[8,12]].tolist()})
            result['input_agreement']={'common_target_frames':len(agreement),
                'mean_median21_px':mean_or_none([r['median21_px'] for r in agreement]),
                'not_accuracy':True,'rows':agreement}
        clips.append(result)
    matched_tip_errors=[]
    for clip in clips:
        modes=clip['methods']
        by_mode={m:{r['frame']:r for r in s['region_rows'] if not r['missing']} for m,s in modes.items()}
        for f in sorted(by_mode.get('raw',{}).keys()&by_mode.get('rotated',{}).keys()):
            for a,b in zip(by_mode['raw'][f]['matches'],by_mode['rotated'][f]['matches'],strict=True):
                matched_tip_errors.append((a['outside_region_px'],b['outside_region_px']))
    common={'regions':len(matched_tip_errors),'raw_mean_px':mean_or_none([r[0] for r in matched_tip_errors]),
            'rotated_mean_px':mean_or_none([r[1] for r in matched_tip_errors]),'optimistic_tip_identity':True}
    return {'clips':clips,'common_tip_region_comparison':common,'anatomical_tip_identity_verified':False,
            'outcome_prediction':{'speed_kmh':None,'rpm':None,'ivb_cm':None,'hb_cm':None},
            'note':'Missing outcomes are not zero errors. Input agreement is not accuracy.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--manual-source',type=Path);args=parser.parse_args()
    if args.manual_source:freeze_regions(args.manual_source,args.out/'manual-existing.private.json')
    paths=[args.out/n for n in ('inference.private.json','input_manifest.private.json','manual-existing.private.json')]
    result=evaluate(*(load(p) for p in paths))
    result['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.out/'evaluation.private.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps([{**{'clip':r['clip_id']},**{m:{k:v for k,v in s.items() if k not in ('region_rows','relative_world_shape')} for m,s in r['methods'].items()}} for r in result['clips']],ensure_ascii=False))


if __name__=='__main__':main()
