"""Lossless archive filename manifest. Never infer decimal places or speed units."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def parse_name(name):
    path=PurePosixPath(name)
    subject=re.fullmatch(r'Subject (\d+)',path.parent.name,re.I)
    match=re.fullmatch(r'(\d+)\s*oz\.?\s+(model|throw\s+(\d+)\s+(.+))\.c3d',path.name,re.I)
    row={'archive_path':name,'subject':int(subject[1]) if subject else None,
         'weight_oz':None,'kind':'unparsed','repeat':None,'speed_token':None,
         'speed_kmh':None,'speed_unit':'unverified','flags':[]}
    if not subject or not match:
        row['flags'].append('unparsed_name');return row
    row.update(weight_oz=int(match[1]),kind='static' if match[2].lower()=='model' else 'dynamic')
    if row['weight_oz'] not in range(3,8):row['flags'].append('unexpected_weight')
    if row['kind']=='dynamic':
        token=match[4];row.update(repeat=int(match[3]),speed_token=token)
        row['token_format']='decimal' if re.fullmatch(r'\d+\.\d+',token) else 'integer' if token.isdigit() else 'other'
        if row['token_format']!='decimal':row['flags'].append('speed_encoding_unresolved')
        if row['repeat'] not in (1,2,3):row['flags'].append('repeat_outside_1_to_3')
    return row


def build(inventory):
    rows=[]
    for item in inventory['files']:
        row=parse_name(item['file']);labels=item['labels']
        row.update(point_rate_hz=item['rate'],event_labels=item['event_labels'],
                   right_hand_exact=all(labels.count(x)==1 for x in ('RFIN','RWRA','RWRB')),
                   left_hand_exact=all(labels.count(x)==1 for x in ('LFIN','LWRA','LWRB')),
                   alias_labels=[x for x in labels if x.startswith('FKA-') or 'Skeleton' in x],
                   unlabeled_count=sum('unlabeled' in x.lower() for x in labels))
        rows.append(row)
    keys=Counter((r['subject'],r['weight_oz'],r['repeat']) for r in rows if r['kind']=='dynamic')
    missing=[];counts=[]
    for subject in sorted(set(r['subject'] for r in rows if r['subject'] is not None)):
        for weight in range(3,8):
            group=[r for r in rows if r['subject']==subject and r['weight_oz']==weight and r['kind']=='dynamic']
            counts.append({'subject':subject,'weight_oz':weight,'dynamic_count':len(group)})
            missing.extend({'subject':subject,'weight_oz':weight,'repeat':repeat} for repeat in (1,2,3) if (subject,weight,repeat) not in keys)
    summary={'files':len(rows),'kind_counts':dict(Counter(r['kind'] for r in rows)),
             'token_formats':dict(Counter(r.get('token_format') for r in rows if r['kind']=='dynamic')),
             'unparsed':[r['archive_path'] for r in rows if r['kind']=='unparsed'],
             'missing_expected_repeats':missing,'condition_counts':counts,
             'duplicate_keys':[list(k) for k,v in keys.items() if v>1],
             'extra_repeats':[r['archive_path'] for r in rows if 'repeat_outside_1_to_3' in r['flags']],
             'right_hand_exact_files':sum(r['right_hand_exact'] for r in rows),
             'files_with_aliases':sum(bool(r['alias_labels']) for r in rows),
             'files_with_unlabeled':sum(r['unlabeled_count']>0 for r in rows),
             'speed_conversion_authorized':False,'ball_release_validation_ready':False}
    return {'summary':summary,'files':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,default=Path('../ball-observability/artifacts/external/weighted-inventory.json'))
    p.add_argument('--out',type=Path,default=Path('artifacts/weighted-manifest.json'));a=p.parse_args()
    raw=a.inventory.read_bytes();result=build(json.loads(raw));result['inventory_sha256']=hashlib.sha256(raw).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result['summary'].items() if k!='condition_counts'},indent=2))
