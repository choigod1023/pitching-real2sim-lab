"""Audit alternative filename encodings, without accepting a trial identity."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

POLICIES=('literal','three_digit_tenths','three_digit_tenths_strip_y')


def decode(token,policy):
    if policy not in POLICIES:raise ValueError('Unknown hypothesis')
    if policy=='three_digit_tenths_strip_y' and re.fullmatch(r'\d{3}y',token):token=token[:-1]
    if not re.fullmatch(r'\d+(?:\.\d+)?',token):return None
    if policy!='literal' and re.fullmatch(r'\d{3}',token):return float(token)/10
    return float(token)


def audit(manifest,table,policy):
    rows=[]
    for f in manifest['files']:
        if f['kind']!='dynamic':continue
        value=decode(f['speed_token'],policy)
        candidates=[]
        if value is not None:
            subset=table[(table.subject==f['subject'])&(table.ball_weight==str(f['weight_oz'])+'oz')]
            candidates=[int(i) for i in subset.index if abs(float(subset.loc[i,'pitch_velocity'])-value)<=.0001]
        rows.append({'file':f['archive_path'],'raw_token':f['speed_token'],
                     'hypothetical_mph':value,'hypothetical_kmh':None if value is None else value*1.609344,
                     'candidate_table_rows':candidates})
    degree=Counter(i for r in rows for i in r['candidate_table_rows'])
    for r in rows:
        ids=r['candidate_table_rows']
        r['mutually_unique_candidate']=len(ids)==1 and degree[ids[0]]==1
        r['confirmed_identity']=False
    summary={'policy':policy,'dynamic_files':len(rows),'table_rows':len(table),
             'tolerance_mph':.0001,'unparsed_tokens':sum(r['hypothetical_mph'] is None for r in rows),
             'no_candidate_files':sum(not r['candidate_table_rows'] for r in rows),
             'multiple_candidate_files':sum(len(r['candidate_table_rows'])>1 for r in rows),
             'mutually_unique_candidates':sum(r['mutually_unique_candidate'] for r in rows),
             'unmatched_table_rows':[int(i) for i in table.index if degree[i]==0],
             'multiply_claimed_table_rows':[int(i) for i in table.index if degree[i]>1],
             'confirmed_matches':0}
    return {'summary':summary,'files':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('../weighted-manifest/artifacts/weighted-manifest.json'))
    p.add_argument('--table',type=Path,default=Path('../weighted-manifest/artifacts/compiled/data_table.csv'))
    p.add_argument('--out',type=Path,default=Path('artifacts/linkage'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text());table=pd.read_csv(a.table)
    for policy in POLICIES:
        r=audit(manifest,table,policy)
        r['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.manifest,a.table)}
        (a.out/(policy+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(json.dumps(r['summary']))
