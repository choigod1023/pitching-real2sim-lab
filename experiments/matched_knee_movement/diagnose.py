"""Prespecified post-result diagnostics; never replaces the primary result."""
import argparse
import json
from pathlib import Path
import cv2
from .analyze import read,digest,background_frame,background_features,compare_background,evaluate,FEATURES


def complete_link(order, compatible):
    groups=[]
    for key in order:
        for group in groups:
            if all(compatible[key,other] and compatible[other,key] for other in group):
                group.append(key);break
        else:groups.append([key])
    return groups


def run(folder):
    cv2.setNumThreads(2);r=read(folder/'results.private.json')
    keys=list(dict.fromkeys(k for p in r['rows'] for k in p['keys']))
    f={k:background_features(background_frame(r['source_clips'][k])[0]) for k in keys}
    matches={};compatible={}
    for a in keys:
        for b in keys:
            m={'pass':True,'identity':True} if a==b else compare_background(f[a],f[b])
            matches[a+'/'+b]=m;compatible[a,b]=m['pass']
    groups=complete_link(keys,compatible);cluster_rows=[]
    for index,group in enumerate(groups):
        eligible=[p for p in r['rows'] if set(p['keys'])<=set(group) and p['context_match']['strict']]
        analyses={}
        for cohort in sorted({p['cohort'] for p in r['rows']}):
            rows=[p for p in eligible if p['cohort']==cohort];ng=len({p['game'] for p in rows})
            analyses[cohort]={'n_pairs':len(rows),'n_games':ng,'models':[]}
            if ng>=3:
                analyses[cohort]['models']=[{'feature':feature,'scenarios':[evaluate(rows,feature,s) for s in range(9)]} for feature in FEATURES]
        cluster_rows.append({'cluster':index,'clips':group,'games':sorted({p['game'] for p in r['rows'] if any(k in group for k in p['keys'])}),
                             'eligible_pairs':[p['id'] for p in eligible],'analyses':analyses})
    ablation={}
    for cohort in sorted({p['cohort'] for p in r['rows']}):
        rows=[p for p in r['rows'] if p['cohort']==cohort and p['context_match']['strict'] and p['pair_camera']['pass']]
        ablation[cohort]=[{'feature':feature,'scenarios':[evaluate(rows,feature,s,control_mode='speed_rpm') for s in range(9)]} for feature in FEATURES]
    out={'diagnostic_protocol_sha256':digest(Path(__file__).with_name('DIAGNOSTIC_PROTOCOL.md')),
         'parent_results_sha256':digest(folder/'results.private.json'),'clusters':cluster_rows,'camera_matches':matches,'ablation':ablation}
    (folder/'diagnostics.private.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('clusters',[(len(g['clips']),len(g['games']),g['analyses']) for g in cluster_rows],flush=True)
    for name,models in ablation.items():
        for model in models:print(name,model['feature'],model['scenarios'][4]['mae_cm'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);run(parser.parse_args().folder)
