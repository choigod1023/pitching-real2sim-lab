"""Both-hand validity audit on every dynamic weighted trial; no imputation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import ezc3d
import numpy as np


def quality(c,side):
    labels=c['parameters']['POINT']['LABELS']['value'];required=[side+x for x in ('FIN','WRA','WRB')]
    if any(labels.count(x)!=1 for x in required):return {'status':'missing_or_ambiguous_labels'}
    p=c['data']['points'][:,[labels.index(x) for x in required],:]
    scale={'m':1.,'mm':.001,'cm':.01}[c['parameters']['POINT']['UNITS']['value'][0].strip().lower()]
    xyz=p[:3].transpose(2,1,0)*scale
    valid=np.isfinite(xyz).all(axis=(1,2))&(p[3]>=0).all(axis=0)
    area=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    valid&=area>1e-8
    edges=np.diff(np.r_[False,valid,False].astype(int));runs=np.where(edges==-1)[0]-np.where(edges==1)[0]
    return {'status':'ok','frames':len(valid),'valid_frames':int(valid.sum()),
            'valid_fraction':float(valid.mean()),'longest_valid_run_frames':int(max(runs,default=0))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,default=Path('../ball-observability/artifacts/external/Motion Data.zip'))
    p.add_argument('--out',type=Path,default=Path('artifacts/track-quality'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    records=[];failures=[];temporary=a.out/'inspect.c3d'
    with zipfile.ZipFile(a.zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith('.c3d') or 'model' in Path(name).name.lower():continue
            try:
                raw=archive.read(name);temporary.write_bytes(raw);c=ezc3d.c3d(str(temporary))
                records.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest(),'R':quality(c,'R'),'L':quality(c,'L')})
            except Exception as exc:failures.append({'file':name,'error':str(exc)})
    summary={'dynamic_files':len(records),'failures':failures}
    for side in ('R','L'):
        rows=[r[side] for r in records];ok=[r for r in rows if r['status']=='ok']
        summary[side]={'label_ok_files':len(ok),'all_frames_valid_files':sum(r['valid_fraction']==1 for r in ok),
                       'zero_valid_files':sum(r['valid_frames']==0 for r in ok),
                       'minimum_valid_fraction':min((r['valid_fraction'] for r in ok),default=None)}
    (a.out/'report.json').write_text(json.dumps({'summary':summary,'files':records},indent=2),encoding='utf-8')
    print(json.dumps(summary))
