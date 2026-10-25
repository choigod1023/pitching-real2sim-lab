"""Inventory CustomMarker10 without interpreting marker identity from speed."""
import argparse
import hashlib
import json
from pathlib import Path
import ezc3d
import numpy as np


def trajectory_stats(points, rate, scale):
    xyz=points[:3].T*scale
    valid=np.isfinite(xyz).all(axis=1)&(points[3]>=0)
    consecutive=valid[:-1]&valid[1:]
    speed=np.linalg.norm(np.diff(xyz,axis=0),axis=1)*rate*3.6
    return {'frames':len(valid),'valid_frames':int(valid.sum()),
            'valid_consecutive_pairs':int(consecutive.sum()),
            'axis_range_mm':(np.ptp(xyz[valid],axis=0)*1000).tolist() if valid.any() else None,
            'max_step_speed_kmh':float(speed[consecutive].max()) if consecutive.any() else None,
            'residual_values':np.unique(points[3]).tolist()}


def run(data,out):
    rows=[];failures=[];files=sorted(data.rglob('*.c3d'))
    for p in files:
        try:
            c=ezc3d.c3d(str(p));point=c['parameters']['POINT'];labels=point['LABELS']['value']
            if 'CustomMarker10' not in labels:continue
            scale={'m':1.,'mm':.001,'cm':.01}[point['UNITS']['value'][0].strip().lower()]
            i=labels.index('CustomMarker10')
            rows.append({'file':p.name,'static':p.name.endswith('_model.c3d'),
                         'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'description':point.get('DESCRIPTIONS',{}).get('value',[])[i:i+1],
                         **trajectory_stats(c['data']['points'][:,i,:],float(point['RATE']['value'][0]),scale)})
        except Exception as e:failures.append({'file':p.name,'error':str(e)})
    result={'scanned_files':len(files),'matches':rows,'failures':failures,
            'interpretation':'Identity unresolved; static-only occurrence cannot supply a pitching ball-release trajectory.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('../openbiomechanics/baseball_pitching/data/c3d'))
    p.add_argument('--out',type=Path,default=Path('artifacts/custom-marker/report.json'));a=p.parse_args();run(a.data,a.out)
