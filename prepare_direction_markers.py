"""Raw skin-marker frames; not anatomical bone orientations."""
import json
from pathlib import Path
import numpy as np,ezc3d
from scipy.signal import butter,sosfiltfilt
from scipy.interpolate import CubicSpline
from trace_error_audit import ObservedPitch
BASE=Path(__file__).resolve().parent/'artifacts/anatomical-shoulder'
OUT=BASE.parent/'shoulder-marker-stability'

def frame(long_axis,cross_axis):
    z=long_axis/np.linalg.norm(long_axis,axis=1)[:,None]
    x=cross_axis-(cross_axis*z).sum(1)[:,None]*z
    norm=np.linalg.norm(x,axis=1)
    if norm.min()<1e-5:raise ValueError('Degenerate marker frame')
    x/=norm[:,None];y=np.cross(z,x)
    return np.stack([x,y,z],axis=2)

def main():
    OUT.mkdir(parents=True,exist_ok=True);audit=[]
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        data=np.load(BASE/f'{pitch}_input.npz');t=data['time'];obs=ObservedPitch(pitch)
        c=ezc3d.c3d(obs.ledger['source_paths']['c3d']);labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
        rate=float(c['parameters']['POINT']['RATE']['value'][0]);unit=c['parameters']['POINT']['UNITS']['value'][0].strip().lower();scale={'m':1.,'mm':.001,'cm':.01}[unit]
        raw={}
        for name in ['RUPA','RFRM','RWRA','RWRB','RFIN']:
            points=c['data']['points'][:,labels.index(name),:].T
            if np.any(points[:,3]<0) or not np.isfinite(points).all():raise ValueError('Missing/invalid marker')
            filtered=sosfiltfilt(butter(4,20,fs=rate,output='sos'),points[:,:3]*scale,axis=0)
            raw[name]=CubicSpline(np.arange(len(filtered))/rate,filtered)(t)
        shoulder,elbow,wrist=np.moveaxis(data['arm'],1,0);mid=(raw['RWRA']+raw['RWRB'])/2
        frames=np.stack([frame(shoulder-elbow,raw['RUPA']-shoulder),
            frame(elbow-wrist,raw['RFRM']-wrist),frame(raw['RFIN']-mid,raw['RWRA']-raw['RWRB'])],axis=1)
        np.savez_compressed(OUT/f'{pitch}_directions.npz',frames=frames,time=t)
        audit.append(dict(pitch=pitch,n=len(t),determinant_min=float(np.linalg.det(frames).min()),
            source_sha256=obs.ledger['source_sha256']['c3d'],filter_hz=20,missing_markers=0))
    (OUT/'marker_manifest.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
