"""Prepare five reused OBP pitches; speed labels are evaluation-only."""
from pathlib import Path
import json
import numpy as np,ezc3d
from scipy.signal import butter,sosfiltfilt
from scipy.interpolate import CubicSpline
from trace_error_audit import ObservedPitch
OUT=Path(__file__).resolve().parent/'artifacts/anatomical-shoulder'

def main():
    meta={}
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        o=ObservedPitch(pitch,reconstruct_rounding=True)
        c=ezc3d.c3d(o.ledger['source_paths']['c3d']);labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
        rate=float(c['parameters']['POINT']['RATE']['value'][0]);unit=c['parameters']['POINT']['UNITS']['value'][0].strip().lower();factor={'m':1.,'mm':.001,'cm':.01}[unit]
        times=np.linspace(o.a,o.b,round((o.b-o.a)*360)+1);markers=[]
        for name in ['CLAV','C7','STRN']:
            raw=c['data']['points'][:,labels.index(name),:].T
            if np.any(raw[:,3]<0) or not np.isfinite(raw).all():raise ValueError('Invalid marker')
            x=sosfiltfilt(butter(4,20,fs=rate,output='sos'),raw[:,:3]*factor,axis=0)
            markers.append(CubicSpline(np.arange(len(x))/rate,x)(times))
        np.savez_compressed(OUT/f'{pitch}_input.npz',time=times,torso=np.stack(markers,axis=1),arm=o.points(times))
        meta[pitch]=dict(measured_speed_kmh=o.ledger['recorded_speed_kmh'],n_samples=len(times),source_sha256=o.ledger['source_sha256'])
    (OUT/'input_manifest.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
