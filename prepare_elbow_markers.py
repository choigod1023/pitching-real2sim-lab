"""Export filtered epicondyle marker proxies in the prior fit's timestamps."""
import numpy as np,ezc3d
from pathlib import Path
from scipy.signal import butter,sosfiltfilt
from scipy.interpolate import CubicSpline
from trace_error_audit import ObservedPitch
ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'artifacts/wrist-muscle'
OUT=ROOT/'artifacts/elbow-registration-audit'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for i in range(1,6):
        pitch=f'3034_{i}';z=np.load(INPUT/f'{pitch}_ik.npz');c=ezc3d.c3d(ObservedPitch(pitch).ledger['source_paths']['c3d'])
        labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']];fs=float(c['parameters']['POINT']['RATE']['value'][0]);unit=c['parameters']['POINT']['UNITS']['value'][0].strip().lower();scale={'mm':.001,'m':1.,'cm':.01}[unit];points=[]
        for name in ['RELB','RMELB']:
            raw=c['data']['points'][:,labels.index(name),:].T
            if np.any(raw[:,3]<0) or not np.isfinite(raw).all():raise ValueError(f'{pitch} {name}: invalid marker')
            filtered=sosfiltfilt(butter(4,20,fs=fs,output='sos'),raw[:,:3]*scale,axis=0)
            points.append(CubicSpline(np.arange(len(raw))/fs,filtered)(z['time']))
        np.savez_compressed(OUT/f'{pitch}_epicondyles.npz',lateral=points[0],medial=points[1],time=z['time'])

if __name__=='__main__':main()
