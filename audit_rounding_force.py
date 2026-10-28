"""Independent force check of bounded COM dequantization; not speed validation."""
import json
from pathlib import Path
import numpy as np,pandas as pd,ezc3d
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt
from full_body_comparison import DATA
from pilot import regular_time,vector
from quantization_reconstruction import smooth_rounding_bins

def run():
    meta=pd.read_csv(DATA/'metadata.csv').set_index('session_pitch');allg=pd.read_csv(DATA/'full_sig/landmarks.csv');rows=[]
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        g=allg[allg.session_pitch.eq(pitch)].reset_index(drop=True);t=regular_time(g);mask=(t>=g.fp_10_time.iloc[0])&(t<=g.BR_time.iloc[0]);p=vector(g,'centerofmass');smooth,info=smooth_rounding_bins(p)
        if not info['converged']:raise RuntimeError('Not converged')
        c=ezc3d.c3d(str(list((DATA/'c3d').rglob(meta.loc[pitch,'filename_new']))[0]),extract_forceplat_data=True);force=sum(x['force'].T for x in c['data']['platform']);force=sosfiltfilt(butter(4,40,fs=1080,output='sos'),force,axis=0);f=CubicSpline(np.arange(len(force))/1080,force)(t[mask]);errors={}
        for name,points in [('rounded',p),('bounded',smooth)]:
            required=meta.loc[pitch,'session_mass_kg']*(CubicSpline(t,points)(t[mask],2)-[0,0,-9.81]);errors[name]=float(np.sqrt(np.mean(np.sum((required-f)**2,axis=1))))
        rows.append(dict(pitch=pitch,force_residual_rms_n=errors,optimizer=info))
    out=Path('artifacts/baseline-calibration/com_rounding_audit.json');out.write_text(json.dumps(dict(rows=rows,scope='Observed derived COM versus raw global GRF. Not model output or radar speed validation.'),indent=2));print(rows)
if __name__=='__main__':run()
