"""Audit raw force-platform frames against supplied 1080Hz force series."""
import json,itertools
from pathlib import Path
import numpy as np,pandas as pd,ezc3d
from scipy.signal import butter,sosfiltfilt
from full_body_comparison import DATA
OUT=Path(__file__).resolve().parent/'artifacts/baseline-calibration'

def audit():
    meta=pd.read_csv(DATA/'metadata.csv').set_index('session_pitch');csv=pd.read_csv(DATA/'full_sig/force_plate.csv');runs=[]
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        path=list((DATA/'c3d').rglob(meta.loc[pitch,'filename_new']))[0];c=ezc3d.c3d(str(path),extract_forceplat_data=True);rate=float(c['parameters']['ANALOG']['RATE']['value'][0])
        f=csv[csv.session_pitch.eq(pitch)].reset_index(drop=True);t=np.arange(len(f))/rate
        if np.max(abs(t-f.time.values))>.000051:raise ValueError('Analog timing mismatch')
        raw=np.stack([p['force'].T for p in c['data']['platform']]);target=np.stack([f[[s+'_force_'+a for a in 'xyz']].values for s in ['rear','lead']])
        results=[]
        for processing in ['raw','20Hz','40Hz']:
            p=raw if processing=='raw' else sosfiltfilt(butter(4,float(processing[:-2]),fs=rate,output='sos'),raw,axis=1)
            for assign in itertools.product([0,1,2],repeat=len(p)):
                if not 1 in assign or not 2 in assign:continue
                for sign in [1,-1]:
                    pred=np.stack([p[np.array(assign)==i].sum(0)*sign for i in [1,2]])
                    results.append(dict(rmse_n=float(np.sqrt(np.mean((pred-target)**2))),assignment=list(assign),sign=sign,processing=processing))
        results.sort(key=lambda x:x['rmse_n']);platforms=[]
        for p in c['data']['platform']:
            active=np.linalg.norm(p['force'],axis=0)>20;cop=p['center_of_pressure'];valid=np.isfinite(cop).all(0)&np.isfinite(p['Tz']).all(0)
            platforms.append(dict(force_unit=p['unit_force'],moment_unit=p['unit_moment'],position_unit=p['unit_position'],active_samples=int(active.sum()),active_valid_samples=int((active&valid).sum()),fp_to_br_active_samples=int((active&(t>=float(f.fp_10_time.iloc[0]))&(t<=float(f.BR_time.iloc[0]))).sum()),corners_center_m=p['corners'].mean(1).tolist()))
        maps=[]
        for side,plate in [('rear',1),('lead',2)]:
            x=raw[plate];y=target[0 if side=='rear' else 1];mapping=np.linalg.lstsq(x,y,rcond=None)[0]
            maps.append(dict(side=side,plate_index=plate,matrix=mapping.tolist(),rmse_n=float(np.sqrt(np.mean((x@mapping-y)**2))),determinant=float(np.linalg.det(mapping)),interpretation='Diagnostic unconstrained fit only, not an authorized physical coordinate conversion'))
        runs.append(dict(axis_diagnostics=maps,pitch=pitch,analog_rate_hz=rate,samples=len(f),point_rate_hz=float(c['parameters']['POINT']['RATE']['value'][0]),best=results[:5],platforms=platforms))
    report=dict(runs=runs,assignment_codes={'0':'unassigned','1':'rear','2':'lead'},ground_contact_validated=False,
                scope='Force-channel identity audit. 40Hz is documented in the current OBP datasheet; raw/20Hz comparisons are preserved. Raw moment/COP validity does not establish contact-model adequacy.')
    (OUT/'force_platform_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(runs,indent=2));return report
if __name__=='__main__':audit()
