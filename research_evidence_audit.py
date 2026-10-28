"""Audit current source identity, units, frame geometry and constraints."""
import json,hashlib
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import pandas as pd
import ezc3d
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'openbiomechanics/baseball_pitching/data'
OUT=ROOT/'artifacts/research-evidence-audit'

def run():
    OUT.mkdir(parents=True,exist_ok=True);meta=pd.read_csv(DATA/'metadata.csv');poi=pd.read_csv(DATA/'poi/poi_metrics.csv')
    if meta.duplicated('session_pitch').any():raise ValueError('Ambiguous session_pitch in metadata')
    if poi.duplicated('session_pitch').any():raise ValueError('Ambiguous session_pitch in POI')
    manifest=json.loads((ROOT/'artifacts/anatomical-shoulder/input_manifest.json').read_text());rows=[]
    for pitch in [f'3034_{i}' for i in range(1,6)]:
        row=meta.loc[meta.session_pitch.eq(pitch)].iloc[0];p=poi.loc[poi.session_pitch.eq(pitch)].iloc[0]
        files=list((DATA/'c3d').rglob(row.filename_new))
        if len(files)!=1:raise ValueError('Nonunique raw file')
        digest=hashlib.sha256(files[0].read_bytes()).hexdigest()
        if digest!=manifest[pitch]['source_sha256']['c3d']:raise ValueError('Raw C3D differs from fitting input')
        c=ezc3d.c3d(str(files[0]));labels=[x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
        z=np.load(ROOT/f'artifacts/wrist-muscle/{pitch}_ik.npz');t=z['time'];R=z['orientation_targets']
        error=float(np.max(np.abs(np.swapaxes(R,-1,-2)@R-np.eye(3))));det=float(np.min(np.linalg.det(R)))
        if error>1e-8 or det<.999999:raise ValueError('Improper orientation target')
        if not np.all(np.diff(t)>0):raise ValueError('Nonmonotonic time')
        speed=float(row.pitch_speed_mph)*1.609344
        if not np.isclose(speed,manifest[pitch]['measured_speed_kmh'],atol=1e-10):raise ValueError('Recorded speed mismatch')
        if not np.isclose(float(row.pitch_speed_mph),float(p.pitch_speed_mph),atol=1e-10):raise ValueError('Metadata / POI speed mismatch')
        rows.append(dict(pitch=pitch,user=int(row.user),mass_kg=float(row.session_mass_kg),height_m=float(row.session_height_m),filename=row.filename_new,c3d_sha256=digest,observed_speed_kmh=speed,c3d_rate_hz=float(c['parameters']['POINT']['RATE']['value'][0]),c3d_units=c['parameters']['POINT']['UNITS']['value'][0],frames=len(t),dt_min=float(np.min(np.diff(t))),rotation_orthogonality_max=error,rotation_det_min=det,scapular_landmarks_present={n:n in labels for n in ['AA','TS','AI']},scope='Marker/metadata identity is verified; BR detector and radar location/time remain unconfirmed.'))
    model=E.parse(ROOT/'artifacts/shoulder-load-audit/mass_corrected.osim');names=[x.get('name') for x in model.findall('.//BodySet/objects/*')]
    bad=[(m.get('name'),p.findtext('socket_parent_frame')) for m in model.findall('.//ForceSet/objects/*') for p in m.findall('.//PathPointSet/objects/*') if p.findtext('socket_parent_frame','').split('/')[-1] not in names]
    if bad:raise ValueError(str(bad))
    summary={'metadata_pitches':len(meta),'metadata_athletes':int(meta.user.nunique()),'checked_pitches':rows,'body_count':len(names),'muscle_count':len(model.findall('.//ForceSet/objects/*')),'missing_body_references':bad,'units':{'mph_to_kmh':1.609344,'m_s_to_kmh':3.6,'ft_to_m':.3048,'degree_to_rad':float(np.pi/180)},'performance_validation':'No athlete intervention or independently validated joint-change-to-ball-performance prediction.'}
    (OUT/'data_checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
