"""Exact pitch evidence and algebraic bounds; never fit grip to radar."""
import hashlib
import json
import sys
from pathlib import Path
import ezc3d
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from grip_identifiability import velocity_map

data = ROOT.parent / 'openbiomechanics/baseball_pitching/data'
meta = pd.read_csv(data/'metadata.csv').set_index('session_pitch').loc['3034_1']
c3d_path = next((data/'c3d').rglob(meta.filename_new))
c = ezc3d.c3d(str(c3d_path))
labels = [x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
g = pd.read_csv(data/'full_sig/landmarks.csv')
g = g[g.session_pitch.eq('3034_1')].sort_values('time')
trace = ROOT.parent/'speed-journey/artifacts/speed-video/rollout.npz'
reference = ROOT.parent/'grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz'
with np.load(trace) as f, np.load(reference) as ref:
    vw, a, rotation, omega = velocity_map(ref['lengths'], float(f['dt']), f['qpos'][-1], f['qvel'][-1])
    vb = vw + a@np.array([.15, 0, 0])
    sim_speed = float(np.linalg.norm(vb)*3.6)
    np.testing.assert_allclose(sim_speed, f['speed_kmh'][-1], atol=1e-10)
target = float(meta.pitch_speed_mph)*1.609344
gap_mps = (target-sim_speed)/3.6
angular_norm = float(np.linalg.norm(a,2))
rng=np.random.default_rng(3034)
offsets=rng.normal(size=(10000,3))
offsets=offsets/np.linalg.norm(offsets,axis=1)[:,None]*(rng.random(10000)**(1/3)*.01)[:,None]
sample_speeds=np.linalg.norm(vb+offsets@a.T,axis=1)*3.6
assert np.all(sample_speeds <= sim_speed+3.6*angular_norm*.01+1e-10)
result = {
    'pitch':'3034_1','anonymous_user_id':int(meta.user),'playing_level':str(meta.playing_level),
    'record_speed_mph':float(meta.pitch_speed_mph),'record_speed_kmh':target,
    'record_measurement_time_location':'unconfirmed','BR_time_s':float(g.BR_time.iloc[0]),
    'footplant_time_s':float(g.fp_10_time.iloc[0]),
    'c3d_path':str(c3d_path),'c3d_sha256':hashlib.sha256(c3d_path.read_bytes()).hexdigest(),
    'c3d_rate_hz':float(c['parameters']['POINT']['RATE']['value'][0]),
    'c3d_event_used':c['parameters'].get('EVENT',{}).get('USED',{}).get('value',[]).tolist() if hasattr(c['parameters'].get('EVENT',{}).get('USED',{}).get('value',[]),'tolist') else [],
    'explicit_ball_labels':[x for x in labels if 'ball' in x.lower()],
    'required_hand_marker_labels_present':all(x in labels for x in ['RFIN','RWRA','RWRB','RFRM']),
    'all_point_labels':labels,
    'sim_ball_velocity_mps':vb.tolist(),'sim_wrist_velocity_mps':vw.tolist(),
    'sim_ball_speed_kmh':sim_speed,'sim_wrist_speed_kmh':float(np.linalg.norm(vw)*3.6),
    'sim_hand_angular_speed_rad_s':float(np.linalg.norm(omega)),
    'conditional_missing_velocity_norm_lower_bound_mps':gap_mps,
    'conditional_grip_displacement_necessary_lower_bound_mm':gap_mps/angular_norm*1000,
    'one_cm_grip_ball_speed_upper_bound_kmh':sim_speed+3.6*angular_norm*.01,
    'bound_random_sanity_checks':10000,
    'scope':'Bounds conditional on same true observation time and exact modeled wrist/rotation. Not measured finger speed, required actual grip correction, physiological limit, or proof of cause. Lower bounds are necessary, not sufficient.',
    'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
    'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
}
out=ROOT/'artifacts/gap-observation/exact_pitch.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['all_point_labels','c3d_path']},indent=2))
