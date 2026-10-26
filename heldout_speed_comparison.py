"""Frozen temporal comparison. Fit functions accept training observations only."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def fit_models(times, ball, origin, rotation):
    times, ball, origin, rotation = map(np.asarray, (times, ball, origin, rotation))
    if len(times) < 2 or not all(np.isfinite(x).all() for x in (times, ball, origin, rotation)):
        raise ValueError('Need finite training observations')
    local = np.einsum('nji,nj->ni', rotation, ball-origin)
    t0 = float(times[0])
    design = np.column_stack((np.ones(len(times)), times-t0))
    return {'t0': t0, 'A': local.mean(axis=0),
            'B': np.linalg.lstsq(design, local, rcond=None)[0],
            'C': np.linalg.lstsq(design, ball, rcond=None)[0]}


def predict(models, times, origin, rotation):
    design = np.column_stack((np.ones(len(times)), np.asarray(times)-models['t0']))
    return {'A': origin + np.einsum('nij,j->ni', rotation, models['A']),
            'B': origin + np.einsum('nij,nj->ni', rotation, design @ models['B']),
            'C': design @ models['C']}


def evaluate(predicted, observed, dt, radar_kmh):
    v = np.diff(predicted, axis=0)/dt
    truth = np.diff(observed, axis=0)/dt
    speed = np.linalg.norm(v, axis=1)*3.6
    measured = np.linalg.norm(truth, axis=1)*3.6
    return {'predicted_interval_speed_kmh': speed.tolist(),
            'candidate_interval_speed_kmh': measured.tolist(),
            'mean_predicted_speed_kmh': float(speed.mean()),
            'mean_candidate_speed_kmh': float(measured.mean()),
            'mean_absolute_speed_discrepancy_kmh': float(np.abs(speed-measured).mean()),
            'mean_velocity_vector_discrepancy_kmh': float((np.linalg.norm(v-truth,axis=1)*3.6).mean()),
            'position_rmse_mm': float(np.sqrt(np.mean(np.sum((predicted-observed)**2,axis=1)))*1000),
            'mean_absolute_radar_scalar_discrepancy_kmh': float(np.abs(speed-radar_kmh).mean()),
            'predicted_positions_m': predicted.tolist()}


def run():
    import ezc3d
    source = ROOT.parent/'hand-ball-link/artifacts/hand-ball/observation.json'
    report = json.loads(source.read_text())
    body = Path(report['source_sha256']['body']['path'])
    alignment_path = Path(report['source_sha256']['alignment']['path'])
    alignment = json.loads(alignment_path.read_text())
    c3d = ezc3d.c3d(str(body))
    labels = c3d['parameters']['ROTATION']['LABELS']['value']
    rotation = c3d['data']['rotations'][:3,:3,labels.index('l_hand_4X4'),:]
    q = np.asarray(alignment['rotation'])
    frames = np.arange(973,980)
    times = (frames-973)/report['rate_hz']
    radar = 62.7*1.609344
    results = []
    for row in report['conditions']:
        ball = np.array([p['ball_rig_m'] for p in row['positions']])
        origin = np.array([p['hand_origin_rig_m'] for p in row['positions']])
        rotations = np.einsum('ij,jkn->nik', q, rotation[:,:,frames+row['c3d_frame_offset']])
        fitted = fit_models(times[:4], ball[:4], origin[:4], rotations[:4])
        future = predict(fitted, times[4:], origin[4:], rotations[4:])
        metrics = {name:evaluate(value, ball[4:], 1/report['rate_hz'], radar) for name,value in future.items()}
        results.append({'threshold':row['threshold'], 'offset':row['c3d_frame_offset'],
                        'fit':{k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in fitted.items()},
                        'metrics':metrics,
                        'A_minus_B_speed_discrepancy_kmh':metrics['A']['mean_absolute_speed_discrepancy_kmh']-metrics['B']['mean_absolute_speed_discrepancy_kmh'],
                        'C_minus_B_speed_discrepancy_kmh':metrics['C']['mean_absolute_speed_discrepancy_kmh']-metrics['B']['mean_absolute_speed_discrepancy_kmh']})
    output = {'trial':report['trial'], 'training_frames':frames[:4].tolist(), 'evaluation_frames':frames[4:].tolist(),
              'primary_condition':{'threshold':80,'offset':1}, 'radar_scalar_kmh':radar,
              'scope':'Previously inspected development trial; training excludes future ball positions. B conditions on observed future hand transforms. Optical candidates and provisional calibration are not ground truth. No physics/athlete improvement claim; not 3034_1.',
              'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,body,alignment_path,ROOT/'COMPARISON_PROTOCOL.md']},
              'conditions':results}
    out = ROOT/'artifacts/speed-comparison'; out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for row in results:
        print(row['threshold'],row['offset'],json.dumps({k:round(v['mean_absolute_speed_discrepancy_kmh'],6) for k,v in row['metrics'].items()}))
    return output


if __name__ == '__main__':
    run()
