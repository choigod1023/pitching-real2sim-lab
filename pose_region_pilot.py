"""Single-frame inverse-kinematics diagnostic; not a pitching intervention."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from prime17_hand_alignment import Source, project, box_residual, sha

ROOT = Path(__file__).resolve().parent


def reconstruct(shoulder, elbow, wrist, rows, cameras):
    s, e, w = map(lambda p: np.asarray(p, float), (shoulder, elbow, wrist))
    if any(r['camera'] == 17 for r in rows):
        raise ValueError('Held-out camera17 cannot fit the pilot')
    initial = np.r_[e, w]
    lengths = np.array([np.linalg.norm(e-s), np.linalg.norm(w-e)])
    if not np.isfinite(initial).all() or min(lengths) < .05:
        raise ValueError('Invalid arm geometry')
    def residual(x, group):
        points = {'l_larm': x[:3], 'l_hand': x[3:]}
        return np.array([box_residual(project(points[r['segment']], cameras[r['camera']])[0], r['box']) for r in group])
    def loss(x):
        pixels = residual(x, rows)
        return float(np.mean(np.sum(pixels**2, axis=1))+.001*np.sum(((x-initial)/.01)**2))
    def constraints(x):
        return np.array([np.dot(x[:3]-s, x[:3]-s)-lengths[0]**2,
                         np.dot(x[3:]-x[:3], x[3:]-x[:3])-lengths[1]**2])
    result = minimize(loss, initial, method='SLSQP', bounds=list(zip(initial-.2, initial+.2)),
                      constraints={'type': 'eq', 'fun': constraints}, options={'maxiter': 1000, 'ftol': 1e-10})
    if not result.success or np.max(np.abs(constraints(result.x))) > 1e-8:
        raise RuntimeError('Constrained reconstruction failed: '+result.message)
    return result.x, lengths, {'solver_success': bool(result.success), 'iterations': result.nit,
                              'length_squared_residual_m2': constraints(result.x).tolist()}


def run():
    source = Source();frame = 950
    rows = json.loads((ROOT/'counterview_regions.json').read_text(encoding='utf-8'))['observations']
    held = [r for r in json.loads((ROOT/'long_hand_regions.json').read_text(encoding='utf-8'))['observations'] if r['frame'] == frame]
    s, e, w = [source.origins[name](frame+1) for name in ['l_uarm', 'l_larm', 'l_hand']]
    fitted, lengths, info = reconstruct(s, e, w, rows, source.cameras)
    report = {'frame': frame, 'clock_offset': 1., 'fitted_cameras': [20, 21], 'heldout_camera': 17,
              'source_sha256': {str(ROOT/'POSE_RECONSTRUCTION_PILOT_PROTOCOL.md'): sha(ROOT/'POSE_RECONSTRUCTION_PILOT_PROTOCOL.md'),
                               str(ROOT/'counterview_regions.json'): sha(ROOT/'counterview_regions.json')},
              'shoulder_m': s.tolist(), 'original_elbow_m': e.tolist(), 'original_wrist_m': w.tolist(),
              'fitted_elbow_m': fitted[:3].tolist(), 'fitted_wrist_m': fitted[3:].tolist(), 'lengths_m': lengths.tolist(),
              'joint_shift_mm': {'elbow': float(np.linalg.norm(fitted[:3]-e)*1000), 'wrist': float(np.linalg.norm(fitted[3:]-w)*1000)},
              'solver': info, 'metrics': {}, 'scientific_stage_passed': False, 'baseline_model_replaced': False,
              'actual_performance_gain_kmh': None, 'force_work_verified': False, 'wrist_orientation_verified': False}
    for group, observations in [('fit', rows), ('heldout', held)]:
        metrics = {}
        for name, x in [('before', np.r_[e, w]), ('after', fitted)]:
            distances = []
            for r in observations:
                point = x[:3] if r['segment'] == 'l_larm' else x[3:]
                uv = project(point, source.cameras[r['camera']])[0]
                distances.append(float(np.linalg.norm(box_residual(uv, r['box']))))
            metrics[name] = {'count': len(distances), 'mean_outside_px': float(np.mean(distances)), 'per_observation_px': distances}
        report['metrics'][group] = metrics
    (ROOT/'pose_region_pilot_results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    run()
