"""Frozen raw-image regions score train-only clock/translation calibration.

Surface regions are not anatomical centres. Camera calibration is shared,
and the annotations are not an independent measurement accuracy standard.
"""
import hashlib
import json
from pathlib import Path
import ezc3d
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, RotationSpline
from hand_speed_derivative import numerical_rotations, metrics_for_state
from physical_release_comparison import launch_states
from covariant_release import sha
from multiview_release_validation import camera_models, project

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = ROOT/'artifacts/prime17-hand'
OFFSETS = np.arange(-2., 4.00001, .25)


def box_residual(point, box, margin=0.):
    p, b = np.asarray(point, float), np.asarray(box, float)
    if p.shape != (2,) or b.shape != (4,) or not np.isfinite(np.r_[p, b, margin]).all():
        raise ValueError('Finite2D point and box required')
    lower, upper = b[:2]-margin, b[2:]+margin
    if np.any(lower > upper):
        raise ValueError('Reversed box')
    return p-np.clip(p, lower, upper)


class Source:
    def __init__(self):
        self.observation_path = LAB/'hand-ball-link/artifacts/hand-ball/observation.json'
        self.observation = json.loads(self.observation_path.read_text(encoding='utf-8'))
        for key, value in self.observation['source_sha256'].items():
            if sha(value['path']) != value['sha256']:
                raise ValueError('Changed source '+key)
        self.alignment_path = Path(self.observation['source_sha256']['alignment']['path'])
        a = json.loads(self.alignment_path.read_text(encoding='utf-8'))
        c = ezc3d.c3d(self.observation['source_sha256']['body']['path'])
        labels = c['parameters']['ROTATION']['LABELS']['value']
        q, t = np.array(a['rotation']), np.array(a['translation'])
        n = c['data']['rotations'].shape[-1]; indices = np.arange(n)
        self.origins = {}
        for segment in ['l_hand', 'l_larm', 'l_uarm']:
            raw = c['data']['rotations'][:, :, labels.index(segment+'_4X4'), :]
            self.origins[segment] = CubicSpline(indices, a['scale']*(raw[:3, 3, :].T@q.T)+t)
        raw = c['data']['rotations'][:3, :3, labels.index('l_hand_4X4'), :]
        proper, _ = numerical_rotations(np.einsum('ij,jkn->nik', q, raw))
        self.hand = RotationSpline(indices, Rotation.from_matrix(proper))
        self.cameras = camera_models()
        directory = LAB/'openbiomechanics/computer_vision/calibration/results'
        cal = json.loads((directory/'calibration_results.json').read_text())['calibrations']['optitrack_17']
        rig = json.loads((directory/'optitrack_rig_provisional.json').read_text())['camera_poses']['17']
        if np.any(cal['distortion']):
            raise ValueError('Unsupported nonzero distortion')
        self.cameras[17] = (np.array([[cal['fx_px'], 0, cal['cx_px']], [0, cal['fy_px'], cal['cy_px']], [0, 0, 1.]]),
                            np.array(rig['reference_to_camera_rotation_matrix']), np.array(rig['reference_to_camera_translation_mm'])/1000.)
        self.gravity = q@np.array([0., 0., -9.81])

    def point(self, row, offset, delta):
        return self.origins[row['segment']](row['frame']+offset)+np.asarray(delta)

    def residuals(self, rows, offset, delta, margin=0.):
        return np.array([box_residual(project(self.point(r, offset, delta), self.cameras[r['camera']])[0], r['box'], margin) for r in rows])

    def summary(self, rows, offset, delta, margin=0.):
        d = np.linalg.norm(self.residuals(rows, offset, delta, margin), axis=1)
        per = {}
        for camera in sorted(set(r['camera'] for r in rows)):
            x = d[[r['camera'] == camera for r in rows]]
            per[str(camera)] = {'count': len(x), 'mean_outside_px': float(x.mean()), 'max_outside_px': float(x.max()), 'inside': int((x <= 1e-9).sum())}
        return {'count': len(d), 'inside': int((d <= 1e-9).sum()), 'mean_outside_px': float(d.mean()), 'max_outside_px': float(d.max()),
                'region_not_point_accuracy': True, 'per_camera': per}

    def ball_metric(self, offset, delta):
        condition = next(r for r in self.observation['conditions'] if r['threshold'] == 80 and r['c3d_frame_offset'] == 1)
        ball = np.array([r['ball_rig_m'] for r in condition['positions']])
        frames = np.arange(973, 977); rate = float(self.observation['rate_hz'])
        origin = self.origins['l_hand'](frames+offset)+delta
        rotation = self.hand(frames+offset).as_matrix()
        if offset == 1. and np.linalg.norm(delta) == 0:
            saved = np.array([r['hand_origin_rig_m'] for r in condition['positions']])[:4]
            np.testing.assert_allclose(origin, saved, atol=1e-9, rtol=0)
        p, v = launch_states((frames-973)/rate, ball[:4], origin, rotation)['B']
        return metrics_for_state(p, v, self.gravity, ball[4:], rate, 62.7*1.609344)


def rows_for_analysis():
    new = json.loads((ROOT/'prime17_regions.json').read_text(encoding='utf-8'))['observations']
    old = json.loads((ROOT/'hand_image_regions.json').read_text(encoding='utf-8'))['regions']
    rows = [dict(r) for r in new]
    for row in old:
        r = dict(row); r['split'] = 'fit' if row['camera'] == 20 and row['frame'] == 974 else 'eval'
        rows.append(r)
    return rows


def fit_grid(source, training, allow_translation):
    cases = []
    for offset in OFFSETS:
        def residual(delta):
            image = source.residuals(training, offset, delta).ravel()/np.sqrt(len(training))
            return np.r_[image, np.sqrt(.001)*delta/.01]
        if allow_translation:
            fitted = least_squares(residual, np.zeros(3), bounds=(-.03, .03), max_nfev=300, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            delta = fitted.x
            if not fitted.success:
                raise RuntimeError('Translation fit did not converge')
        else:
            delta = np.zeros(3)
        loss = float(np.sum(residual(delta)**2))
        cases.append({'offset_frames': float(offset), 'translation_m': delta.tolist(), 'training_objective': loss,
                      'training': source.summary(training, offset, delta)})
    best = min(cases, key=lambda c: (c['training_objective'], abs(c['offset_frames']-1.), c['offset_frames']))
    return best, cases


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((ROOT/'prime17_annotation_freeze.json').read_text(encoding='utf-8'))
    for path, digest in freeze['sha256'].items():
        if sha(path) != digest:
            raise ValueError('Annotation/protocol changed after blind freeze')
    source = Source(); rows = rows_for_analysis()
    train = [r for r in rows if r['split'] == 'fit']; test = [r for r in rows if r['split'] == 'eval']
    if len(train) != 8 or len(test) != 20:
        raise ValueError('Frozen split changed')
    m1, c1 = fit_grid(source, train, False); m2, c2 = fit_grid(source, train, True)
    models = {'M0': {'offset_frames': 1., 'translation_m': [0., 0., 0.]}, 'M1': m1, 'M2': m2}
    for model in models.values():
        offset, delta = model['offset_frames'], np.array(model['translation_m'])
        model['training'] = source.summary(train, offset, delta)
        model['evaluation'] = source.summary(test, offset, delta)
        model['annotation_boundary_sensitivity'] = {str(margin): source.summary(test, offset, delta, margin) for margin in (-3, 0, 3)}
        model['B_ball_evaluation'] = source.ball_metric(offset, delta)
    np.testing.assert_allclose(models['M0']['B_ball_evaluation']['mean_absolute_speed_discrepancy_kmh'], 1.3241394341169155, atol=1e-8, rtol=0)
    before, after = models['M0'], models['M2']
    checks = {
        'camera17_eval_improved': after['evaluation']['per_camera']['17']['mean_outside_px'] < before['evaluation']['per_camera']['17']['mean_outside_px'],
        'all_eval_improved': after['evaluation']['mean_outside_px'] < before['evaluation']['mean_outside_px'],
        'existing_views_nonworsening': all(after['evaluation']['per_camera'][str(cam)]['mean_outside_px'] <= before['evaluation']['per_camera'][str(cam)]['mean_outside_px']+1e-9 for cam in (19, 20, 21)),
        'boundary_sensitivity_improved': all(after['annotation_boundary_sensitivity'][str(m)]['mean_outside_px'] < before['annotation_boundary_sensitivity'][str(m)]['mean_outside_px'] for m in (-3, 0, 3)),
        'translation_not_at_boundary': bool(np.max(np.abs(after['translation_m'])) < .03-1e-5),
        'ball_three_metrics_nonworsening': all(after['B_ball_evaluation'][k] <= before['B_ball_evaluation'][k]+1e-9 for k in ['mean_absolute_speed_discrepancy_kmh', 'mean_velocity_vector_discrepancy_kmh', 'position_rmse_mm'])}
    # Follow-up frozen after zero training loss at several clock offsets.
    # Report every compatible grid state; never select one by ball error.
    compatible = []
    for candidate in c1:
        if candidate['training']['inside'] == len(train):
            offset = candidate['offset_frames']
            metric = source.ball_metric(offset, np.zeros(3))
            hand_delta = source.origins['l_hand'](976+offset)-source.origins['l_hand'](977)
            angle = (source.hand(977).inv()*source.hand(976+offset)).magnitude()
            compatible.append({'offset_frames': offset, 'B_ball_evaluation': metric,
                               'wrist_at_boundary_difference_mm': float(np.linalg.norm(hand_delta)*1000),
                               'hand_orientation_at_boundary_difference_deg': float(np.degrees(angle)),
                               'physical_posture_intervention': False})
    sensitivity = {}
    for margin in (-3, 0, 3):
        sensitivity[str(margin)] = [float(offset) for offset in OFFSETS
                                   if source.summary(train, offset, np.zeros(3), margin)['inside'] == len(train)]
    result = {'protocol': 'PRIME17_HAND_PROTOCOL.md', 'reference': 'OBP_movement_BaseballThrow_001', 'models': models,
              'time_only_cases': c1, 'time_translation_cases': c2,
              'training_rows': train, 'evaluation_rows': test,
              'compatible_time_only_states': compatible,
              'boundary_sensitive_compatible_offsets': sensitivity,
              'engineering_checks': checks, 'engineering_candidate_accepted': all(checks.values()),
              'scientific_stage_passed': False, 'anatomical_joint_ground_truth': False, 'actual_athlete_improvement': False,
              'force_work_change': None, 'body_intervention': False,
              'source_sha256': {str(ROOT/'prime17_regions.json'): sha(ROOT/'prime17_regions.json'), str(ROOT/'hand_image_regions.json'): sha(ROOT/'hand_image_regions.json'), **{v['path']: v['sha256'] for v in source.observation['source_sha256'].values()}},
              'annotation_freeze': freeze, 'actual_capture_clock_verified': False,
              'interpretation': 'Training-only time/translation calibration and fit-excluded image regions; shared camera model and approximate single-reader regions are not anatomical accuracy or valid same-budget joint intervention evidence.'}
    for p in [OUT/'results.json', ROOT/'prime17_hand_results.json']:
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'models': {name: {'offset': m['offset_frames'], 'translation_mm': (np.array(m['translation_m'])*1000).tolist(), 'eval': m['evaluation'], 'ball_speed_mae': m['B_ball_evaluation']['mean_absolute_speed_discrepancy_kmh']} for name, m in models.items()}, 'checks': checks}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
