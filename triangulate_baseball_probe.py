"""Exploratory two-frame probe. Not calibrated truth or prediction validation."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('observations_dir', type=Path)
parser.add_argument('calibration_dir', type=Path)
args = parser.parse_args()
OUT = args.observations_dir.resolve()
CAL = args.calibration_dir.resolve()
paths = [OUT/'exploratory_2d_candidates.json', CAL/'calibration_results.json',
         CAL/'optitrack_rig_provisional.json']
data, calibration, rig = [json.loads(p.read_text()) for p in paths]
cameras = (19, 22)
thresholds = (130, 90)
frames = (980, 981)
points = np.array([[[r['x_px'], r['y_px']] for cam, th in zip(cameras, thresholds)
                    for r in data['rows'] if r['camera'] == f'camera{cam}'
                    and r['threshold'] == th and r['frame_zero_based'] == f]
                   for f in frames])
assert points.shape == (2, 2, 2)
K, poses = [], []
for cam in cameras:
    intrinsic = calibration['calibrations'][f'optitrack_{cam}']
    # Released calibration chooses zero distortion for these cameras. Applying
    # an unreported lens coefficient instead would change the model arbitrarily.
    assert np.all(np.array(intrinsic['distortion']) == 0)
    assert calibration['video_info'][f'optitrack_{cam}']['width'] == 1280
    assert calibration['video_info'][f'optitrack_{cam}']['height'] == 720
    K.append(np.array([[intrinsic['fx_px'],0,intrinsic['cx_px']],
                       [0,intrinsic['fy_px'],intrinsic['cy_px']], [0,0,1]]))
    pose = rig['camera_poses'][str(cam)]
    poses.append(np.column_stack((pose['reference_to_camera_rotation_matrix'],
                                  pose['reference_to_camera_translation_mm'])))

def solve(observations):
    equations = []
    for pixel, intr, p in zip(observations, K, poses):
        normalized = np.linalg.solve(intr, np.r_[pixel,1])
        equations.extend([normalized[0]*p[2]-p[0], normalized[1]*p[2]-p[1]])
    _, _, v = np.linalg.svd(equations)
    x = v[-1,:3] / v[-1,3]
    depths, residuals = [], []
    for pixel, intr, p in zip(observations,K,poses):
        xc = p @ np.r_[x,1]
        uv = intr @ xc
        depths.append(float(xc[2]))
        residuals.append(float(np.linalg.norm(uv[:2]/uv[2]-pixel)))
    centers = [-p[:,:3].T@p[:,3] for p in poses]
    rays = [(x-c)/np.linalg.norm(x-c) for c in centers]
    angle = np.degrees(np.arccos(np.clip(np.dot(*rays),-1,1)))
    return dict(position_camera19_mm=x.tolist(), depth_mm=depths,
                positive_depth=bool(min(depths)>0), reprojection_residual_px=residuals,
                ray_angle_deg=float(angle))

def speed(results):
    # Conditional on matching frame indexes being synchronized at 360 Hz.
    return float(np.linalg.norm(np.subtract(results[1]['position_camera19_mm'],
                                           results[0]['position_camera19_mm']))*360*3.6/1000)

base = [solve(p) for p in points]
perturbed = []
for signs in itertools.product((-1,1), repeat=8):
    delta = np.array(signs).reshape(2,2,2)
    values = [solve(p) for p in points+delta]
    perturbed.append(dict(independent_coordinate_offsets_px=list(signs),
                          conditional_speed_kmh=speed(values), frames=values))
# Internal algebra check only: known 3D point reprojected through same cameras.
known = np.array([100., 200., 4000.])
synthetic = []
for intr,p in zip(K,poses):
    uv = intr @ (p @ np.r_[known,1])
    synthetic.append(uv[:2]/uv[2])
assert np.linalg.norm(np.array(solve(synthetic)['position_camera19_mm'])-known)<1e-6
result = dict(scope='Exploratory bright-region centroid triangulation, not true ball center or validated speed.',
              sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              calibration_status=rig['status'], units='mm', coordinates=rig['coordinate_convention'],
              camera_axes='Camera19 optical axes: x image-right, y image-down, z optical-forward; not anatomical axes.',
              distortion='Both source models zero_square with five zero distortion coefficients; inverse distortion is identity.',
              frame_rate_assumption_hz=360, synchronization_validated=False,
              camera_ids=list(cameras), thresholds=list(thresholds), frame_indexes=list(frames),
              image_points_px=points.tolist(), baseline=base, conditional_speed_kmh=speed(base),
              perturbation_definition='All 256 corners of independent +/-1 px errors in 8 image coordinates; finite stress scenarios, not confidence interval or guaranteed continuous-box extrema.',
              perturbation_speed_range_kmh=[min(p['conditional_speed_kmh'] for p in perturbed),
                                            max(p['conditional_speed_kmh'] for p in perturbed)],
              perturbation_all=perturbed, synthetic_algebra_check_passed=True)
threshold_combinations = []
for t19, t22 in itertools.product((100,130,160),(70,90,110)):
    observations = []
    missing = []
    for frame in frames:
        pair = []
        for camera,threshold in zip(cameras,(t19,t22)):
            match = [r for r in data['rows'] if r['camera']==f'camera{camera}'
                     and r['threshold']==threshold and r['frame_zero_based']==frame]
            if len(match)!=1 or 'x_px' not in match[0]:
                missing.append([camera,frame,threshold])
            else:
                pair.append([match[0]['x_px'],match[0]['y_px']])
        observations.append(pair)
    case = dict(thresholds=[t19,t22], missing=missing)
    if not missing:
        values = [solve(p) for p in observations]
        case.update(frames=values,conditional_speed_kmh=speed(values))
    threshold_combinations.append(case)
result['all_threshold_combinations'] = threshold_combinations
(OUT/'triangulation_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['perturbation_all','sources_sha256','all_threshold_combinations']},indent=2))
print('All threshold combinations:', [(r['thresholds'],r.get('conditional_speed_kmh'),r['missing']) for r in threshold_combinations])
