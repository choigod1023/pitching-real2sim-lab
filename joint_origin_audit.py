"""Source-graph and point-definition audit, not measured anatomical accuracy."""
import hashlib
import json
from pathlib import Path
import ezc3d
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_transforms(t):
    t = np.asarray(t, float)
    if t.ndim != 3 or t.shape[1:] != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Finite Nx4x4 transforms required')
    if not np.allclose(t[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError('Invalid homogeneous row')
    r = t[:, :3, :3]
    if np.max(np.abs(np.swapaxes(r, 1, 2)@r-np.eye(3))) > 16*np.finfo(np.float32).eps or np.any(np.linalg.det(r) <= 0):
        raise ValueError('Rotation exceeds float32 source precision')
    return t


def relative_origin(parent, child):
    parent, child = map(validate_transforms, (parent, child))
    if parent.shape != child.shape:
        raise ValueError('Parent and child samples differ')
    return np.einsum('nji,nj->ni', parent[:, :3, :3], child[:, :3, 3]-parent[:, :3, 3])


def fixed_local_prediction(parent, local_training_points):
    parent = validate_transforms(parent)
    p = np.asarray(local_training_points, float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Finite training-only local points required')
    center = p.mean(axis=0)
    return parent[:, :3, 3]+np.einsum('nij,j->ni', parent[:, :3, :3], center), center


def statistics(points, scale_mm):
    p = np.asarray(points)
    return {'mean_local_m': (p.mean(axis=0)*scale_mm/1000).tolist(),
            'axis_peak_to_peak_mm': (np.ptp(p, axis=0)*scale_mm).tolist(),
            'max_distance_from_mean_mm': float(np.linalg.norm(p-p.mean(axis=0), axis=1).max()*scale_mm)}


def run():
    observation = LAB/'hand-ball-link/artifacts/hand-ball/observation.json'
    o = json.loads(observation.read_text(encoding='utf-8'))
    for entry in o['source_sha256'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed source')
    body_path = Path(o['source_sha256']['body']['path'])
    alignment_path = Path(o['source_sha256']['alignment']['path'])
    c = ezc3d.c3d(str(body_path)); a = json.loads(alignment_path.read_text(encoding='utf-8'))
    labels = c['parameters']['ROTATION']['LABELS']['value']
    transforms = {name: np.moveaxis(c['data']['rotations'][:, :, labels.index(name+'_4X4'), :], -1, 0)
                  for name in ['torso', 'l_uarm', 'l_larm', 'l_hand']}
    scale_mm = float(a['scale'])*1000
    pairs = {}
    for parent, child in [('l_uarm', 'l_larm'), ('l_larm', 'l_hand'), ('torso', 'l_uarm')]:
        local = relative_origin(transforms[parent], transforms[child])
        window = np.arange(940, 980)+1
        pairs[parent+'->'+child] = {'all_frames': statistics(local, scale_mm),
                                   'video940_979': statistics(local[window], scale_mm)}
    local = relative_origin(transforms['torso'], transforms['l_uarm'])
    train = np.array([940, 950, 960])+1
    evaluate = np.array([945, 955, 965, 970, 977, 979])+1
    predicted, center = fixed_local_prediction(transforms['torso'][evaluate], local[train])
    discrepancy = np.linalg.norm(predicted-transforms['l_uarm'][evaluate, :3, 3], axis=1)*scale_mm
    meta = {}
    for group, values in c['parameters'].items():
        for key, val in values.items():
            if key in ['FREE_ARMS', 'THEIA3D_VERSION', 'MODEL_VERSION', 'FILT_FREQ'] and isinstance(val, dict):
                value = val.get('value'); meta[group+':'+key] = value.tolist() if isinstance(value, np.ndarray) else value
    source_dir = LAB/'openbiomechanics/computer_vision/4d_reconstruction'
    table_path = source_dir/'release_keypoints.csv'; table = pd.read_csv(table_path)
    result = {'protocol': 'JOINT_ORIGIN_PROTOCOL.md', 'reference': 'OBP_movement_BaseballThrow_001',
              'source_sha256': {str(p): sha(p) for p in [observation, body_path, alignment_path, table_path]},
              'file_metadata': meta, 'source_body_units': c['parameters']['POINT']['UNITS']['value'],
              'matrix_frame_count': len(local), 'source_to_rig_scale': float(a['scale']),
              'internal_parent_child_geometry': pairs,
              'fixed_torso_attachment_sensitivity': {'training_video_frames': (train-1).tolist(), 'evaluation_video_frames': (evaluate-1).tolist(),
                    'fitted_torso_local_point_raw_mm': center.tolist(), 'discrepancy_from_source_mm': discrepancy.tolist(),
                    'rmse_mm': float(np.sqrt(np.mean(discrepancy**2))), 'maximum_mm': float(discrepancy.max()),
                    'scope': 'Source-estimate disagreement after imposing fixed local attachment; not actual shoulder error or a new correction.'},
              'original_reference_table': {'columns': table.columns.tolist(), 'rows': len(table),
                    'video_frames': [int(table.video_frame.min()), int(table.video_frame.max())],
                    'segments': sorted(table.segment.unique().tolist()),
                    'original_2d_labels_and_anatomical_protocol_available': False,
                    'fit_code_available': a.get('fit_code_available')},
              'current_documented_origins': {'upper_arm': 'shoulder', 'lower_arm': 'elbow', 'hand': 'wrist'},
              'exact_clinical_point_and_2023_model_equivalence_verified': False,
              'point_name_equivalence_to_original_COCO_reference_verified': False,
              'new_shoulder_correction_applied': False, 'actual_speed_gain_kmh': None,
              'actual_force_or_work_capacity_measured': False, 'scientific_stage_passed': False}
    (ROOT/'joint_origin_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'metadata': meta, 'pairs': pairs, 'fixed_attachment': result['fixed_torso_attachment_sensitivity'],
                      'reference_table': result['original_reference_table']}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
