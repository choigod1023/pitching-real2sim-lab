"""Keep observed ball outcomes separate from kinematic candidate proxies."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def attach_observed_speed(candidates, metadata):
    keys = ['user', 'session_pitch']
    if metadata.duplicated(keys).any():
        raise ValueError('Ambiguous metadata pitch identity')
    merged = candidates.merge(metadata[keys + ['pitch_speed_mph']], on=keys,
                              how='left', validate='many_to_one', indicator=True)
    if (merged['_merge'] != 'both').any():
        raise ValueError('Candidate has no measured pitch reference')
    speed = pd.to_numeric(merged.pop('pitch_speed_mph'), errors='raise')
    if not np.isfinite(speed).all() or (speed <= 0).any():
        raise ValueError('Measured ball speed must be finite and positive')
    merged['observed_original_ball_speed_kmh'] = speed * 1.609344
    # Deliberately never add a hand-speed delta to observed ball speed.
    return merged.drop(columns='_merge')


def evidence_contract():
    return {
        'stage': 'kinematic_exploration',
        'candidate_metric': 'hand_joint_proxy_speed_change_m_s',
        'observed_ball_speed_scope': 'original recorded pitch only',
        'predicted_candidate_ball_speed_kmh': None,
        'predicted_candidate_target_error_cm': None,
        'dynamic_feasibility': 'not_evaluated',
        'athlete_comfort': 'not_measured',
        'athlete_training_benefit': 'not_tested',
        'coaching_recommendation_ready': False,
    }


def audit(data):
    meta = pd.read_csv(data / 'metadata.csv')
    files = ['landmarks', 'joint_angles', 'joint_velos', 'forces_moments',
             'force_plate', 'energy_flow']
    signals = {}
    for name in files:
        path = data / 'full_sig' / (name + '.csv')
        signals[name] = {'exists': path.is_file(),
                         'columns': list(pd.read_csv(path, nrows=0)) if path.is_file() else []}
    return {
        'pitches': len(meta), 'athletes': int(meta.user.nunique()),
        'sessions': int(meta.session.nunique()),
        'athletes_with_multiple_sessions': int((meta.groupby('user').session.nunique() > 1).sum()),
        'valid_observed_ball_speed_rows': int((np.isfinite(meta.pitch_speed_mph) & (meta.pitch_speed_mph > 0)).sum()),
        'metadata_columns': list(meta), 'signals': signals,
        'evidence': evidence_contract(),
        'scope': 'Local metadata and full_sig file headers; presence is not validation or synchronization quality.'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent.parent / 'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--output', type=Path, default=Path('artifacts/data-audit.json'))
    args = parser.parse_args()
    result = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['signals', 'metadata_columns']}, ensure_ascii=False, indent=2))
