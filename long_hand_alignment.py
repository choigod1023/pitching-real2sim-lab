"""Additional raw-image trajectory constraints; no speed-driven clock choice."""
import json
from pathlib import Path
import numpy as np
from prime17_hand_alignment import Source, rows_for_analysis, fit_grid, OFFSETS, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/prime17-long'


def combine_observations(long_rows, earlier_rows):
    rows = [dict(r) for r in long_rows]+[dict(r) for r in earlier_rows]
    keys = [(r['camera'], r['frame'], r['segment']) for r in rows]
    if len(keys) != len(set(keys)) or any(r['split'] not in ('fit', 'eval') for r in rows):
        raise ValueError('Duplicate observations or invalid split')
    return rows


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((ROOT/'long_hand_annotation_freeze.json').read_text(encoding='utf-8'))
    for p, digest in freeze['sha256'].items():
        if sha(p) != digest:
            raise ValueError('Frozen raw annotation/source changed')
    long = json.loads((ROOT/'long_hand_regions.json').read_text(encoding='utf-8'))['observations']
    earlier = rows_for_analysis(); rows = combine_observations(long, earlier)
    train = [r for r in rows if r['split'] == 'fit']; test = [r for r in rows if r['split'] == 'eval']
    new_train = [r for r in long if r['split'] == 'fit']; new_test = [r for r in long if r['split'] == 'eval']
    if (len(train), len(test), len(new_train), len(new_test)) != (13, 27, 5, 7):
        raise ValueError('Frozen split changed')
    source = Source(); m1, c1 = fit_grid(source, train, False); m2, c2 = fit_grid(source, train, True)
    models = {'M0': {'offset_frames': 1., 'translation_m': [0., 0., 0.]}, 'M1': m1, 'M2': m2}
    for m in models.values():
        offset, delta = m['offset_frames'], np.array(m['translation_m'])
        m['fit'] = source.summary(train, offset, delta)
        m['evaluation'] = source.summary(test, offset, delta)
        m['long_window_evaluation'] = source.summary(new_test, offset, delta)
        m['earlier_evaluation'] = source.summary([r for r in earlier if r['split'] == 'eval'], offset, delta)
        m['boundary_sensitivity'] = {str(x): source.summary(test, offset, delta, x) for x in (-3, 0, 3)}
        m['B_ball_evaluation'] = source.ball_metric(offset, delta)
    before, after = models['M0'], models['M2']
    checks = {
        'long_eval_improved': after['long_window_evaluation']['mean_outside_px'] < before['long_window_evaluation']['mean_outside_px'],
        'all_eval_improved': after['evaluation']['mean_outside_px'] < before['evaluation']['mean_outside_px'],
        'earlier_views_nonworsening': all(after['evaluation']['per_camera'][str(c)]['mean_outside_px'] <= before['evaluation']['per_camera'][str(c)]['mean_outside_px']+1e-9 for c in (19, 20, 21)),
        'boundary_sensitivity_improved': all(after['boundary_sensitivity'][str(x)]['mean_outside_px'] < before['boundary_sensitivity'][str(x)]['mean_outside_px'] for x in (-3, 0, 3)),
        'translation_not_on_boundary': bool(np.max(np.abs(after['translation_m'])) < .03-1e-5),
        'ball_all_metrics_nonworsening': all(after['B_ball_evaluation'][k] <= before['B_ball_evaluation'][k]+1e-9 for k in ['mean_absolute_speed_discrepancy_kmh', 'mean_velocity_vector_discrepancy_kmh', 'position_rmse_mm'])}
    compatibility = {}
    for name, group in [('long_only', new_train), ('combined', train)]:
        compatibility[name] = {}
        for margin in (-3, 0, 3):
            selected = [float(o) for o in OFFSETS if source.summary(group, o, np.zeros(3), margin)['inside'] == len(group)]
            compatibility[name][str(margin)] = selected
    states = [{'offset_frames': o, 'B_ball_evaluation': source.ball_metric(o, np.zeros(3))}
              for o in compatibility['combined']['0']]
    result = {'reference': 'OBP_movement_BaseballThrow_001', 'protocol': 'LONG_HAND_PROTOCOL.md',
              'models': models, 'time_only_cases': c1, 'time_translation_cases': c2,
              'training_rows': train, 'evaluation_rows': test, 'new_training_count': 5, 'new_evaluation_count': 7,
              'compatible_clock_offsets': compatibility, 'combined_compatible_B_states': states,
              'engineering_checks': checks, 'engineering_adopted': all(checks.values()),
              'scientific_stage_passed': False, 'actual_athlete_improvement': False, 'force_work_verified': False,
              'annotations': freeze, 'source_sha256': {str(ROOT/'long_hand_regions.json'): sha(ROOT/'long_hand_regions.json'),
                  str(ROOT/'LONG_HAND_PROTOCOL.md'): sha(ROOT/'LONG_HAND_PROTOCOL.md'),
                  **{v['path']: v['sha256'] for v in source.observation['source_sha256'].values()}},
              'interpretation': 'Approximate raw-image surface regions and shared camera model; deterministic clock compatibility is not physical clock truth, anatomical accuracy or a confidence interval.'}
    for p in [OUT/'results.json', ROOT/'long_hand_results.json']:
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'models': {name: {'offset': m['offset_frames'], 'translation_mm': (np.array(m['translation_m'])*1000).tolist(),
            'fit': m['fit']['mean_outside_px'], 'eval': m['evaluation']['mean_outside_px'],
            'new_eval': m['long_window_evaluation']['mean_outside_px'], 'B_speed_mae': m['B_ball_evaluation']['mean_absolute_speed_discrepancy_kmh']} for name, m in models.items()},
            'compatibility': compatibility, 'checks': checks}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
