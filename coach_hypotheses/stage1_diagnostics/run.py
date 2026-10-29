"""Run frozen diagnostics and nested selection without changing stage1 outputs."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from ..stage1.metrics import FEATURES, CUTOFFS
from ..stage1.modeling import prediction_summary, scientific_gate
from ..stage1.run import file_hash
from .diagnostics import (FAMILIES, WAVE, verify_sources, add_waveforms, data_diagnostics,
                          compare_models, permutation_controls)


def synthetic_control():
    rng = np.random.default_rng(20261029)
    columns = sorted(set(sum(FAMILIES.values(), [])))
    rows = []
    for player in range(8):
        values = rng.normal(size=(5,len(columns)))
        x = values[:, columns.index(FEATURES[0])]
        y = 100+player+1.5*(x-x.mean())+rng.normal(0,.02,5)
        for trial in range(5):
            rows.append(dict(zip(columns, values[trial]), user=str(player),
                             session_pitch=f'synthetic_{player}_{trial}', pitch_speed_kmh=y[trial]))
    predictions, _, _ = compare_models(pd.DataFrame(rows))
    selector = predictions[predictions.family == 'selector']
    summary, _ = prediction_summary(selector)
    return {'synthetic_only': True, 'known_signal_mae_kmh': summary['player_balanced_model_mae_kmh'],
            'null_mae_kmh': summary['player_balanced_baseline_mae_kmh'],
            'recovers_known_signal': summary['player_balanced_model_mae_kmh'] < .2*summary['player_balanced_baseline_mae_kmh']}


def analyze(data, output):
    package = Path(__file__).parent
    frozen = package.parent/'stage1/results'
    output.mkdir(parents=True, exist_ok=True)
    source_checks = verify_sources(data, frozen)
    frame = pd.read_csv(frozen/'pitch_metrics.csv', dtype={'user': str, 'session_pitch': str})
    frame = add_waveforms(data, frame)
    frame[['user','session_pitch','cutoff_hz',*WAVE]].to_csv(output/'waveform_features.csv', index=False)
    qc, feature = data_diagnostics(frame)
    feature.to_csv(output/'feature_diagnostics.csv', index=False)
    summaries, all_predictions, all_folds, all_scores = {}, [], [], []
    regression_checks = {}
    for cutoff in CUTOFFS:
        predictions, folds, scores = compare_models(frame[frame.cutoff_hz == cutoff])
        all_predictions.append(predictions.assign(cutoff_hz=cutoff))
        all_folds.append(folds.assign(cutoff_hz=cutoff))
        all_scores.append(scores.assign(cutoff_hz=cutoff))
        summaries[str(cutoff)] = {}
        for family, block in predictions.groupby('family'):
            summary, errors = prediction_summary(block)
            summaries[str(cutoff)][family] = summary
            errors.to_csv(output/f'errors_{family}_{cutoff}.csv', index=False)
        previous = pd.read_csv(frozen/f'within_{cutoff}_predictions.csv', dtype={'user': str,'session_pitch':str})
        current = predictions[predictions.family == 'original12'].merge(previous, on=['user','session_pitch'], suffixes=('_new','_old'), validate='one_to_one')
        regression_checks[str(cutoff)] = bool(np.allclose(current.prediction_new, current.prediction_old, atol=1e-10, rtol=0))
        if not regression_checks[str(cutoff)]:
            raise AssertionError('original_model_not_reproduced')
    predictions = pd.concat(all_predictions, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    predictions.to_csv(output/'predictions.csv', index=False)
    folds.to_csv(output/'fold_diagnostics.csv', index=False)
    pd.concat(all_scores).to_csv(output/'inner_scores.csv', index=False)
    permutation = permutation_controls(frame[frame.cutoff_hz == 20])
    permutation.to_csv(output/'permutation_controls.csv', index=False)
    gates = {str(c): {'within': summaries[str(c)]['selector']} for c in CUTOFFS}
    integrity = not folds.held_player_in_training.any() and all(regression_checks.values()) and all(source_checks.values())
    summary = {'data': qc, 'families': summaries, 'selector_gate': scientific_gate(gates, integrity),
               'original_gate_is_unchanged': True, 'source_hash_checks': source_checks,
               'original_predictions_reproduced': regression_checks,
               'selector_counts': {str(c): predictions[(predictions.cutoff_hz == c)&(predictions.family == 'selector')].groupby('user').selected_family.first().value_counts().to_dict() for c in CUTOFFS},
               'synthetic_control': synthetic_control(),
               'permutation_actual_mae_kmh': float(permutation.iloc[0].mae_kmh),
               'permutation_shuffled_median_mae_kmh': float(permutation.iloc[1:].mae_kmh.median()),
               'permutation_shuffled_min_max_kmh': [float(permutation.iloc[1:].mae_kmh.min()),float(permutation.iloc[1:].mae_kmh.max())],
               'independent_external_validation': False}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    manifest = {'preregistered_commit': '1edccac', 'head_at_execution': subprocess.check_output(['git','rev-parse','HEAD'],cwd=package,text=True).strip(),
                'protocol_sha256': file_hash(package/'PROTOCOL.md'),
                'original_protocol_sha256': file_hash(package.parent/'stage1/PROTOCOL.md'),
                'code_sha256': {p.name:file_hash(p) for p in package.glob('*.py')},
                'input_features_sha256':file_hash(frozen/'pitch_metrics.csv')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description='Frozen stage1 failure diagnostics')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/results')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must stay inside stage1_diagnostics')
    summary = analyze(args.data,args.output)
    print(json.dumps({'gate':summary['selector_gate'],'selector_counts':summary['selector_counts']},indent=2))


if __name__ == '__main__':
    main()
