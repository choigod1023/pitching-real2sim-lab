"""Reproducible stage1 entry point; all new outputs stay under this package."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from .metrics import read_cohort, CUTOFFS, COLUMNS
from .modeling import (nested_predictions, prediction_summary, associations,
                       filter_sensitivity, scientific_gate)

PREREGISTRATION_COMMIT = '053c22251e62a1ac42489bc1dbe4a62918b41393'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(data, output):
    output.mkdir(parents=True, exist_ok=True)
    derived, excluded, total = read_cohort(data)
    derived.to_csv(output / 'pitch_metrics.csv', index=False)
    (output / 'exclusions.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding='utf-8')
    summaries, audits, correlations, individual_errors = {}, {}, [], []
    integrity = True
    for cutoff in CUTOFFS:
        subset = derived[derived.cutoff_hz == cutoff]
        summaries[str(cutoff)], audits[str(cutoff)] = {}, {}
        for within, label in ((True, 'within'), (False, 'between')):
            predictions, audit = nested_predictions(subset, within)
            summary, errors = prediction_summary(predictions)
            summaries[str(cutoff)][label] = summary
            audits[str(cutoff)][label] = audit
            integrity &= not any(a['overlap'] for a in audit)
            predictions.to_csv(output / f'{label}_{cutoff}_predictions.csv', index=False)
            individual_errors.append(errors.assign(cutoff_hz=cutoff, level=label))
        correlations.append(associations(subset).assign(cutoff_hz=cutoff))
    correlation = pd.concat(correlations, ignore_index=True)
    correlation.to_csv(output / 'associations.csv', index=False)
    pd.concat(individual_errors).to_csv(output / 'player_errors.csv', index=False)
    spread, sign_changes = filter_sensitivity(derived)
    spread.to_csv(output / 'filter_sensitivity.csv', index=False)
    # Variation of observable proxies is descriptive, not command accuracy.
    derived[derived.cutoff_hz == 20].groupby('user')[['wrist_height_br_m', 'arm_slot_poi_deg']].std().to_csv(output / 'release_proxy_variability.csv')
    source_files = ['metadata.csv', 'poi/poi_metrics.csv', 'data_dictionary.csv',
                    *[f'full_sig/{name}.csv' for name in COLUMNS]]
    package = Path(__file__).parent
    manifest = {'sources': {name: {'path': str((data / name).resolve()), 'sha256': file_hash(data / name)} for name in source_files},
                'code': {p.name: file_hash(p) for p in sorted(package.glob('*.py'))},
                'protocol_sha256': file_hash(package / 'PROTOCOL.md'),
                'preregistration_commit': PREREGISTRATION_COMMIT,
                'head_at_execution': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=package, text=True).strip()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'split_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    summary = {'source_pitches': total, 'included_pitches': int(derived.session_pitch.nunique()),
               'included_players': int(derived.user.nunique()), 'excluded_pitches': len(excluded),
               'conditions': summaries, 'filter_lag_sign_reversals': sign_changes,
               'gate': scientific_gate(summaries, integrity),
               'historical_reference': {'players': 89, 'evaluation_pitches': 203, 'model_mae_kmh': .990, 'mean_baseline_mae_kmh': .964, 'comparable_before_after': False},
               'actual_ball_release_height_available': False,
               'raw_filter_reprocessing': False, 'retrospective_oracle_baseline': True}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description='Stage1 observational coach hypotheses')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'artifacts/results')
    args = parser.parse_args()
    package = Path(__file__).parent.resolve()
    if not args.output.resolve().is_relative_to(package):
        parser.error('output must stay inside coach_hypotheses/stage1')
    summary = analyze(args.data, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
