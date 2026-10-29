"""Evaluate added waveform information with train-only family selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..research_reset.validation import development_verdict, file_digest
from .signals import CUTOFFS, load_features
from .models import FAMILIES, fit_predict, nested_loao


def run(data, metrics, manifest, output):
    tables, audit = load_features(data, metrics, manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, splits, invariance = {}, {}, {}
    for cutoff in CUTOFFS:
        frame = tables[cutoff]
        predictions, summary, fold = nested_loao(frame)
        summaries[str(cutoff)], splits[str(cutoff)] = summary, fold
        for family, rows in predictions.items():
            rows.to_csv(output/f'{family}_{cutoff}_predictions.csv', index=False)
        frame.to_csv(output/f'observed_features_{cutoff}.csv', index=False)
        player = sorted(frame.user.unique())[0]
        train, test = frame[frame.user != player], frame[frame.user == player]
        checks = {}
        for family in FAMILIES:
            before, _ = fit_predict(train, test, family)
            changed = test.copy()
            changed['pitch_speed_kmh'] = 1e6 + np.arange(len(test))
            after, _ = fit_predict(train, changed, family)
            checks[family] = bool(np.array_equal(before.predicted_pair_delta_kmh,
                                                 after.predicted_pair_delta_kmh))
        invariance[str(cutoff)] = dict(heldout_labels_unused=all(checks.values()),
                                       all_outer_splits_disjoint=all(s['players_disjoint'] for s in fold))
        # Preserve completed CV before serializing the larger source/support audit.
        (output/f'checkpoint_{cutoff}.json').write_text(json.dumps(
            dict(summary=summary, splits=fold, invariance=invariance[str(cutoff)]),
            indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'cutoff_hz': cutoff, 'selector': summary['selector']}), flush=True)
    verdict = development_verdict({key: value['selector'] for key, value in summaries.items()},
                                   audit, all(all(v.values()) for v in invariance.values()))
    verdict['development_new_signal_pass'] = verdict.pop('development_pairwise_criteria_pass')
    result = dict(protocol_sha256=file_digest(Path(__file__).with_name('PROTOCOL.md')),
                  audit=audit, summaries=summaries, outer_splits=splits,
                  invariance=invariance, verdict=verdict,
                  hypothesis_intervention_performed=False, data_role='already_seen_development')
    (output/'new_signal_results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('artifacts')/'wholebody')
    args = parser.parse_args()
    run(args.data, args.metrics, args.source_manifest, args.output)


if __name__ == '__main__':
    main()
