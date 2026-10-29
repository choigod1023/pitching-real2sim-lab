"""Run the frozen eligibility audit and one fixed observed-pair experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import (CUTOFFS, FEATURES, IDS, METRICS_SHA256, audit_c3d, audit_grf,
                         development_verdict, file_digest, fit_pair_ridge, make_pairs,
                         pair_loao, pair_summary, predict_pairs, validate_metrics, verify_sources)


def run(data, metrics_path, source_manifest, output):
    data, output = Path(data), Path(output)
    if file_digest(metrics_path) != METRICS_SHA256:
        raise ValueError('changed_frozen_metrics')
    sources = verify_sources(data, source_manifest)
    metadata = pd.read_csv(data/'metadata.csv', dtype=IDS)
    metrics = pd.read_csv(metrics_path, dtype=IDS)
    cohort = validate_metrics(metrics, metadata)
    c3d = audit_c3d(data, metadata)
    force = pd.read_csv(data/'full_sig/force_plate.csv', dtype={'session_pitch': str})
    grf = audit_grf(force, metadata, metrics.session_pitch.unique())
    sources['full_sig/force_plate.csv'] = {'sha256': file_digest(data/'full_sig/force_plate.csv'),
                                          'role': 'availability_audit_only'}
    output.mkdir(parents=True, exist_ok=True)
    summaries, split_audits, invariance = {}, {}, {}
    for cutoff in CUTOFFS:
        pairs = make_pairs(metrics[metrics.cutoff_hz == cutoff])
        predictions, splits = pair_loao(pairs)
        predictions.to_csv(output/f'pair_predictions_{cutoff}.csv', index=False)
        summaries[str(cutoff)] = pair_summary(predictions)
        split_audits[str(cutoff)] = splits
        # The first heldout player's entire outcome vector is changed, never input.
        player = sorted(pairs.user.unique())[0]
        changed = pairs.copy()
        changed.loc[changed.user == player, 'truth_delta_kmh'] = 1e6 + np.arange((changed.user == player).sum())
        repeated, _ = pair_loao(changed)
        expected = predictions[predictions.user == player].predicted_pair_delta_kmh.to_numpy()
        actual = repeated[repeated.user == player].predicted_pair_delta_kmh.to_numpy()
        test = pairs[pairs.user == player]
        model = fit_pair_ridge(pairs[pairs.user != player])
        reverse = test.copy()
        reverse[FEATURES] = -reverse[FEATURES]
        invariance[str(cutoff)] = dict(
            heldout_outcomes_replaced_unchanged=bool(np.array_equal(actual, expected)),
            direction_antisymmetric=bool(np.array_equal(predict_pairs(reverse, model), -predict_pairs(test, model))),
            all_splits_disjoint=all(s['players_disjoint'] for s in splits))
    integrity = all(all(checks.values()) for checks in invariance.values())
    report = dict(protocol_sha256=file_digest(Path(__file__).with_name('PROTOCOL.md')),
                  metrics_sha256=file_digest(metrics_path), sources=sources, cohort=cohort,
                  c3d=c3d, grf=grf, summaries=summaries, invariance_checks=invariance,
                  split_audits=split_audits, verdict=development_verdict(summaries, cohort, integrity),
                  data_role='already_seen_development', hypothesis_intervention_performed=False)
    (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(cohort=cohort, summaries=summaries, verdict=report['verdict']), indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('artifacts'))
    args = parser.parse_args()
    run(args.data, args.metrics, args.source_manifest, args.output)


if __name__ == '__main__':
    main()
