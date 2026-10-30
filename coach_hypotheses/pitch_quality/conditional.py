"""Preregistered game-heldout, within-pitcher standardized observational analysis."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


PROTOCOL_COMMIT = '9198cf13067e031ef109288b57745299f56aa9e1'
FIELDS = ('startSpeed', 'pX', 'pZ', 'spinRate', 'breakVerticalInduced', 'breakHorizontal')
MODELS = ('control', 'spin', 'movement', 'both')
TARGETS = ('strict_SW', 'official_nonbunt_swingMissStatus')
START, STOP = '2025-06-01', '2025-06-14'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_samples(rows):
    """Do not impute unknown values or silently reuse ambiguous identities."""
    selected, reasons, seen = [], Counter(), set()
    for row in rows:
        key = (row.get('gamePk'), row.get('atBatIndex'), row.get('pitchIndex'))
        if key in seen:
            raise ValueError('duplicate_pitch_identity')
        seen.add(key)
        flags = row.get('official_pitch_code') or {}
        reason = None
        if not START <= str(row.get('gameDate', '')) <= STOP:
            reason = 'outside_fixed_date_range'
        elif (row.get('identity_status') != 'verified_within_play'
              or type(row.get('pitcherId')) is not int or row['pitcherId'] <= 0
              or type(row.get('batterId')) is not int or row['batterId'] <= 0):
            reason = 'identity_uncertain'
        elif any(type(flags.get(k)) is not bool for k in ('swingStatus', 'buntAttemptStatus', 'swingMissStatus')):
            reason = 'unknown_official_flags'
        elif (not isinstance(row.get('callCode'), str) or flags.get('code') != row['callCode']
              or row.get('detailsCode') != row['callCode']):
            reason = 'call_lookup_or_details_code_mismatch'
        elif not flags['swingStatus'] or flags['buntAttemptStatus']:
            reason = 'not_nonbunt_swing'
        elif row.get('pitchHand') not in ('R', 'L') or row.get('batSide') not in ('R', 'L'):
            reason = 'unknown_handedness'
        elif not isinstance(row.get('pitchType'), str) or not row['pitchType']:
            reason = 'unknown_pitch_type'
        elif any(type(row.get(k)) not in (int, float) or not np.isfinite(row[k]) for k in FIELDS):
            reason = 'missing_or_nonfinite_required_numeric'
        else:
            count = row.get('pre_count') or {}
            if (type(count.get('balls')) is not int or type(count.get('strikes')) is not int
                    or not 0 <= count['balls'] <= 3 or not 0 <= count['strikes'] <= 2):
                reason = 'invalid_pre_count'
        if reason:
            reasons[reason] += 1
        else:
            selected.append(row)
    return selected, dict(reasons)


def features(rows, model):
    """Return continuous terms first, followed by unchanged categorical controls."""
    if model not in MODELS or not rows:
        raise ValueError('invalid_model_or_empty_rows')
    names = ['startSpeed', 'pX', 'pZ', 'pX_squared', 'pZ_squared', 'pX_pZ']
    names += [n + '_batSide' for n in names.copy()]
    if model in ('spin', 'both'):
        names += ['spinRate']
    if model in ('movement', 'both'):
        names += [n + suffix for n in ('breakVerticalInduced', 'breakHorizontal')
                  for suffix in ('', '_pitchHand', '_batSide')]
    n_cont = len(names)
    counts = [(b, s) for b in range(4) for s in range(3) if (b, s) != (0, 0)]
    names += ['batSide_R'] + [f'count_{b}_{s}' for b, s in counts]
    values = []
    for r in rows:
        speed, x, z = (r[k] for k in ('startSpeed', 'pX', 'pZ'))
        bat, hand = (1 if r[k] == 'R' else -1 for k in ('batSide', 'pitchHand'))
        base = [speed, x, z, x*x, z*z, x*z]
        v = base + [t*bat for t in base]
        if model in ('spin', 'both'):
            v += [r['spinRate']]
        if model in ('movement', 'both'):
            for k in ('breakVerticalInduced', 'breakHorizontal'):
                v += [r[k], r[k]*hand, r[k]*bat]
        c = r['pre_count']
        v += [int(bat == 1)] + [int((c['balls'], c['strikes']) == pair) for pair in counts]
        values.append(v)
    return np.asarray(values, dtype=float), names, n_cont


def group_ids(rows):
    return np.asarray([f"{r['pitcherId']}:{r['pitchType']}" for r in rows])


class Encoder:
    """All means, scales and group columns are trained without evaluation rows."""
    def fit(self, values, groups, n_cont):
        values = np.asarray(values, dtype=float)
        groups = np.asarray(groups)
        if (values.ndim != 2 or len(values) == 0 or len(values) != len(groups)
                or not np.isfinite(values).all() or not 0 <= n_cont <= values.shape[1]):
            raise ValueError('invalid_training_design')
        self.n_cont = n_cont
        self.groups = sorted(set(groups))
        self.indices = {g: i for i, g in enumerate(self.groups)}
        self.fallback = values[:, :n_cont].mean(axis=0)
        self.means = {g: values[groups == g, :n_cont].mean(axis=0) for g in self.groups}
        residual = values[:, :n_cont] - np.asarray([self.means[g] for g in groups])
        scale = residual.std(axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.)
        self.width = values.shape[1]
        return self

    def transform(self, values, groups):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.width or len(values) != len(groups):
            raise ValueError('evaluation_design_mismatch')
        values = values.copy()
        values[:, :self.n_cont] = (values[:, :self.n_cont]
            - np.asarray([self.means.get(g, self.fallback) for g in groups])) / self.scale
        if not np.isfinite(values).all():
            raise ValueError('nonfinite_evaluation_design')
        positions = [(i, self.indices[g]) for i, g in enumerate(groups) if g in self.indices]
        ii, jj = zip(*positions) if positions else ([], [])
        intercepts = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(values), len(self.groups)))
        return sparse.hstack([np.ones((len(values), 1)), intercepts, values], format='csr')


def objective(beta, design, y, penalty):
    eta = np.asarray(design @ beta)
    value = np.logaddexp(0., eta).sum() - y @ eta + .5*np.sum(penalty*beta*beta)
    gradient = np.asarray(design.T @ (expit(eta)-y)).ravel() + penalty*beta
    return float(value), gradient


def fit_logistic(design, y):
    y = np.asarray(y, dtype=float)
    if not len(y) or len(y) != design.shape[0] or not np.isin(y, [0., 1.]).all():
        raise ValueError('invalid_binary_target')
    if len(np.unique(y)) != 2:
        raise ValueError('both_target_classes_required')
    penalty = np.ones(design.shape[1]); penalty[0] = 0.
    initial = np.zeros(design.shape[1]); mean = (y.sum()+1)/(len(y)+2)
    initial[0] = np.log(mean/(1-mean))
    result = minimize(objective, initial, args=(design, y, penalty), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=500, ftol=1e-10, gtol=1e-6))
    return result.x, dict(success=bool(result.success), message=str(result.message),
                          iterations=int(result.nit), objective=float(result.fun),
                          max_abs_gradient=float(np.max(np.abs(result.jac))))


def training_eligibility(rows):
    counts, games = Counter(), defaultdict(set)
    for row in rows:
        g = f"{row['pitcherId']}:{row['pitchType']}"
        counts[g] += 1; games[g].add(row['gamePk'])
    return {g for g in counts if counts[g] >= 20 and len(games[g]) >= 2}


def baseline_probabilities(train_groups, train_y, test_groups):
    counts, successes = Counter(train_groups), Counter()
    for g, y in zip(train_groups, train_y):
        successes[g] += y
    global_mean = float(np.mean(train_y))
    player = np.asarray([(successes[g]+1)/(counts[g]+2) if g in counts else global_mean for g in test_groups])
    return np.full(len(test_groups), global_mean), player


def losses(y, probability):
    probability = np.asarray(probability, dtype=float)
    y = np.asarray(y, dtype=float)
    if (probability.shape != y.shape or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('invalid_probabilities')
    p = np.clip(probability, 1e-12, 1-1e-12)
    return (probability-y)**2, -(y*np.log(p)+(1-y)*np.log1p(-p))


def summarize_predictions(predictions, *, bootstrap_n=2000):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row['pitcherId']].append(row)
    ids = sorted(grouped)
    metric_rows, metrics = {}, {}
    for model in ('global_mean', 'pitcher_mean') + MODELS:
        per_player = []
        for pitcher in ids:
            rr = grouped[pitcher]
            brier, logloss = losses([r['y'] for r in rr], [r[model] for r in rr])
            per_player.append([np.mean(brier), np.mean(logloss)])
        metric_rows[model] = np.asarray(per_player)
        metrics[model] = dict(brier=float(np.mean(metric_rows[model][:, 0])),
                              log_loss=float(np.mean(metric_rows[model][:, 1])))
    delta = metric_rows['control'][:, 0] - metric_rows['both'][:, 0]
    rng = np.random.default_rng(20261030)
    sampled = delta[rng.integers(0, len(ids), size=(bootstrap_n, len(ids)))].mean(axis=1)
    ci = np.quantile(sampled, [.025, .975])
    return dict(swings=len(predictions), pitchers=len(ids), metrics=metrics,
                control_minus_both_brier=float(delta.mean()),
                pitcher_cluster_bootstrap_95pct=[float(v) for v in ci],
                bootstrap_draws=bootstrap_n, seed=20261030,
                pitchers_with_positive_brier_gain=int(np.sum(delta > 0)))


def acceptance(summary, fits):
    metrics = summary['metrics']
    checks = dict(swings_at_least_2000=summary['swings'] >= 2000,
                  pitchers_at_least_50=summary['pitchers'] >= 50,
                  all_five_folds_and_models_converged=len(fits) == 20 and all(f['success'] for f in fits),
                  brier_improvement_ci_lower_positive=summary['pitcher_cluster_bootstrap_95pct'][0] > 0,
                  lower_log_loss=metrics['both']['log_loss'] < metrics['control']['log_loss'],
                  better_than_pitcher_mean_brier=metrics['both']['brier'] < metrics['pitcher_mean']['brier'])
    return dict(checks=checks, passed=all(checks.values()))


def correlations(rows, target):
    eligible = training_eligibility(rows)
    grouped = defaultdict(list)
    for r in rows:
        if f"{r['pitcherId']}:{r['pitchType']}" in eligible:
            grouped[r['pitcherId']].append(r)
    result = {}
    for field in ('spinRate', 'breakVerticalInduced', 'breakHorizontal'):
        means, demeaned = [], []
        for rr in grouped.values():
            x = np.asarray([r[field] for r in rr]); y = target_values(rr, target)
            means.append([x.mean(), y.mean()])
            demeaned.extend(zip(x-x.mean(), y-y.mean()))
        result[field] = dict(between_pitchers=pearson(means), within_pitcher=pearson(demeaned))
    return dict(eligible_pitchers=len(grouped), fields=result)


def pearson(pairs):
    values = np.asarray(pairs)
    if values.ndim != 2 or len(values) < 2 or np.min(values.std(axis=0)) <= 1e-12:
        return None
    return float(np.corrcoef(values.T)[0, 1])


def target_values(rows, target):
    if target not in TARGETS:
        raise ValueError('unknown_target')
    return np.asarray([int(r['callCode'] in ('S', 'W')) if target == 'strict_SW'
                       else int(r['official_pitch_code']['swingMissStatus']) for r in rows], dtype=float)


def crossfit(rows, target):
    folds, fits, predictions = [], [], []
    groups = group_ids(rows); y = target_values(rows, target)
    games = np.asarray([r['gamePk'] for r in rows])
    for fold in range(5):
        train_idx, raw_test_idx = np.flatnonzero(games % 5 != fold), np.flatnonzero(games % 5 == fold)
        train = [rows[i] for i in train_idx]
        eligible = training_eligibility(train)
        test_idx = np.asarray([i for i in raw_test_idx if groups[i] in eligible], dtype=int)
        test = [rows[i] for i in test_idx]
        audit = dict(fold=fold, training_swings=len(train), test_swings=len(test),
                     excluded_unseen_or_sparse_test_swings=len(raw_test_idx)-len(test_idx),
                     eligible_groups=len(eligible), train_games=sorted(set(games[train_idx].tolist())),
                     test_games=sorted(set(games[test_idx].tolist())))
        if set(audit['train_games']) & set(audit['test_games']):
            raise ValueError('game_leakage')
        folds.append(audit)
        if not train or not test:
            continue
        if len(np.unique(y[train_idx])) != 2:
            fits.extend(dict(fold=fold, model=model, success=False, message='both_target_classes_required') for model in MODELS)
            audit['all_predictions_quarantined_due_to_single_class_training'] = True
            continue
        mean_p, pitcher_p = baseline_probabilities(groups[train_idx], y[train_idx], groups[test_idx])
        output = [dict(gamePk=r['gamePk'], atBatIndex=r['atBatIndex'], pitchIndex=r['pitchIndex'],
                       pitcherId=r['pitcherId'], pitchType=r['pitchType'], fold=fold, y=float(y[i]),
                       global_mean=float(a), pitcher_mean=float(b))
                  for r, i, a, b in zip(test, test_idx, mean_p, pitcher_p)]
        fold_ok = True
        for model in MODELS:
            matrix, names, n_cont = features(train, model)
            encoder = Encoder().fit(matrix, groups[train_idx], n_cont)
            design = encoder.transform(matrix, groups[train_idx])
            beta, fit = fit_logistic(design, y[train_idx])
            fit.update(fold=fold, model=model)
            fits.append(fit); fold_ok &= fit['success']
            test_values = features(test, model)[0]
            p = expit(encoder.transform(test_values, groups[test_idx]) @ beta)
            for record, probability in zip(output, p):
                record[model] = float(probability)
            fit['feature_coefficients'] = (dict(zip(names, beta[1+len(encoder.groups):].tolist()))
                                           if fit['success'] else None)
        if fold_ok:
            predictions.extend(output)
        else:
            audit['all_predictions_quarantined_due_to_nonconvergence'] = True
    if not predictions:
        return dict(status='insufficient_successful_folds', folds=folds, fits=fits), []
    summary = summarize_predictions(predictions)
    summary.update(folds=folds, fits=fits, descriptive_correlations=correlations(rows, target))
    all_values, names, n_cont = features(rows, 'both')
    encoder = Encoder().fit(all_values, groups, n_cont)
    beta, full_fit = fit_logistic(encoder.transform(all_values, groups), y)
    full_fit['exploratory_feature_coefficients'] = (dict(zip(names, beta[1+len(encoder.groups):].tolist()))
                                                   if full_fit['success'] else None)
    full_fit['interpretation'] = 'conditional_log_odds_per_training_within_group_SD_not_causal'
    summary['full_data_both_fit'] = full_fit
    summary['preregistered_primary_checks'] = acceptance(summary, fits)
    return summary, predictions


def run(input_path, output_dir, local_dir):
    input_path, output_dir, local_dir = map(Path, (input_path, output_dir, local_dir))
    rows = json.loads(input_path.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('expected_pitch_row_list')
    selected, exclusions = select_samples(rows)
    type_counts = Counter(r['pitchType'] for r in selected)
    analyses, prediction_sets, skipped = {}, {}, {}
    for pitch_type in sorted(type_counts, key=lambda k: (k != 'FF', k)):
        rr = [r for r in selected if r['pitchType'] == pitch_type]
        pitchers = len({r['pitcherId'] for r in rr})
        if pitch_type != 'FF' and (len(rr) < 1000 or pitchers < 20):
            skipped[pitch_type] = dict(swings=len(rr), pitchers=pitchers, reason='preregistered_secondary_sample_threshold')
            continue
        analyses[pitch_type] = {}
        for target in TARGETS:
            print(f'Running {pitch_type} {target}: {len(rr)} swings', flush=True)
            summary, predictions = crossfit(rr, target)
            analyses[pitch_type][target] = summary
            prediction_sets[f'{pitch_type}:{target}'] = predictions
    ff = analyses.get('FF', {}).get('strict_SW', {})
    result = dict(protocol_commit=PROTOCOL_COMMIT, raw_pitches=len(rows), complete_nonbunt_swings=len(selected),
                  exclusions=exclusions, pitch_type_counts=dict(type_counts), skipped_types=skipped,
                  analyses=analyses, primary_passed=ff.get('preregistered_primary_checks', {}).get('passed', False),
                  standardized_observational_analysis_ready=(bool(ff.get('swings', 0))
                      and len(ff.get('fits', [])) == 20 and all(f['success'] for f in ff.get('fits', []))),
                  physics_units_verified=False, original_conditional_study_gate_changed=False,
                  actual_athlete_effect_validated=False, body_to_spin_link_validated=False)
    manifest = dict(protocol_commit=PROTOCOL_COMMIT, input_path=str(input_path.resolve()),
                    input_sha256=digest(input_path), code_sha256=digest(__file__),
                    protocol_sha256=digest(Path(__file__).with_name('CONDITIONAL_PROTOCOL.md')),
                    execution_notes_sha256=digest(Path(__file__).with_name('CONDITIONAL_EXECUTION_NOTES.md')),
                    date_range=[START, STOP], unit_conversions_performed=False,
                    raw_or_pitch_level_data_redistributed=False)
    output_dir.mkdir(parents=True, exist_ok=True); local_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [('conditional_summary.json', result), ('conditional_manifest.json', manifest)]:
        (output_dir/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (local_dir/'predictions.json').write_text(json.dumps(prediction_sets, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'analyses'}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--local-output', type=Path, default=Path(__file__).parents[2]/'artifacts'/'conditional_pitch_quality')
    args = parser.parse_args()
    run(args.input, args.output, args.local_output)


if __name__ == '__main__':
    main()
