"""Frozen low-rank cubic-basis logistic models with train-only group transforms."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit

from ..pitch_quality import conditional as old

PROTOCOL_COMMIT = '59ae41da8c5f605689a87f4e81a017240539ba38'
MODELS = ('L0', 'L1', 'G0', 'G1', 'GB0', 'GB1')
TARGETS = old.TARGETS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_rows(path, start, end):
    """Keep complete non-bunt FF rows; date checks fail closed at file boundaries."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, list) or not raw:
        raise ValueError('expected_nonempty_pitch_list')
    rows, reasons, seen = [], Counter(), set()
    keep = ('gamePk', 'gameDate', 'atBatIndex', 'pitchIndex', 'pitcherId', 'batterId', 'pitchType',
            'pitchHand', 'batSide', 'pre_count', 'callCode', 'official_pitch_code') + old.FIELDS
    for r in raw:
        if not start <= str(r.get('gameDate', '')) <= end:
            raise ValueError('file_contains_out_of_split_dates')
        key = (r.get('gamePk'), r.get('atBatIndex'), r.get('pitchIndex'))
        if key in seen:
            raise ValueError('duplicate_pitch_identity')
        seen.add(key)
        flags = r.get('official_pitch_code') or {}
        count = r.get('pre_count') or {}
        reason = None
        if r.get('pitchType') != 'FF': reason = 'not_FF'
        elif (r.get('identity_status') != 'verified_within_play'
              or any(type(r.get(k)) is not int or r[k] <= 0 for k in ('pitcherId', 'batterId'))):
            reason = 'identity_uncertain'
        elif (any(type(flags.get(k)) is not bool for k in ('swingStatus', 'swingMissStatus', 'buntAttemptStatus'))
              or not isinstance(r.get('callCode'), str) or flags.get('code') != r['callCode']
              or r.get('detailsCode') != r['callCode']): reason = 'invalid_official_call'
        elif flags['buntAttemptStatus']: reason = 'bunt'
        elif any(type(r.get(k)) not in (int, float) or not np.isfinite(r[k]) for k in old.FIELDS):
            reason = 'missing_numeric'
        elif r.get('batSide') not in ('R', 'L') or r.get('pitchHand') not in ('R', 'L'):
            reason = 'unknown_handedness'
        elif (type(count.get('balls')) is not int or type(count.get('strikes')) is not int
              or not 0 <= count['balls'] <= 3 or not 0 <= count['strikes'] <= 2): reason = 'invalid_pre_count'
        if reason: reasons[reason] += 1
        else: rows.append({k:r[k] for k in keep})
    return rows, dict(raw=len(raw), selected_nonbunt_FF=len(rows), exclusions=dict(reasons))


def swings(rows):
    return [r for r in rows if r['official_pitch_code']['swingStatus']]


def target(rows, kind):
    if kind == 'swing':
        return np.asarray([int(r['official_pitch_code']['swingStatus']) for r in rows], dtype=float)
    return old.target_values(rows, kind)


class Spline:
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError('invalid_spline_training_values')
        self.lo, self.hi = float(x.min()), float(x.max())
        interior = np.unique(np.quantile(x, [1/3, 2/3]))
        interior = interior[(interior > self.lo) & (interior < self.hi)]
        self.knots = np.r_[[self.lo]*4, interior, [self.hi]*4]
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all(): raise ValueError('nonfinite_spline_input')
        clip = dict(below=int(np.sum(x < self.lo)), above=int(np.sum(x > self.hi)),
                    training_min=self.lo, training_max=self.hi)
        if self.lo == self.hi:
            return np.ones((len(x), 1)), clip
        values = BSpline.design_matrix(np.clip(x, self.lo, self.hi), self.knots, 3).toarray()
        return values, clip


class Features:
    def fit(self, rows, model):
        if model not in MODELS: raise ValueError('unknown_model')
        self.model, self.quality = model, model.endswith('1')
        self.splines = {}
        if not model.startswith('L'):
            for field in old.FIELDS if self.quality else old.FIELDS[:3]:
                self.splines[field] = Spline().fit([r[field] for r in rows])
        return self

    def transform(self, rows):
        if self.model.startswith('L'):
            x, self.names, n_cont = old.features(rows, 'both' if self.quality else 'control')
            return x, n_cont, {}
        basis, clips = {}, {}
        for field, spline in self.splines.items():
            basis[field], clips[field] = spline.transform([r[field] for r in rows])
        bat = np.asarray([1 if r['batSide'] == 'R' else -1 for r in rows])[:, None]
        hand = np.asarray([1 if r['pitchHand'] == 'R' else -1 for r in rows])[:, None]
        location = np.einsum('ni,nj->nij', basis['pX'], basis['pZ']).reshape(len(rows), -1)[:, 1:]
        control = np.column_stack([basis['startSpeed'][:, 1:], location])
        pieces = [control, control*bat]
        base_names = [f'speed_basis_{i}' for i in range(basis['startSpeed'].shape[1]-1)]
        base_names += [f'location_tensor_{i}' for i in range(location.shape[1])]
        names = base_names+[n+'_batSide' for n in base_names]
        if self.quality:
            iv, hb = basis['breakVerticalInduced'][:, 1:], basis['breakHorizontal'][:, 1:]
            x = np.asarray([r['pX'] for r in rows])[:, None]
            z = np.asarray([r['pZ'] for r in rows])[:, None]
            movement = np.column_stack([iv, hb, iv*z, hb*x])
            pieces += [basis['spinRate'][:, 1:], movement, movement*hand, movement*bat]
            movement_names = [f'{field}_basis_{i}{suffix}' for field, n in [('IVB',iv.shape[1]),('HB',hb.shape[1])]
                              for suffix in ('',) for i in range(n)]
            movement_names += [f'IVB_basis_{i}_raw_pZ' for i in range(iv.shape[1])]
            movement_names += [f'HB_basis_{i}_raw_pX' for i in range(hb.shape[1])]
            names += [f'spin_basis_{i}' for i in range(basis['spinRate'].shape[1]-1)]
            names += movement_names+[n+'_pitchHand' for n in movement_names]+[n+'_batSide' for n in movement_names]
        continuous = np.column_stack(pieces)
        counts = np.asarray([3*r['pre_count']['balls']+r['pre_count']['strikes'] for r in rows])
        categorical = np.column_stack([(bat[:, 0] == 1).astype(float)] + [(counts == k).astype(float) for k in range(1, 12)])
        self.names = names+['batSide_R']+[f'count_{k//3}_{k%3}' for k in range(1,12)]
        return np.column_stack([continuous, categorical]), continuous.shape[1], clips


def indicator(values, mapping):
    positions = [(i, mapping[v]) for i, v in enumerate(values) if v in mapping]
    ii, jj = zip(*positions) if positions else ([], [])
    return sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(values), len(mapping)))


class Logistic:
    def fit(self, rows, kind, model, lam):
        self.features = Features().fit(rows, model)
        matrix, n_cont, _ = self.features.transform(rows)
        self.encoder = old.Encoder().fit(matrix, old.group_ids(rows), n_cont)
        design = self.encoder.transform(matrix, old.group_ids(rows))
        self.batters = {v:i for i, v in enumerate(sorted({r['batterId'] for r in rows}))} if model.startswith('GB') else {}
        if self.batters:
            design = sparse.hstack([design, indicator([r['batterId'] for r in rows], self.batters)], format='csr')
        y = target(rows, kind)
        if len(np.unique(y)) != 2: raise ValueError('both_target_classes_required')
        penalty = np.ones(design.shape[1]); penalty[0] = 0
        penalty[1+len(self.encoder.groups):1+len(self.encoder.groups)+matrix.shape[1]] = lam
        initial = np.zeros(design.shape[1]); mean = (y.sum()+1)/(len(y)+2)
        initial[0] = np.log(mean/(1-mean))
        result = minimize(old.objective, initial, args=(design, y, penalty), jac=True, method='L-BFGS-B',
                          options=dict(maxiter=1000, ftol=1e-10, gtol=1e-6))
        self.beta = result.x
        centered_sd = np.std(matrix[:, :n_cont]
            - np.asarray([self.encoder.means[g] for g in old.group_ids(rows)]), axis=0)
        self.audit = dict(model=model, lambda_=lam, coefficients=self.beta.tolist(),
                          feature_names=self.features.names,
                          continuous_columns=n_cont, within_sd=centered_sd.tolist(),
                          group_column_order=self.encoder.groups, training_scale=self.encoder.scale.tolist(),
                          group_means={g:v.tolist() for g,v in self.encoder.means.items()},
                          unknown_group_mean=self.encoder.fallback.tolist(), batter_columns=self.batters,
                          spline_knots={k:v.knots.tolist() for k,v in self.features.splines.items()},
                          coefficient_comparison_requires_identical_basis_and_scaling=True)
        self.meta = dict(model=model, lambda_=lam, success=bool(result.success), message=str(result.message),
                         iterations=int(result.nit), max_abs_gradient=float(np.max(np.abs(result.jac))),
                         n_parameters=design.shape[1], train_rows=len(rows), train_misses_or_swings=int(y.sum()),
                         zero_within_sd_features=int(np.sum(centered_sd < 1e-12)),
                         within_sd_percentiles=np.quantile(centered_sd,[0,.5,1]).tolist() if len(centered_sd) else [],
                         coefficient_norm=float(np.linalg.norm(result.x)))
        return self

    def predict(self, rows):
        matrix, _, clips = self.features.transform(rows)
        design = self.encoder.transform(matrix, old.group_ids(rows))
        if self.batters:
            design = sparse.hstack([design, indicator([r['batterId'] for r in rows], self.batters)], format='csr')
        return expit(design @ self.beta), clips


def eligible_test(train, test):
    eligible = old.training_eligibility(train)
    kept = [r for r in test if f"{r['pitcherId']}:FF" in eligible]
    return kept, dict(test_before_eligibility=len(test), test_after_eligibility=len(kept),
                      excluded_unseen_or_sparse=len(test)-len(kept),
                      pitchers=len({r['pitcherId'] for r in kept}))


def metrics(predictions, names):
    by_player = defaultdict(list)
    for r in predictions: by_player[r['pitcherId']].append(r)
    if not by_player: raise ValueError('no_eligible_evaluation_rows')
    per_player, output = {}, {}
    for name in names:
        values = []
        for pid in sorted(by_player):
            rr = by_player[pid]
            brier, loss = old.losses([r['y'] for r in rr], [r[name] for r in rr])
            values.append([float(brier.mean()), float(loss.mean())])
        per_player[name] = np.asarray(values)
        output[name] = dict(brier=float(per_player[name][:, 0].mean()), log_loss=float(per_player[name][:, 1].mean()))
    return output, per_player


def contrast(reference, candidate, per_player, n=2000):
    delta = per_player[reference][:, 0]-per_player[candidate][:, 0]
    rng = np.random.default_rng(20261030)
    draws = delta[rng.integers(0, len(delta), size=(n, len(delta)))].mean(axis=1)
    return dict(reference=reference, candidate=candidate, brier_gain=float(delta.mean()),
                pitcher_bootstrap_95pct=np.quantile(draws, [.025, .975]).tolist(), draws=n, seed=20261030)


def evaluate(train, test, kind, names, lam, *, enforce_eligibility=True, audit_dir=None):
    if {r['gamePk'] for r in train} & {r['gamePk'] for r in test}: raise ValueError('game_leakage')
    test, audit = eligible_test(train, test) if enforce_eligibility else (test, dict(test_after_eligibility=len(test), pitchers=len({r['pitcherId'] for r in test})))
    if not test: raise ValueError('no_eligible_evaluation_rows')
    mean, player = old.baseline_probabilities(old.group_ids(train), target(train, kind), old.group_ids(test))
    preds = [dict(gamePk=r['gamePk'], atBatIndex=r['atBatIndex'], pitchIndex=r['pitchIndex'],
                  pitcherId=r['pitcherId'], batterId=r['batterId'], y=float(y), global_mean=float(a), pitcher_mean=float(b))
             for r, y, a, b in zip(test, target(test, kind), mean, player)]
    fits, clips = [], {}
    for name in names:
        print(f'fit {kind} {name} lambda={lam} train={len(train)} test={len(test)}', flush=True)
        fitted = Logistic().fit(train, kind, name, 1 if name.startswith('L') else lam)
        if audit_dir is not None:
            audit_path = Path(audit_dir)/f'{kind}_{name}_{lam}.json'
            save(audit_path, fitted.audit)
            fitted.meta.update(fit_audit_path=str(audit_path.resolve()), fit_audit_sha256=digest(audit_path))
        p, clips[name] = fitted.predict(test)
        fits.append(fitted.meta)
        for r, probability in zip(preds, p): r[name] = float(probability)
    metric, per_player = metrics(preds, ('global_mean', 'pitcher_mean')+tuple(names))
    training_batters = {r['batterId'] for r in train}
    audit.update(train_rows=len(train), train_games=len({r['gamePk'] for r in train}),
                 test_games=len({r['gamePk'] for r in test}),
                 unseen_batter_rows=sum(r['batterId'] not in training_batters for r in test))
    return dict(metrics=metric, fits=fits, audit=audit, clipped_inputs=clips), preds, per_player


def verdict(summary, comparison, technical_passed=False):
    a, m = summary['audit'], summary['metrics']
    ref, candidate = comparison['reference'], comparison['candidate']
    checks = dict(collection_and_selection_integrity=technical_passed is True,
                  swings_at_least_2000=a['test_after_eligibility'] >= 2000, pitchers_at_least_50=a['pitchers'] >= 50,
                  all_six_primary_fits_converged=len(summary['fits']) == 6 and all(f['success'] for f in summary['fits']),
                  improvement_ci_lower_positive=comparison['pitcher_bootstrap_95pct'][0] > 0,
                  lower_log_loss=m[candidate]['log_loss'] < m[ref]['log_loss'],
                  better_than_pitcher_mean=m[candidate]['brier'] < m['pitcher_mean']['brier'])
    return dict(checks=checks, passed=all(checks.values()))


def choose(train_file, validation_file, output, local=None):
    train, a = read_rows(train_file, '2025-06-01', '2025-07-15')
    valid, b = read_rows(validation_file, '2025-07-16', '2025-07-31')
    train, valid = swings(train), swings(valid)
    candidates = {}
    for lam in (1, 10):
        summary, _, _ = evaluate(train, valid, TARGETS[0], ('GB0',), lam,
                                audit_dir=Path(local)/str(lam) if local is not None else None)
        candidates[str(lam)] = summary
    successful = [lam for lam in (1, 10) if candidates[str(lam)]['fits'][0]['success']]
    if not successful: raise ValueError('all_selection_fits_failed')
    selected = min(successful, key=lambda lam: (candidates[str(lam)]['metrics']['GB0']['brier'], lam))
    result = dict(protocol_commit=PROTOCOL_COMMIT, code_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),
                  train_sha256=digest(train_file), validation_sha256=digest(validation_file),
                  selected_lambda=selected, candidates=candidates, file_audits=[a, b],
                  all_selection_candidates_converged=len(successful) == 2,
                  final_test_opened=False, selection_target=TARGETS[0])
    save(output, result)
    return result


def learning_curve(train, validation, lam, audit_dir=None):
    periods = ('2025-07-01', '2025-06-16', '2025-06-01')
    subsets = [[r for r in train if r['gameDate'] >= start] for start in periods]
    common = set.intersection(*(old.training_eligibility(rr) for rr in subsets))
    fixed = [r for r in validation if f"{r['pitcherId']}:FF" in common]
    results = []
    if not fixed: return dict(status='no_common_eligible_players', periods=[])
    for start, subset in zip(periods, subsets):
        s, _, _ = evaluate(subset, fixed, TARGETS[0], ('GB0', 'GB1'), lam, enforce_eligibility=False,
                            audit_dir=Path(audit_dir)/start if audit_dir is not None else None)
        results.append(dict(train_start=start, **s))
    return dict(status='descriptive_only_not_used_for_selection', fixed_validation_swings=len(fixed),
                fixed_validation_pitchers=len({r['pitcherId'] for r in fixed}), periods=results)


def collection_integrity(audit_file, files):
    audit = json.loads(Path(audit_file).read_text(encoding='utf-8'))
    if (audit.get('collection_complete') is not True or audit.get('interval') != ['2025-06-01','2025-08-31']
            or audit.get('schedule_days_received') != 92 or audit.get('schedule_days_requested') != 92
            or audit.get('failures') != [] or not audit.get('selected_game_count', 0)
            or audit.get('selected_game_count') != audit.get('parsed_game_count')):
        raise ValueError('collection_incomplete')
    for name, path in zip(('dev_train','dev_validation','final_test'), files):
        if audit.get('split_exports', {}).get(name, {}).get('sha256') != digest(path):
            raise ValueError('collection_split_hash_mismatch')
    return dict(passed=True, audit_sha256=digest(audit_file), games=audit['parsed_game_count'], days=92)


def final(train_file, validation_file, test_file, choice_file, output, local, collection_audit):
    choice = json.loads(Path(choice_file).read_text(encoding='utf-8'))
    if (choice['code_sha256'] != digest(__file__) or choice['train_sha256'] != digest(train_file)
            or choice['validation_sha256'] != digest(validation_file)
            or choice['protocol_sha256'] != digest(Path(__file__).with_name('PROTOCOL.md'))
            or choice['selected_lambda'] not in (1, 10) or choice['final_test_opened'] is not False):
        raise ValueError('selection_provenance_mismatch')
    technical = collection_integrity(collection_audit, (train_file,validation_file,test_file))
    technical['all_selection_candidates_converged'] = choice.get('all_selection_candidates_converged') is True
    technical['passed'] &= technical['all_selection_candidates_converged']
    dev, a = read_rows(train_file, '2025-06-01', '2025-07-15')
    valid, b = read_rows(validation_file, '2025-07-16', '2025-07-31')
    # Only this command opens the August pitch rows after a selection record exists.
    test, t = read_rows(test_file, '2025-08-01', '2025-08-31')
    lam = choice['selected_lambda']; train = dev+valid
    outputs, predictions = {}, {}
    for kind in TARGETS:
        s, p, pp = evaluate(swings(train), swings(test), kind, MODELS, lam, audit_dir=Path(local)/'fits'/kind)
        s['A_prediction'] = contrast('L0', 'GB0', pp); s['B_quality_increment'] = contrast('GB0', 'GB1', pp)
        s['verdict_A'] = verdict(s, s['A_prediction'], technical['passed'])
        s['verdict_B'] = verdict(s, s['B_quality_increment'], technical['passed'])
        outputs[kind], predictions[kind] = s, p
    # This different target is a diagnostic, not a replacement primary endpoint.
    s, p, pp = evaluate(train, test, 'swing', ('GB0', 'GB1'), lam, audit_dir=Path(local)/'fits'/'swing')
    s['quality_increment'] = contrast('GB0', 'GB1', pp)
    outputs['swing'], predictions['swing'] = s, p
    curve = learning_curve(swings(dev), swings(valid), lam, audit_dir=Path(local)/'fits'/'learning_curve')
    result = dict(protocol_commit=PROTOCOL_COMMIT, selection_sha256=digest(choice_file), selected_lambda=lam,
                  input_hashes=[digest(p) for p in (train_file, validation_file, test_file)],
                  file_audits=[a, b, t], technical=technical, analyses=outputs, learning_curve=curve,
                  coefficient_stability_across_different_bases='not_directly_comparable_see_local_fit_audits',
                  body_intervention_validated=False, physics_units_verified=False, real2sim_gate_changed=False)
    save(Path(output)/'summary.json', result)
    save(Path(local)/'predictions.json', predictions)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['choose', 'final'])
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--validation', required=True, type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--collection-audit', type=Path)
    parser.add_argument('--choice', type=Path, default=Path(__file__).with_name('results')/'choice.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--local-output', type=Path, default=Path(__file__).parents[2]/'artifacts'/'temporal_quality')
    args = parser.parse_args()
    if args.stage == 'choose': choose(args.train, args.validation, args.choice, args.local_output/'selection')
    else:
        if args.test is None or args.collection_audit is None: parser.error('final requires --test and --collection-audit')
        final(args.train, args.validation, args.test, args.choice, args.output, args.local_output, args.collection_audit)


if __name__ == '__main__': main()
