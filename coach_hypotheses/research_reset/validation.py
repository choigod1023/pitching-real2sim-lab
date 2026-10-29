"""Eligibility and fixed pairwise LOAO checks for already-observed pitching data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ['lead_knee_extension_deg', 'stride_reported', 'stride_reported_sq',
            'hip_shoulder_sep_fp10_deg', 'hip_shoulder_sep_peak_deg',
            'wrist_height_br_m', 'arm_slot_poi_deg']
CUTOFFS = (10, 20, 30)
IDS = {'session_pitch': str, 'user': str, 'session': str}
GRF = ['rear_force_x', 'rear_force_y', 'rear_force_z',
       'lead_force_x', 'lead_force_y', 'lead_force_z']
METRICS_SHA256 = 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(data, source_manifest):
    data = Path(data).resolve()
    frozen = json.loads(Path(source_manifest).read_text(encoding='utf-8'))['sources']
    checks = {}
    for relative, spec in frozen.items():
        path = (data/relative).resolve()
        if not path.is_relative_to(data):
            raise ValueError('source_outside_data')
        actual = file_digest(path)
        if actual != spec['sha256']:
            raise ValueError('changed_frozen_source:' + relative)
        checks[relative] = {'sha256': actual, 'verified': True}
    if not checks:
        raise ValueError('empty_source_manifest')
    return checks


def validate_metrics(metrics, metadata):
    if metadata.session_pitch.duplicated().any() or metadata.filename_new.duplicated().any():
        raise ValueError('duplicate_metadata_key')
    if metrics.duplicated(['session_pitch', 'cutoff_hz']).any():
        raise ValueError('duplicate_metric_key')
    if set(metrics.cutoff_hz) != set(CUTOFFS):
        raise ValueError('unexpected_cutoffs')
    if not np.isfinite(metrics[FEATURES + ['pitch_speed_kmh']].to_numpy(float)).all():
        raise ValueError('nonfinite_metrics')
    reference = metadata.set_index('session_pitch')
    if not set(metrics.session_pitch) <= set(reference.index):
        raise ValueError('unlinked_metric_pitch')
    for cutoff in CUTOFFS:
        frame = metrics[metrics.cutoff_hz == cutoff].set_index('session_pitch').sort_index()
        if cutoff == CUTOFFS[0]:
            pitches = set(frame.index)
        if set(frame.index) != pitches:
            raise ValueError('cutoff_cohort_mismatch')
        original = reference.loc[frame.index]
        if not np.array_equal(frame.user.to_numpy(str), original.user.to_numpy(str)):
            raise ValueError('player_mismatch')
        if not np.array_equal(frame.session.to_numpy(str), original.session.to_numpy(str)):
            raise ValueError('session_mismatch')
        if not np.allclose(frame.pitch_speed_kmh, original.pitch_speed_mph*1.609344,
                           rtol=0, atol=1e-10):
            raise ValueError('speed_unit_or_link_mismatch')
    primary = metrics[metrics.cutoff_hz == 20]
    if primary.groupby('user').session.nunique().max() != 1:
        raise ValueError('multiple_sessions_require_new_protocol')
    if primary.groupby('user').size().min() < 2:
        raise ValueError('insufficient_repeats')
    return dict(players=int(primary.user.nunique()), pitches=len(primary),
                sessions=int(primary.session.nunique()), cutoff_cohort_identical=True,
                speed_unit_and_player_link_verified=True)


def audit_grf(force, metadata, selected_pitches):
    if force.duplicated(['session_pitch', 'time']).any():
        raise ValueError('duplicate_grf_sample')
    if not np.isfinite(force.time).all():
        raise ValueError('nonfinite_grf_time')
    if not set(force.session_pitch) <= set(metadata.session_pitch):
        raise ValueError('unlinked_grf_pitch')
    records = []
    for pitch, frame in force.groupby('session_pitch', sort=True):
        events = {}
        for event in ('pkh_time', 'BR_time'):
            values = frame[event].unique()
            if len(values) != 1 or not np.isfinite(values[0]):
                raise ValueError('invalid_grf_event:' + pitch)
            events[event] = float(values[0])
        if events['pkh_time'] >= events['BR_time']:
            raise ValueError('reversed_grf_window')
        window = frame.time.between(events['pkh_time'], events['BR_time'])
        finite = np.isfinite(frame[GRF].to_numpy(float)).all(axis=1)
        records.append(dict(session_pitch=pitch, rows=len(frame), missing_rows=int((~finite).sum()),
                            all_rows_finite=bool(finite.all()),
                            window_rows=int(window.sum()),
                            pkh_br_finite=bool(window.any() and finite[window].all())))
    present = {r['session_pitch'] for r in records}
    all_finite = {r['session_pitch'] for r in records if r['all_rows_finite']}
    window_finite = {r['session_pitch'] for r in records if r['pkh_br_finite']}
    selected = set(selected_pitches)
    return dict(present_pitches=len(present), all_rows_finite_pitches=len(all_finite),
                pkh_br_finite_pitches=len(window_finite),
                missing_metadata_pitches=sorted(set(metadata.session_pitch)-present),
                selected_with_grf=len(selected & present),
                selected_all_rows_finite=len(selected & all_finite),
                selected_pkh_br_finite=len(selected & window_finite),
                selected_not_all_finite=sorted(selected-all_finite),
                missing_rows=sum(r['missing_rows'] for r in records),
                physical_contact_validated=False, records=records)


def audit_c3d(data, metadata):
    index = {}
    for path in (Path(data)/'c3d').rglob('*.c3d'):
        index.setdefault(path.name, []).append(path)
    counts = [len(index.get(name, [])) for name in metadata.filename_new]
    if any(count != 1 for count in counts):
        raise ValueError('missing_or_ambiguous_c3d')
    return dict(files=sum(map(len, index.values())), matched_dynamic_files=len(counts),
                missing_or_ambiguous=0, direct_ball_observation_validated=False)


def make_pairs(frame):
    if frame.session_pitch.duplicated().any():
        raise ValueError('duplicate_pair_pitch')
    records = []
    for player, group in frame.groupby('user', sort=True):
        group = group.sort_values('session_pitch')
        if len(group) < 2:
            raise ValueError('insufficient_pair_repeats')
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                a, b = group.iloc[i], group.iloc[j]
                row = dict(user=str(player), pitch_i=a.session_pitch, pitch_j=b.session_pitch,
                           truth_delta_kmh=float(b.pitch_speed_kmh-a.pitch_speed_kmh))
                row.update({name: float(b[name]-a[name]) for name in FEATURES})
                records.append(row)
    if not records:
        raise ValueError('empty_pairs')
    pairs = pd.DataFrame(records)
    if not np.isfinite(pairs[FEATURES + ['truth_delta_kmh']].to_numpy(float)).all():
        raise ValueError('nonfinite_pair')
    return pairs


def fit_pair_ridge(pairs, alpha=10.):
    if not np.isfinite(alpha) or alpha <= 0 or pairs.empty:
        raise ValueError('invalid_ridge_input')
    x, y = pairs[FEATURES].to_numpy(float), pairs.truth_delta_kmh.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite_ridge_input')
    weights = 1./pairs.groupby('user').user.transform('size').to_numpy(float)
    scale = np.sqrt(np.sum(weights[:, None]*x*x, axis=0)/weights.sum())
    scale[scale == 0] = 1.
    z = x/scale
    beta = np.linalg.solve(z.T@(weights[:, None]*z)+alpha*np.eye(len(FEATURES)),
                           z.T@(weights*y))
    return dict(scale=scale, beta=beta, alpha=alpha)


def predict_pairs(pairs, model):
    x = pairs[FEATURES].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite_prediction_input')
    return x/model['scale']@model['beta']


def pair_loao(pairs):
    if pairs.user.nunique() < 2:
        raise ValueError('insufficient_players_for_loao')
    output, splits = [], []
    for player in sorted(pairs.user.unique()):
        train, test = pairs[pairs.user != player], pairs[pairs.user == player]
        model = fit_pair_ridge(train)
        records = test[['user', 'pitch_i', 'pitch_j', 'truth_delta_kmh']].copy()
        records['predicted_pair_delta_kmh'] = predict_pairs(test, model)
        records['zero_delta_baseline_kmh'] = 0.
        output.append(records)
        splits.append(dict(heldout_user=player, train_players=int(train.user.nunique()),
                           test_pairs=len(test), train_pairs=len(train),
                           players_disjoint=player not in set(train.user)))
    return pd.concat(output, ignore_index=True), splits


def pair_summary(predictions, repetitions=2000, seed=20261029):
    frame = predictions.copy()
    frame['model_error'] = abs(frame.predicted_pair_delta_kmh-frame.truth_delta_kmh)
    frame['baseline_error'] = abs(frame.truth_delta_kmh)
    players = frame.groupby('user')[['model_error', 'baseline_error']].mean()
    delta = (players.model_error-players.baseline_error).to_numpy()
    if repetitions < 2 or not len(delta):
        raise ValueError('invalid_bootstrap')
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(repetitions, len(delta)))].mean(axis=1)
    return dict(players=len(players), pairs=len(frame),
                model_mae_kmh=float(players.model_error.mean()),
                zero_delta_mae_kmh=float(players.baseline_error.mean()),
                paired_mae_difference_kmh=float(delta.mean()),
                player_bootstrap_95_difference_kmh=np.quantile(means, [.025, .975]).tolist(),
                absolute_error_p95_kmh=float(np.quantile(frame.model_error, .95)),
                retrospective_observed_pair=True, prospective_validated=False,
                coaching_recommendation_ready=False)


def development_verdict(summaries, cohort, integrity_verified):
    primary = summaries['20']
    checks = dict(minimum_players=cohort['players'] >= 50, minimum_pitches=cohort['pitches'] >= 150,
                  integrity_verified=integrity_verified is True,
                  primary_better=primary['model_mae_kmh'] < primary['zero_delta_mae_kmh'],
                  player_bootstrap_upper_negative=primary['player_bootstrap_95_difference_kmh'][1] < 0,
                  other_cutoffs_not_worse=all(summaries[str(c)]['model_mae_kmh'] <=
                                             summaries[str(c)]['zero_delta_mae_kmh'] for c in (10, 30)))
    return dict(development_pairwise_criteria_pass=all(checks.values()), checks=checks,
                one_kmh_development_screen=bool(primary['model_mae_kmh'] < .5 and
                                               primary['absolute_error_p95_kmh'] < 1.),
                original_scientific_gate_changed=False, physical_constraints_validated=False,
                actual_athlete_effect_validated=False, coaching_recommendation_ready=False)
