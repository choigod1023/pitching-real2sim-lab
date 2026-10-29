"""Frozen-cohort within-player comparisons, without fitting a speed model."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL_COMMIT = '2cab00c9474dbc84950292668e76efd60cf3ea81'
SIGN_AMENDMENT_COMMIT = '5175773c38e4fac7e41392cb1c652d30f5446542'
METRICS_SHA256 = 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4'
CUTOFFS = (10, 20, 30)
LOADS = ('elbow_varus_peak_nm', 'shoulder_resultant_peak_n',
         'shoulder_distraction_peak_n')
TENSION = 'shoulder_tension_peak_magnitude_n'
MOTION = ('sequence_lag_ms', 'lead_knee_extension_deg',
          'hip_shoulder_sep_fp10_deg', 'wrist_height_br_m',
          'arm_slot_poi_deg', 'stride_reported')
EPSILON = 1e-9


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_frame(frame, metadata, poi):
    """Reject ambiguous identities, incomplete filters and changed speed links."""
    frame = frame.copy()
    numeric = ['cutoff_hz', 'pitch_speed_kmh', *LOADS, *MOTION]
    keys = ['user', 'session', 'session_pitch']
    if not set(keys + numeric).issubset(frame):
        raise ValueError('missing_metrics_columns')
    if frame.empty or frame[keys].isna().any().any():
        raise ValueError('empty_or_missing_identity')
    frame[keys] = frame[keys].astype(str)
    if frame[keys].eq('').any().any() or not frame.user.str.fullmatch(r'\d+').all():
        raise ValueError('invalid_identity')
    if frame.duplicated(['session_pitch', 'cutoff_hz']).any():
        raise ValueError('duplicate_pitch_filter')
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError('nonfinite_metric')
    if (frame[['pitch_speed_kmh', *LOADS[:2]]].to_numpy(float) < 0).any():
        raise ValueError('negative_speed_or_peak_load')
    if not frame.cutoff_hz.isin(CUTOFFS).all():
        raise ValueError('unexpected_filter')
    if frame.groupby('session_pitch').cutoff_hz.nunique().ne(3).any():
        raise ValueError('incomplete_filter_support')
    for column in [*keys[:-1], 'pitch_speed_kmh']:
        if frame.groupby('session_pitch')[column].nunique().ne(1).any():
            raise ValueError('filter_identity_or_speed_mismatch')
    if (frame.groupby('user').session.nunique().ne(1).any() or
            frame.groupby('session').user.nunique().ne(1).any()):
        raise ValueError('ambiguous_player_session')
    nominal = frame[frame.cutoff_hz == 20]
    if nominal.groupby('user').size().lt(2).any():
        raise ValueError('insufficient_player_repeats')
    for source in (metadata, poi):
        if ('session_pitch' not in source or source.session_pitch.isna().any() or
                source.session_pitch.duplicated().any()):
            raise ValueError('ambiguous_source_pitch')
    meta = metadata.set_index('session_pitch')
    reference = poi.set_index('session_pitch')
    selected = nominal.session_pitch.tolist()
    if not set(selected).issubset(meta.index) or not set(selected).issubset(reference.index):
        raise ValueError('missing_source_link')
    for row in nominal.itertuples(index=False):
        m, p = meta.loc[row.session_pitch], reference.loc[row.session_pitch]
        if (str(m['user']) != row.user or str(m['session']) != row.session or
                str(p['session']) != row.session or p['pitch_type'] != 'FF'):
            raise ValueError('source_identity_or_pitch_type_mismatch')
        mph = np.array([m['pitch_speed_mph'], p['pitch_speed_mph']], float)
        if (not np.isfinite(mph).all() or
                not np.allclose(mph * 1.609344, row.pitch_speed_kmh, atol=1e-6, rtol=0)):
            raise ValueError('source_speed_mismatch')
    return frame.sort_values(['user', 'session_pitch', 'cutoff_hz']).reset_index(drop=True)


def load_inputs(metrics_dir, data_dir):
    """Reuse frozen derived metrics and keep upstream exclusions and hashes."""
    metrics_dir, data_dir = Path(metrics_dir), Path(data_dir)
    path = metrics_dir / 'results/pitch_metrics.csv'
    if file_digest(path) != METRICS_SHA256:
        raise ValueError('changed_frozen_metrics')
    upstream = json.loads((metrics_dir / 'results/manifest.json').read_text(encoding='utf-8'))
    exclusions = json.loads((metrics_dir / 'results/exclusions.json').read_text(encoding='utf-8'))
    sources = {'derived_metrics': {'path': str(path.resolve()), 'sha256': file_digest(path)}}
    for name in ('metadata.csv', 'poi/poi_metrics.csv'):
        source = data_dir / name
        digest = file_digest(source)
        if digest != upstream['sources'][name]['sha256']:
            raise ValueError('changed_upstream_source:' + name)
        sources[name] = {'path': str(source.resolve()), 'sha256': digest}
    identity = {'user': str, 'session': str, 'session_pitch': str}
    frame = validate_frame(pd.read_csv(path, dtype=identity),
                           pd.read_csv(data_dir / 'metadata.csv', dtype=identity),
                           pd.read_csv(data_dir / 'poi/poi_metrics.csv', dtype=identity))
    if len(frame) != 1197 or frame.user.nunique() != 98 or frame.session_pitch.nunique() != 399:
        raise ValueError('unexpected_frozen_cohort')
    if set(frame.session_pitch) & {row['session_pitch'] for row in exclusions}:
        raise ValueError('excluded_pitch_reintroduced')
    upstream_hashes = {name: file_digest(metrics_dir / name) for name in (
        'results/manifest.json', 'results/exclusions.json', 'PROTOCOL.md',
        'AMENDMENT_01_KNEE_SIGN.md')}
    documentation = data_dir.parent / 'README.md'
    return frame, dict(sources=sources, upstream_manifest=upstream,
                       upstream_files_sha256=upstream_hashes,
                       upstream_exclusions=exclusions, new_exclusions=[],
                       source_documentation=dict(path=str(documentation.resolve()),
                                                 sha256=file_digest(documentation),
                                                 shoulder_z_convention='Compression positive / Distraction negative'),
                       upstream_metrics_directory=str(metrics_dir.resolve()))


def classify_loads(differences, tied=False):
    """Numerical nonincrease is not physiological or measurement equivalence."""
    values = np.asarray(differences, float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('need_three_finite_load_differences')
    if tied:
        return 'tied_speed'
    if np.all(values <= EPSILON):
        return 'observed_nonincrease'
    if np.all(values > EPSILON):
        return 'all_loads_increase'
    return 'mixed_load_tradeoff'


def make_comparisons(frame):
    """All unordered pairs, oriented by recorded speed for description only."""
    frame = frame.copy()
    frame[TENSION] = np.maximum(0., frame[LOADS[2]].to_numpy(float))
    records = []
    for user, player in frame.groupby('user', sort=True):
        by_pitch = dict(tuple(player.groupby('session_pitch', sort=True)))
        for first, second in itertools.combinations(sorted(by_pitch), 2):
            a, b = by_pitch[first].set_index('cutoff_hz'), by_pitch[second].set_index('cutoff_hz')
            sa, sb = float(a.loc[20, 'pitch_speed_kmh']), float(b.loc[20, 'pitch_speed_kmh'])
            tied = sa == sb
            faster, slower = (second, first) if sb > sa else (first, second)
            high, low = (b, a) if sb > sa else (a, b)
            for cutoff in CUTOFFS:
                differences = {name: float(high.loc[cutoff, name] - low.loc[cutoff, name])
                               for name in (*LOADS, TENSION, *MOTION)}
                record = dict(user=str(user), session=str(a.loc[cutoff, 'session']),
                              pitch_i=first, pitch_j=second, cutoff_hz=cutoff,
                              faster_pitch=None if tied else faster,
                              slower_pitch=None if tied else slower,
                              higher_recorded_speed_kmh=max(sa, sb),
                              lower_recorded_speed_kmh=min(sa, sb),
                              recorded_speed_gap_kmh=abs(sb - sa), tied_speed=tied,
                              raw_signed_status=classify_loads([differences[name] for name in LOADS], tied),
                              status=classify_loads([differences[name] for name in (*LOADS[:2], TENSION)], tied))
                record.update({name + '_delta': value for name, value in differences.items()})
                for name in (*LOADS, TENSION):
                    record[name + '_higher_speed_pitch'] = float(high.loc[cutoff, name])
                    record[name + '_lower_speed_pitch'] = float(low.loc[cutoff, name])
                records.append(record)
    if not records:
        raise ValueError('empty_comparisons')
    return pd.DataFrame(records)


def bootstrap_player_mean(values, repetitions=2000):
    """Player resampling only; this interval excludes measurement uncertainty."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or repetitions < 1:
        raise ValueError('invalid_player_bootstrap')
    rng = np.random.default_rng(20261029)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def summarize_comparisons(comparisons):
    """Describe filter stability and retain mixed/tied/nonqualifying pairs."""
    keys = ['user', 'pitch_i', 'pitch_j']
    grouped = comparisons.groupby(keys, sort=True)
    if (grouped.size().ne(3).any() or grouped.cutoff_hz.nunique().ne(3).any() or
            comparisons.duplicated(keys + ['cutoff_hz']).any()):
        raise ValueError('incomplete_comparison_filters')
    pairs = grouped.agg(recorded_speed_gap_kmh=('recorded_speed_gap_kmh', 'first'),
                        qualifying_filters=('status', lambda s: int(s.eq('observed_nonincrease').sum())),
                        raw_signed_qualifying_filters=('raw_signed_status', lambda s: int(s.eq('observed_nonincrease').sum())),
                        status_count=('status', 'nunique')).reset_index()
    pairs['filter_stable_observed_nonincrease'] = pairs.qualifying_filters.eq(3)
    pairs['raw_signed_filter_stable_nonincrease'] = pairs.raw_signed_qualifying_filters.eq(3)
    positive = pairs.recorded_speed_gap_kmh > 0
    rates = []
    for user, rows in pairs.groupby('user', sort=True):
        valid = rows[rows.recorded_speed_gap_kmh > 0]
        rates.append(dict(user=str(user), non_tied_pairs=len(valid),
                          stable_pairs=int(valid.filter_stable_observed_nonincrease.sum()),
                          stable_fraction=None if valid.empty else float(valid.filter_stable_observed_nonincrease.mean())))
    defined = [r['stable_fraction'] for r in rates if r['stable_fraction'] is not None]
    stable = pairs[pairs.filter_stable_observed_nonincrease]
    selected = stable[stable.recorded_speed_gap_kmh >= 1.]
    if selected.empty:
        selected = stable
    selected = sorted(selected.to_dict('records'), key=lambda r: (int(r['user']), r['pitch_i'], r['pitch_j']))[:3]
    examples = []
    for selected_pair in selected:
        mask = np.logical_and.reduce([comparisons[key].eq(selected_pair[key]).to_numpy() for key in keys])
        examples.append(dict(**selected_pair, filters=comparisons[mask].sort_values('cutoff_hz').to_dict('records')))
    counts = {str(cutoff): comparisons[comparisons.cutoff_hz == cutoff].status.value_counts().to_dict()
              for cutoff in CUTOFFS}
    result = dict(players=len(rates), unordered_pairs=len(pairs), filter_pair_rows=len(comparisons),
                  non_tied_pairs=int(positive.sum()), tied_speed_pairs=int((~positive).sum()),
                  status_counts_per_filter=counts,
                  raw_signed_status_counts_per_filter={str(c): comparisons[comparisons.cutoff_hz == c].raw_signed_status.value_counts().to_dict() for c in CUTOFFS},
                  raw_signed_stable_nonincrease_pairs=int(pairs.raw_signed_filter_stable_nonincrease.sum()),
                  amended_vs_raw_stable_disagreement_pairs=int(pairs.filter_stable_observed_nonincrease.ne(pairs.raw_signed_filter_stable_nonincrease).sum()),
                  filter_stable_nonincrease_pairs=len(stable),
                  filter_stable_nonincrease_players=int(stable.user.nunique()),
                  gap_at_least_1kmh_pairs=int((pairs.recorded_speed_gap_kmh >= 1.).sum()),
                  stable_gap_at_least_1kmh_pairs=int((stable.recorded_speed_gap_kmh >= 1.).sum()),
                  status_disagreement_pairs=int(pairs.status_count.gt(1).sum()),
                  player_balanced_stable_fraction=None if not defined else float(np.mean(defined)),
                  player_cluster_sampling_ci95=None if not defined else bootstrap_player_mean(defined),
                  players_without_non_tied_pairs=[r['user'] for r in rates if r['stable_fraction'] is None],
                  player_rates=rates, examples=examples,
                  observational_finding='PRESENT' if len(stable) else 'ABSENT',
                  measurement_robustness_validated=False, same_capacity_validated=False,
                  intervention_effect_validated=False, coaching_recommendation_ready=False,
                  speed_prediction_gate_changed=False, model_fitted=False)
    return result, pairs


def run_analysis(metrics_dir, data_dir, output):
    frame, provenance = load_inputs(metrics_dir, data_dir)
    comparisons = make_comparisons(frame)
    summary, pairs = summarize_comparisons(comparisons)
    summary['negative_signed_shoulder_extreme_rows'] = int(frame[LOADS[2]].lt(0).sum())
    if len(pairs) != 666 or len(comparisons) != 1998:
        raise ValueError('unexpected_pair_count')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol_commit=PROTOCOL_COMMIT,
                    sign_amendment_commit=SIGN_AMENDMENT_COMMIT,
                    sign_amendment_sha256=file_digest(Path(__file__).with_name('AMENDMENT_01_SIGNED_TENSION.md')),
                    protocol_sha256=file_digest(Path(__file__).with_name('PROTOCOL.md')),
                    code_sha256=file_digest(__file__), **provenance,
                    filtering='Provided full_sig 20Hz + additional zero-phase 10/20/30Hz',
                    epsilon_scope='floating comparison only, not measurement uncertainty',
                    data_license='CC BY-NC-SA 4.0',
                    source_url='https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching')
    comparisons.to_csv(output / 'all_pair_filter_comparisons.csv', index=False)
    pairs.to_csv(output / 'all_pair_stability.csv', index=False)
    for name, content in [('summary.json', summary), ('manifest.json', manifest)]:
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-dir', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must remain in observed_tradeoffs worktree package')
    summary = run_analysis(args.metrics_dir, args.data_dir, args.output)
    print(json.dumps({k: summary[k] for k in ('players', 'unordered_pairs', 'filter_stable_nonincrease_pairs',
                                             'stable_gap_at_least_1kmh_pairs', 'observational_finding')}, indent=2))


if __name__ == '__main__':
    main()
