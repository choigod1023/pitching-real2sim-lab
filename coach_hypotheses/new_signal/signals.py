"""Read fixed observed force/power waveforms; do not infer physical capacity."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from ..research_reset.validation import (FEATURES, IDS, METRICS_SHA256, CUTOFFS,
                                         GRF, file_digest, validate_metrics, verify_sources)

POWER = ['pelvis_leadhip_seg_pwr', 'pelvis_rearhip_seg_pwr', 'pelvis_thorax_seg_pwr',
         'thorax_prox_seg_pwr', 'thorax_dist_seg_pwr']
FORCE_NAMES = ['rear_norm', 'lead_norm', 'lead_share']
ADDED = [f'power_{name}_{i}' for name in POWER for i in range(5)] + [
    f'grf_{name}_{i}' for name in FORCE_NAMES for i in range(5)] + ['fp_br_duration_s']
NEW_FEATURES = FEATURES + ADDED
SOURCE_HASHES = {
    'energy_flow.csv': 'e68e0def4b35900064240f7c93a33df7a58e07ccd75b36628d6fff91236f766e',
    'force_plate.csv': '2e7bd243b08fc1876c1f5db805c2bc920876fdc651323f13ce03869aed690d19'}


def event_window(frame):
    result = []
    for name in ('fp_10_time', 'BR_time'):
        values = frame[name].unique()
        if len(values) != 1 or not np.isfinite(values[0]):
            raise ValueError('invalid_event:' + name)
        result.append(float(values[0]))
    if result[0] >= result[1]:
        raise ValueError('reversed_event_window')
    return tuple(result)


def phase_samples(frame, columns, window, cutoff, rate):
    """Filter only a contiguous finite block, with explicit 15-sample padding."""
    if frame.time.duplicated().any():
        raise ValueError('duplicate_time')
    ordered = frame.sort_values('time')
    time = ordered.time.to_numpy(float)
    values = ordered[columns].to_numpy(float)
    if not np.isfinite(time).all() or len(time) < 2:
        raise ValueError('invalid_time')
    if np.max(np.abs(np.diff(time)-1/rate)) > 1.1e-4:
        raise ValueError('unexpected_sampling_grid')
    start = np.searchsorted(time, window[0], side='right')-1
    end = np.searchsorted(time, window[1], side='left')
    if start < 0 or end >= len(time):
        raise ValueError('unbracketed_window')
    valid = np.isfinite(values).all(axis=1)
    if not valid[start:end+1].all():
        raise ValueError('missing_window_signal')
    left, right = start, end
    while left > 0 and valid[left-1]:
        left -= 1
    while right+1 < len(time) and valid[right+1]:
        right += 1
    if right-left+1 <= 15:
        raise ValueError('insufficient_filter_support')
    sos = butter(4, cutoff, fs=rate, output='sos')
    filtered = sosfiltfilt(sos, values[left:right+1], axis=0, padlen=15)
    phase = np.linspace(*window, 5)
    sampled = np.column_stack([np.interp(phase, time[left:right+1], filtered[:, i])
                               for i in range(len(columns))])
    return sampled, dict(filter_samples=int(right-left+1), left_time_s=float(time[left]),
                          right_time_s=float(time[right]), rate_hz=rate)


def force_representation(components):
    components = np.asarray(components, float)
    if components.ndim != 2 or components.shape[1] != 6 or not np.isfinite(components).all():
        raise ValueError('invalid_force_components')
    rear = np.linalg.norm(components[:, :3], axis=1)
    lead = np.linalg.norm(components[:, 3:], axis=1)
    total = rear+lead
    share = np.divide(lead, total, out=np.zeros_like(lead), where=total > 0)
    return np.column_stack([rear, lead, share]), int((total == 0).sum())


def load_features(data, metrics_path, source_manifest):
    data = Path(data)
    if file_digest(metrics_path) != METRICS_SHA256:
        raise ValueError('changed_metrics')
    hashes = verify_sources(data, source_manifest)
    for name, expected in SOURCE_HASHES.items():
        actual = file_digest(data/'full_sig'/name)
        if actual != expected:
            raise ValueError('changed_source:' + name)
        hashes['full_sig/'+name] = {'sha256': actual}
    metrics = pd.read_csv(metrics_path, dtype=IDS)
    metadata = pd.read_csv(data/'metadata.csv', dtype=IDS)
    validate_metrics(metrics, metadata)
    energy = pd.read_csv(data/'full_sig/energy_flow.csv', dtype={'session_pitch': str},
                         usecols=['session_pitch', 'time', 'fp_10_time', 'BR_time'] + POWER)
    force = pd.read_csv(data/'full_sig/force_plate.csv', dtype={'session_pitch': str})
    energies, forces = dict(tuple(energy.groupby('session_pitch'))), dict(tuple(force.groupby('session_pitch')))
    tables, exclusions, support = {}, [], []
    for cutoff in CUTOFFS:
        records = []
        for _, row in metrics[metrics.cutoff_hz == cutoff].iterrows():
            pitch = row.session_pitch
            try:
                if pitch not in energies or pitch not in forces:
                    raise ValueError('missing_source_pitch')
                e, f = energies[pitch], forces[pitch]
                window = event_window(e)
                if not np.allclose(window, event_window(f), rtol=0, atol=1e-6):
                    raise ValueError('event_file_mismatch')
                powers, ep = phase_samples(e, POWER, window, cutoff, 360.)
                components, fp = phase_samples(f, GRF, window, cutoff, 1080.)
                force_values, zero = force_representation(components)
                additions = np.r_[powers.T.ravel(), force_values.T.ravel(), window[1]-window[0]]
                record = row.to_dict()
                record.update(zip(ADDED, additions))
                records.append(record)
                support.append(dict(session_pitch=pitch, cutoff_hz=cutoff,
                                    power=ep, force=fp, zero_support_phase_samples=zero))
            except ValueError as error:
                exclusions.append(dict(session_pitch=pitch, cutoff_hz=cutoff, reason=str(error)))
        if not records:
            raise ValueError('no_eligible_new_signal_data')
        tables[cutoff] = pd.DataFrame(records)
    common = set.intersection(*(set(t.session_pitch) for t in tables.values()))
    primary = tables[20][tables[20].session_pitch.isin(common)]
    eligible_players = set(primary.groupby('user').size().loc[lambda x: x >= 2].index)
    for cutoff in CUTOFFS:
        tables[cutoff] = tables[cutoff][tables[cutoff].session_pitch.isin(common) &
                                        tables[cutoff].user.isin(eligible_players)].sort_values(['user', 'session_pitch']).reset_index(drop=True)
    primary = tables[20]
    excluded_common = sorted(set(metrics.session_pitch)-set(primary.session_pitch))
    audit = dict(players=int(primary.user.nunique()), pitches=len(primary),
                 candidate_pitches=int(metrics.session_pitch.nunique()), exclusions=exclusions,
                 common_excluded_pitches=excluded_common, sources=hashes,
                 support=support, source_power_units_independently_validated=False,
                 physical_contact_or_work_budget_validated=False)
    return tables, audit
