"""Development-data prediction audit; not causal motion intervention evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from full_body_comparison import peak
from pilot import regular_time

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'openbiomechanics/baseball_pitching/data'
OUT = ROOT / 'artifacts/speed-association'
FEATURES = ['rear_knee_peak_br_ms', 'rotation_hip_shoulder_separation_fp',
            'stride_cm', 'max_pelvis_rotational_velo', 'max_torso_rotational_velo',
            'lead_knee_extension_from_fp_to_br']


def ridge_fit(x, y, alpha=10.):
    """No intercept: inputs/targets are within-session centered training data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    scale = np.sqrt(np.mean(x*x, axis=0))
    scale = np.where(scale > 1e-10, scale, 1.)
    z = x/scale
    return np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@y)/scale


def load():
    meta = pd.read_csv(DATA/'metadata.csv')
    poi = pd.read_csv(DATA/'poi/poi_metrics.csv')
    joined = meta.merge(poi, on='session_pitch', validate='one_to_one', suffixes=('', '_poi'))
    discrepancy = np.abs(joined.pitch_speed_mph-joined.pitch_speed_mph_poi)
    if not np.all(discrepancy < .001):
        raise ValueError('Metadata/POI speed disagreement exceeds tolerance')
    print('pitch types:', joined.pitch_type.value_counts().to_dict(), flush=True)
    # Source labels are explicit; do not infer fastballs from velocity.
    joined = joined[(joined.p_throws == 'R') & (joined.pitch_type.str.lower().isin(['fastball', 'ff', 'fb']))].copy()
    joined['speed_kmh'] = joined.pitch_speed_mph*1.609344
    joined['stride_cm'] = joined.stride_length*joined.session_height_m*100
    wanted = set(joined.session_pitch)
    chunks = []
    for chunk in pd.read_csv(DATA/'full_sig/joint_velos.csv', chunksize=50000):
        chunks.append(chunk[chunk.session_pitch.isin(wanted)])
    velocities = pd.concat(chunks, ignore_index=True)
    timing, invalid_events = [], []
    for pid, g in velocities.groupby('session_pitch'):
        g = g.sort_values('time'); t = regular_time(g)
        fp, br = float(g.fp_10_time.iloc[0]), float(g.BR_time.iloc[0])
        if not np.isfinite([fp, br]).all() or fp >= br or fp < t[0] or br > t[-1]:
            invalid_events.append(dict(pitch=pid, reason='missing or invalid FP10/BR interval'))
            continue
        pt, _ = peak(t, g.rear_knee_velo_x.to_numpy(float), fp, br)
        timing.append({'session_pitch': pid, 'rear_knee_peak_br_ms': (pt-br)*1000,
                       'knee_boundary_peak': min(abs(pt-fp), abs(pt-br)) <= 1/360+.000051})
    joined = joined.merge(pd.DataFrame(timing), on='session_pitch', validate='one_to_one')
    before = len(joined)
    joined = joined[np.isfinite(joined[FEATURES+['speed_kmh']]).all(axis=1)].copy()
    joined = joined.groupby(['user', 'session']).filter(lambda g: len(g) >= 2)
    return joined, dict(matched_metadata_poi=len(meta), max_speed_table_difference_mph=float(discrepancy.max()),
                        right_fastball_before_complete_case=before, eligible_rows=len(joined),
                        athletes=int(joined.user.nunique()), sessions=int(joined.session.nunique()), invalid_events=invalid_events)


def evaluate(frame):
    models = {f: [f] for f in FEATURES}
    models['all_six_fixed_ridge'] = FEATURES
    predictions = []
    for athlete in sorted(frame.user.unique()):
        train = frame[frame.user != athlete]
        centered = train[FEATURES+['speed_kmh']] - train.groupby(['user','session'])[FEATURES+['speed_kmh']].transform('mean')
        coefficients = {name: ridge_fit(centered[cols], centered.speed_kmh) for name, cols in models.items()}
        for session, group in frame[frame.user == athlete].groupby('session'):
            group = group.sort_values('session_pitch')
            anchor = group.iloc[0]  # lexical reference, not claimed to be chronological
            for _, row in group.iloc[1:].iterrows():
                actual = float(row.speed_kmh-anchor.speed_kmh)
                for name, cols in models.items():
                    predicted = float((row[cols].to_numpy(float)-anchor[cols].to_numpy(float))@coefficients[name])
                    predictions.append(dict(user=int(athlete), session=int(session), anchor=anchor.session_pitch,
                        pitch=row.session_pitch, model=name, actual_delta_kmh=actual, predicted_delta_kmh=predicted,
                        absolute_error_kmh=abs(predicted-actual), no_change_error_kmh=abs(actual)))
    p = pd.DataFrame(predictions)
    summaries = []
    for name, g in p.groupby('model'):
        by = g.groupby('user')[['absolute_error_kmh','no_change_error_kmh']].mean()
        summaries.append(dict(model=name, test_pitches=len(g), athletes=len(by),
            mae_kmh=float(g.absolute_error_kmh.mean()), no_change_mae_kmh=float(g.no_change_error_kmh.mean()),
            athlete_macro_mae_kmh=float(by.absolute_error_kmh.mean()),
            athlete_macro_no_change_mae_kmh=float(by.no_change_error_kmh.mean()),
            athletes_improved=int((by.absolute_error_kmh < by.no_change_error_kmh).sum())))
    return p, summaries


def personal_audit(frame):
    g = frame[(frame.user == 2) & (frame.session == 3034)].sort_values('session_pitch')
    rows = []
    for feature in FEATURES:
        x, y = g[feature].to_numpy(float), g.speed_kmh.to_numpy(float)
        errors, baseline = [], []
        for i in range(len(g)):
            mask = np.arange(len(g)) != i
            xm, ym = x[mask].mean(), y[mask].mean()
            slope = np.linalg.lstsq((x[mask]-xm)[:,None], y[mask]-ym, rcond=None)[0][0]
            errors.append(abs(ym+slope*(x[i]-xm)-y[i])); baseline.append(abs(ym-y[i]))
        rows.append(dict(feature=feature, pearson_r=float(np.corrcoef(x,y)[0,1]),
                         leave_one_pitch_out_mae_kmh=float(np.mean(errors)), training_mean_mae_kmh=float(np.mean(baseline))))
    return g, rows


def personalized_cohort(frame):
    rows = []
    for (user, session), g in frame.groupby(['user', 'session']):
        if len(g) < 4:
            continue
        y = g.speed_kmh.to_numpy(float)
        for feature in FEATURES:
            x = g[feature].to_numpy(float)
            for i in range(len(g)):
                mask = np.arange(len(g)) != i
                xm, ym = x[mask].mean(), y[mask].mean()
                slope = np.linalg.lstsq((x[mask]-xm)[:,None], y[mask]-ym, rcond=None)[0][0]
                regularized = ridge_fit((x[mask]-xm)[:,None], y[mask]-ym)[0]
                for method, coefficient in [('OLS', slope), ('fixed_ridge_10', regularized)]:
                    rows.append(dict(user=int(user), session=int(session), pitch=g.iloc[i].session_pitch,
                        feature=feature, method=method, error=abs(ym+coefficient*(x[i]-xm)-y[i]), baseline=abs(ym-y[i])))
    results = pd.DataFrame(rows)
    summaries = []
    for (feature, method), g in results.groupby(['feature', 'method']):
        by = g.groupby('user')[['error','baseline']].mean()
        summaries.append(dict(feature=feature, method=method, n_pitches=len(g), athletes=len(by),
            mae_kmh=float(g.error.mean()), training_mean_mae_kmh=float(g.baseline.mean()),
            athletes_improved=int((by.error < by.baseline).sum())))
    return summaries


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, counts = load()
    predictions, summary = evaluate(frame)
    personal, personal_results = personal_audit(frame)
    predictions.to_csv(OUT/'excluded_athlete_predictions.csv', index=False)
    personal[['session_pitch','speed_kmh']+FEATURES].to_csv(OUT/'athlete2.csv', index=False)
    report = dict(counts=counts, models=summary, athlete2=personal_results,
        personalized_cohort=personalized_cohort(frame),
        ridge_alpha=10, alpha_tuned=False, features_previously_selected_on_development_data=True,
        scope='Retrospective development audit, excluded-athlete slopes with measured anchor speed; not blind external validation, causal effect, or Real2Sim result.',
        boundary_knee_pitches=int(frame.knee_boundary_peak.sum()),
        sources={str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [DATA/'metadata.csv', DATA/'poi/poi_metrics.csv', DATA/'full_sig/joint_velos.csv']})
    (OUT/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
