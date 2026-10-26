"""Independent read-only replication from original CSVs, not analysis helpers."""
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'openbiomechanics/baseball_pitching/data'
OUT = ROOT / 'artifacts/within-athlete'

def main():
    report = json.loads((OUT/'report.json').read_text())
    features = pd.read_csv(OUT/'pitch_features.csv').set_index('session_pitch')
    extrema = pd.read_csv(OUT/'fast_slow_extrema.csv')
    pairs = pd.read_csv(OUT/'all_distinct_speed_pairs.csv')
    raw = pd.read_csv(DATA/'metadata.csv').set_index('session_pitch')
    poi = pd.read_csv(DATA/'poi/poi_metrics.csv').set_index('session_pitch')
    signals = pd.read_csv(DATA/'full_sig/joint_velos.csv')
    checks = []
    def check(name, good):
        assert good, name
        checks.append(name)
    check('every output speed and athlete equals raw metadata', all(
        row.user == raw.loc[pitch,'user'] and row.session == raw.loc[pitch,'session']
        and row.pitch_speed_mph == raw.loc[pitch,'pitch_speed_mph']
        for pitch,row in features.iterrows()))
    check('metadata and POI speed agree', np.allclose(raw.pitch_speed_mph,
        poi.loc[raw.index,'pitch_speed_mph'],rtol=0,atol=1e-12))
    check('athlete partition disjoint', set(features[features.split=='discovery'].user).isdisjoint(
        set(features[features.split=='evaluation'].user)))
    check('partition parity', all((r.split=='discovery')==(int(r.user)%2==0) for _,r in features.iterrows()))
    error = 0.; invalid = []
    cols = {'pelvis_peak_ms':'pelvis_velo_z','torso_peak_ms':'torso_velo_z',
            'elbow_peak_ms':'elbow_velo_x','shoulder_peak_ms':'shoulder_velo_z'}
    for pitch,g in signals.groupby('session_pitch'):
        g = g.sort_values('time')
        fp,br = g.iloc[0][['fp_10_time','BR_time']]
        if not np.isfinite([fp,br]).all() or br<=fp:
            invalid.append(pitch); continue
        if pitch not in features.index: continue
        w = g[(g.time>=fp-0.000051)&(g.time<=br+0.000051)]
        actual = {'fp_to_br_ms':1000*(br-fp)}
        for name,col in cols.items():
            index = np.argmax(np.abs(w[col].to_numpy()))
            actual[name] = 1000*(w.time.to_numpy()[index]-br)
        index = np.argmax(np.sqrt(np.sum(w[['wrist_velo_x','wrist_velo_y','wrist_velo_z']].to_numpy()**2,axis=1)))
        actual['wrist_peak_ms'] = 1000*(w.time.to_numpy()[index]-br)
        actual['torso_minus_pelvis_peak_ms'] = actual['torso_peak_ms']-actual['pelvis_peak_ms']
        for name in actual:
            error = max(error,abs(actual[name]-features.loc[pitch,name]))
    check('raw timing replication', error<1e-9)
    check('invalid event list replication', sorted(invalid)==sorted(x['pitch'] for x in report['excluded_pitches']))
    expected_pairs = set(); expected_extrema = set()
    for _,g in features.reset_index().groupby(['user','session','pitch_type']):
        g=g.sort_values(['pitch_speed_mph','session_pitch'])
        rows=list(g.itertuples())
        if len(rows)<2 or rows[0].pitch_speed_mph==rows[-1].pitch_speed_mph: continue
        expected_extrema.add((rows[0].session_pitch,rows[-1].session_pitch))
        for a,b in combinations(rows,2):
            if a.pitch_speed_mph!=b.pitch_speed_mph:expected_pairs.add((a.session_pitch,b.session_pitch))
    check('all distinct-speed pair IDs exact', expected_pairs==set(zip(pairs.slow_pitch,pairs.fast_pitch)))
    check('fast/slow extrema IDs exact', expected_extrema==set(zip(extrema.slow_pitch,extrema.fast_pitch)))
    for row in report['ranked_features']:
        f=row['feature']; train=extrema[extrema.split=='discovery'][f]
        check('direction '+f, int(np.sign(train.mean()))==row['discovery_direction'])
        for split in ['discovery','evaluation']:
            x=extrema[extrema.split==split][f]
            check(split+' mean '+f, abs(x.mean()-row[split]['mean_fast_minus_slow'])<1e-9)
            fraction=np.mean(x*row['discovery_direction']>1e-8)
            check(split+' sign '+f, abs(fraction-row[split]['same_discovery_direction_fraction_all'])<1e-12)
    result={'checks_passed':len(checks),'max_raw_timing_replication_difference_ms':error,
            'rows':len(features),'extrema':len(extrema),'all_pairs':len(pairs),
            'all_checks':checks,'causal_identification_verified':False}
    (OUT/'independent-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='all_checks'},indent=2))

if __name__=='__main__':main()
