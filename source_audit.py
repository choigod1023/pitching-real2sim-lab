"""Audit public C3D provenance; no speed-target parameter fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import ezc3d
import numpy as np
import pandas as pd


def event_summary(c):
    group = c['parameters'].get('EVENT', {})
    used = group.get('USED', {}).get('value', [0])
    header = c['header'].get('events', {})
    labels = [str(x).strip() for x in header.get('events_label', [])]
    times = header.get('events_time', [])
    return {'event_group_present': bool(group), 'event_used': int(used[0]) if len(used) else 0,
            'header_named_events': sum(bool(x) for x in labels),
            'header_nonzero_times': sum(float(x) != 0 for x in times)}


def hand_quality(c, side):
    labels = [x.strip() for x in c['parameters']['POINT']['LABELS']['value']]
    required = [side + x for x in ('FIN', 'WRA', 'WRB')]
    if any(x not in labels for x in required):
        return {'valid_frames': 0, 'reason': 'missing_label'}
    unit = c['parameters']['POINT']['UNITS']['value'][0].lower()
    if unit not in ('m', 'mm'):
        return {'valid_frames': 0, 'reason': 'unsupported_unit:' + unit}
    points = c['data']['points'][:, [labels.index(x) for x in required], :]
    xyz = points[:3].transpose(2, 1, 0) * (0.001 if unit == 'mm' else 1.)
    valid = np.isfinite(xyz).all(axis=(1, 2)) & (points[3] >= 0).all(axis=0)
    area2 = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)
    valid &= area2 > 1e-8
    return {'valid_frames': int(valid.sum()), 'total_frames': len(valid),
            'reason': 'ok' if valid.any() else 'invalid_or_degenerate'}


def audit(root, out):
    data = root / 'baseball_pitching/data'
    metadata = pd.read_csv(data / 'metadata.csv')
    paths = {}
    for p in sorted((data / 'c3d').rglob('*.c3d')):
        if p.name in paths:
            raise ValueError('Duplicate C3D basename: ' + p.name)
        paths[p.name] = p
    rows, failures, all_labels = [], [], set()
    models = set(metadata.modelname_new)
    names = sorted(set(metadata.filename_new) | models)
    for name in names:
        try:
            p = paths[name]
            c = ezc3d.c3d(str(p))
            labels = c['parameters']['POINT']['LABELS']['value']
            all_labels.update(labels)
            row = {'file': name, 'static': name in models,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), **event_summary(c),
                   'explicit_ball_labels': [x for x in labels if 'ball' in x.lower()],
                   'right_hand': hand_quality(c, 'R'), 'left_hand': hand_quality(c, 'L')}
            rows.append(row)
        except Exception as exc:
            failures.append({'file': name, 'error': str(exc)})
    mismatches = []
    for row in metadata.itertuples():
        encoded = int(Path(row.filename_new).stem.rsplit('_', 1)[1]) / 10
        if not np.isclose(encoded, row.pitch_speed_mph, rtol=0, atol=1e-8):
            mismatches.append(row.filename_new)
    summary = {'metadata_pitches': len(metadata), 'referenced_models': len(models),
               'read_files': len(rows), 'failures': failures, 'filename_speed_mismatches': mismatches,
               'point_labels_union': sorted(all_labels)}
    for kind, static in [('dynamic', False), ('static', True)]:
        selected = [r for r in rows if r['static'] == static]
        summary[kind] = {'files': len(selected), **{key: sum(r[key] > 0 for r in selected)
            for key in ('event_group_present', 'event_used', 'header_named_events', 'header_nonzero_times')},
            'files_with_explicit_ball_label': sum(bool(r['explicit_ball_labels']) for r in selected),
            'right_hand_usable': sum(r['right_hand']['valid_frames'] > 0 for r in selected),
            'left_hand_usable': sum(r['left_hand']['valid_frames'] > 0 for r in selected)}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'files.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('../openbiomechanics'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/source-audit'))
    args = parser.parse_args()
    audit(args.source, args.out)
