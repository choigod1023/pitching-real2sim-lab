"""Lossless archive/workbook linkage and small source C3D availability audit."""
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import ezc3d
import numpy as np

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def column_index(reference):
    match = re.fullmatch(r'([A-Z]+)[1-9][0-9]*', reference)
    if not match:
        raise ValueError('invalid_cell_reference')
    value = 0
    for char in match[1]:
        value = 26*value+ord(char)-ord('A')+1
    return value-1


def read_trial_book(path, subject):
    with zipfile.ZipFile(path) as source:
        shared = [''.join(e.itertext()) for e in ET.fromstring(source.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        workbook = ET.fromstring(source.read('xl/workbook.xml'))
        sheets = workbook.findall('.//m:sheet', NS)
        selected = [s for s in sheets if s.get('name') == subject]
        if len(selected) != 1:
            raise ValueError('missing_or_duplicate_subject_sheet')
        relation_id = selected[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        relationships = ET.fromstring(source.read('xl/_rels/workbook.xml.rels'))
        target = [r.get('Target') for r in relationships if r.get('Id') == relation_id]
        if len(target) != 1 or '..' in PurePosixPath(target[0]).parts:
            raise ValueError('invalid_sheet_relationship')
        sheet_path = target[0].lstrip('/') if target[0].startswith('/') else 'xl/'+target[0]
        root = ET.fromstring(source.read(sheet_path))
        cells = []
        for row in root.findall('m:sheetData/m:row', NS):
            values = {}
            for cell in row.findall('m:c', NS):
                value = cell.find('m:v', NS)
                if value is not None:
                    raw = shared[int(value.text)] if cell.get('t') == 's' else value.text
                    values[column_index(cell.get('r'))] = (raw, cell.get('r'))
            cells.append(values)
    if not cells:
        raise ValueError('empty_trial_sheet')
    days = {column: raw for column, (raw, _) in cells[0].items() if re.fullmatch(r'\d+w\d+d', raw)}
    if not days:
        raise ValueError('missing_day_headers')
    records = []
    for row in cells[1:]:
        for column, day in days.items():
            trial, trial_cell = row.get(column, ('', None))
            match = re.fullmatch(r'(DP|NDP)_(\d+)', trial)
            if match:
                speed, speed_cell = row.get(column+2, (None, None))
                records.append(dict(subject=subject, day=day, side=match[1], trial_number=int(match[2]),
                                    trial_raw=trial, status_raw=row.get(column+1, (None, None))[0],
                                    speed_raw=speed, speed_cell=speed_cell, trial_cell=trial_cell,
                                    speed_unit='UNKNOWN'))
    return records


def archive_trials(names):
    records = []
    for name in names:
        if ':' in name:
            raise ValueError('unsafe_archive_member')
        path = PurePosixPath(name.replace('\\', '/'))
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe_archive_member')
        if path.suffix.lower() != '.c3d' or len(path.parts) != 3:
            continue
        subject, day, filename = path.parts
        match = re.fullmatch(r'(S\d+)_(DP|NDP)_(\d+)_(.+)\.c3d', filename)
        if match:
            if subject != match[1] or not re.fullmatch(r'\d+w\d+d', day):
                raise ValueError('archive_subject_or_day_mismatch')
            records.append(dict(member=str(path), subject=subject, day=day, side=match[2],
                                trial_number=int(match[3]), status_raw=match[4]))
    return records


def link_trials(trials, book):
    columns = ('subject', 'day', 'side', 'trial_number')
    index = {}
    for row in book:
        key = tuple(row[c] for c in columns)
        index.setdefault(key, []).append(row)
    linked = []
    for row in trials:
        matches = index.get(tuple(row[c] for c in columns), [])
        result = dict(row, workbook_matches=len(matches), workbook_rows=matches)
        if len(matches) == 1:
            result['status_agrees'] = row['status_raw'] == matches[0]['status_raw']
            a, b = status_code(row['status_raw']), status_code(matches[0]['status_raw'])
            result['status_code_agrees'] = (a == b) if a is not None and b is not None else None
            result['speed_record_present'] = matches[0]['speed_raw'] is not None
        linked.append(result)
    return linked


def status_code(raw):
    """Compare only declared status prefixes; never repair a filename or workbook."""
    if not isinstance(raw, str):
        return None
    prefix = raw.split('-', 1)[0]
    return prefix if prefix and all(c in 'OFPM SWBL'.replace(' ', '') for c in prefix) else None


def select_new_trials(trials, prior_root, limit=3):
    def order(row):
        week, day = map(int, re.fullmatch(r'(\d+)w(\d+)d', row['day']).groups())
        return week, day, row['member']
    eligible = [r for r in trials if r['side'] == 'DP' and r['status_raw'] == 'O' and
                not (Path(prior_root)/r['member']).exists()]
    return [r['member'] for r in sorted(eligible, key=order)[:limit]]


def extract_member(archive, member, output):
    path = PurePosixPath(member.replace('\\', '/'))
    if ':' in member or path.is_absolute() or '..' in path.parts or path.suffix.lower() != '.c3d':
        raise ValueError('unsafe_selected_member')
    result = subprocess.run(['tar', '-xOf', str(archive), member], capture_output=True, check=True)
    payload = result.stdout
    if len(payload) < 512 or payload[1] != 80:
        raise ValueError('not_a_c3d_payload')
    destination = Path(output).joinpath(*path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise FileExistsError('preserve_existing_extracted_input')
    if not destination.exists():
        destination.write_bytes(payload)
    return destination


def inspect_c3d(path):
    source = ezc3d.c3d(str(path))
    point = source['parameters']['POINT']
    labels = [s.strip() for s in point['LABELS']['value']]
    values = source['data']['points']
    rate = float(point['RATE']['value'][0])
    valid = np.isfinite(values).all(axis=0) & (values[3] >= 0)
    groups = {'ball_marker': ['Ball'], 'throwing_hand': ['R3MC', 'R5MC'],
              'throwing_forearm': ['RFR1', 'RFR2', 'RFR3', 'RFR4'],
              'pelvis': ['LASI', 'RASI', 'LPSI', 'RPSI'],
              'torso': ['C7', 'T8', 'STRN', 'XIPH']}
    label_checks = {name: dict(expected=names, present=[n for n in names if n in labels])
                    for name, names in groups.items()}
    events = []
    if 'EVENT' in source['parameters']:
        event = source['parameters']['EVENT']
        for i, label in enumerate(event['LABELS']['value']):
            seconds = float(60*event['TIMES']['value'][0, i]+event['TIMES']['value'][1, i])
            events.append(dict(label=label.strip(), seconds=seconds))
    candidates = [e for e in events if e['label'] == 'Event']
    windows = []
    if len(candidates) == 1:
        index = int(round(candidates[0]['seconds']*rate))
        for shift in (0, -1):
            indices = np.arange(index+shift-int(round(.09*rate)), index+shift)
            within = indices.min() >= 0 and indices.max() < values.shape[2]
            checks = {}
            for name, names in groups.items():
                complete = all(n in labels for n in names)
                checks[name] = bool(within and complete and valid[[labels.index(n) for n in names]][:, indices].all()) if within and complete else False
            windows.append(dict(event_shift_frames=shift, sample_count=len(indices), all_group_markers_valid=checks))
    force = source['parameters'].get('FORCE_PLATFORM', {}).get('USED', {}).get('value', [])
    return dict(path=str(Path(path).resolve()), sha256=digest(path), frames=values.shape[2],
                point_rate_hz=rate, units=point['UNITS']['value'], point_count=len(labels),
                label_checks=label_checks, ball_valid_frames=int(valid[labels.index('Ball')].sum()) if 'Ball' in labels else None,
                events=events, candidate_windows=windows,
                force_platform_count=int(force[0]) if len(force) else None,
                ball_center_validated=False, physical_release_validated=False,
                source_event_mapping='UNKNOWN; both standard/legacy indices inspected')


def inspect_mat_cohort(root):
    from scipy.io import loadmat
    records = []
    for path in sorted(Path(root).glob('Sub*/*.mat')):
        data = loadmat(path, simplify_cells=True)
        keys = sorted(k for k in data if not k.startswith('__'))
        if set(keys) != {'X', 'Y'}:
            raise ValueError('unexpected_motion_mat_fields')
        x, y = np.asarray(data['X']), np.asarray(data['Y'])
        if x.shape != (101, 15, 3) or y.size != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('invalid_motion_mat_data')
        records.append(dict(subject=path.parent.name, trial=path.stem,
                            shape=list(x.shape), fields=keys, sha256=digest(path)))
    if not records:
        raise ValueError('no_motion_mat_files')
    counts = {subject: sum(r['subject'] == subject for r in records)
              for subject in sorted({r['subject'] for r in records})}
    return dict(subjects=len(counts), trials=len(records), trials_per_subject=counts,
                fields=['X', 'Y'], motion_shape=[101, 15, 3],
                explicit_ball_trajectory_field_present=False,
                explicit_raw_time_axis_present=False, records=records)
