"""Read existing same-pitch records without fitting a vision or flight model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MPH_TO_KMH = 1.609344
INCH_TO_CM = 2.54
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
LINKS = {4: 'https://photos.app.goo.gl/jZE7B66ajCAFCnKW9',
         8: 'https://photos.app.goo.gl/mD4Yd8LxF79HBdBb9'}
VIDEO_RECEIPTS = {
    'Branson4S1': dict(pitch_id=4, sha256='0fe909c6fbee6a96118de5212922e15f712722d48a31ed00aff41be9eacc64fe',
                      encoded_fps=59.90908737713198, header_frames=257, decoded_frames=256),
    'Branson4S2': dict(pitch_id=8, sha256='c0d54661356354549d55122df97c6651031cb0141a333164550f25b3d76c05a5',
                      encoded_fps=59.867638094685205, header_frames=275, decoded_frames=266),
}


def number_or_none(value, factor=1.):
    """Convert an observed numeric cell; preserve blanks and reject nonfinite data."""
    if value is None or str(value).strip() == '':
        return None
    result = float(value) * factor
    if not math.isfinite(result):
        raise ValueError('Nonfinite observed number')
    return result


def read_workbook(source):
    """Extract raw cells and hyperlink targets from the existing first XLSX sheet."""
    with ZipFile(source) as workbook:
        root = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
        strings = [''.join(p.text or '' for p in item.findall('.//s:t', NS)) for item in root]
        sheet = ET.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
        relationships = ET.fromstring(workbook.read('xl/worksheets/_rels/sheet1.xml.rels'))
    targets = {item.get('Id'): item.get('Target') for item in relationships}
    links = {item.get('ref'): targets.get(item.get(REL)) for item in sheet.findall('.//s:hyperlink', NS)}
    header, rows = None, []
    for row in sheet.findall('.//s:sheetData/s:row', NS):
        cells = {}
        for cell in row.findall('s:c', NS):
            value = cell.findtext('s:v', '', NS)
            if cell.get('t') == 's':
                value = strings[int(value)]
            elif cell.get('t') == 'inlineStr':
                value = ''.join(p.text or '' for p in cell.findall('.//s:t', NS))
            cells[cell.get('r').rstrip('0123456789')] = value
        if 'Game Pitch' in cells.values():
            header = {label: column for column, label in cells.items()}
            continue
        if not header or not cells.get(header['Game Pitch'], '').isdigit():
            continue
        values = {key: cells.get(column, '') for key, column in header.items()}
        ref = header.get('HS Vid', '') + row.get('r')
        rows.append({'xlsx_row': int(row.get('r')), 'values': values,
                     'video_hyperlink': links.get(ref), 'video_cell_reference': ref})
    if not header or not rows:
        raise ValueError('Missing pitch header or records')
    ids = [row['values']['Game Pitch'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate pitch identifiers')
    return rows


def record_from_row(row, clip_id):
    """Keep record values distinct from estimates and preserve unknown identity."""
    values = row['values']
    pitch_id = int(values['Game Pitch'])
    hyperlink = row.get('video_hyperlink')
    cell_url = values.get('HS Vid') or None
    if hyperlink and cell_url and hyperlink != cell_url:
        raise ValueError('Video cell and relationship disagree')
    video = hyperlink or cell_url
    if pitch_id not in LINKS or video != LINKS[pitch_id]:
        raise ValueError('Unverified pitch-to-video relationship')
    return dict(
        clip_id=clip_id, pitch_id=pitch_id, xlsx_row=row['xlsx_row'],
        video_hyperlink=hyperlink, video_cell_url=cell_url, video_source_url=video,
        same_row_video_record_link=True, pitch_type=values.get('Pitch Type') or None,
        pitcher_first_name=values.get('Pitcher First Name') or None,
        pitcher_last_name=values.get('Pitcher Last Name') or None,
        same_pitcher_verified=False,
        recorded_speed_kmh=number_or_none(values.get('RelSpeed (MPH)'), MPH_TO_KMH),
        rpm=number_or_none(values.get('SpinRate')),
        effective_rpm=number_or_none(values.get('TrueSpinRelease')),
        spin_efficiency_percent=number_or_none(values.get('Spin Efficiency')),
        ivb_cm=number_or_none(values.get('InducedVertBreak (Inches)'), INCH_TO_CM),
        hb_cm=number_or_none(values.get('HorzBreak (Inches)'), INCH_TO_CM),
        plate_height_cm=number_or_none(values.get('PlateLocHeight (Inches)'), INCH_TO_CM),
        horizontal_release_angle_deg=number_or_none(values.get('HorzRelAngle')),
        vertical_release_angle_deg=number_or_none(values.get('VertRelAngle')),
        spin_direction_clock=values.get('Spin Direction') or None,
        capture_fps_nominal=500., capture_clock_independently_verified=False,
        camera_geometry_independently_calibrated=False,
        contact_or_finger_release_order_measured=False,
        source_role='Same-row instrument records for comparison, not model fitting',
    )


def prime_listing_record(text):
    """Require the exact Prime event row; do not borrow an Edgertronic speed."""
    matches = []
    for line, cells in enumerate(csv.reader(io.StringIO(text)), 1):
        if len(cells) < 8 or cells[0] != '13' or cells[1] != 'OptiTrack Prime Colors':
            continue
        match = re.fullmatch(r'Baseball Throw - ([0-9.]+) MPH', cells[7])
        if not match or cells[2:7] != ['8', '360', 'Movement', 'Dynamic', 'Human']:
            raise ValueError('Prime event metadata changed')
        matches.append(dict(source_line=line, raw_fields=cells,
                            recorded_speed_kmh=number_or_none(match.group(1), MPH_TO_KMH),
                            rpm=None, ivb_cm=None, hb_cm=None,
                            capture_clock_independently_verified=False,
                            speed_measurement_position_time_verified=False))
    if len(matches) != 1:
        raise ValueError('Expected exactly one Prime event 13')
    return matches[0]


def build_evidence(lab):
    """Assemble provenance and previous decoding receipts from existing local files."""
    lab = Path(lab)
    workbook = lab/'public-finger-flight-pair-audit-20261110/example-rapsodo.xlsx'
    data = workbook.read_bytes()
    rows = {int(row['values']['Game Pitch']): row for row in read_workbook(io.BytesIO(data))}
    records = []
    source_audit = lab/'non-skenes-paired-source-audit-20261110/SOURCE_AUDIT.md'
    source_audit_text = source_audit.read_text(encoding='utf-8')
    for clip_id, receipt in VIDEO_RECEIPTS.items():
        video = source_audit.parent/f'{clip_id}.mp4'
        with video.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != receipt['sha256'] or digest not in source_audit_text:
            raise ValueError('Video no longer matches the previous decode receipt')
        record = record_from_row(rows[receipt['pitch_id']], clip_id)
        record.update(video_path=str(video), video_sha256=digest,
                      previous_video_audit={**receipt, 'redecoded_by_this_command': False,
                          'execution_note': 'Read-only worker cv2 sequential decode completed 2026-11-13; agrees with existing SOURCE_AUDIT.md.',
                          'audit_path': str(source_audit),
                          'audit_sha256': hashlib.sha256(source_audit.read_bytes()).hexdigest(),
                          'frame_dimensions': [1920, 1080]})
        records.append(record)
    listing = lab/'ball-coordinate-audit/artifacts/video-audit/prime_color_shots.csv'
    listing_bytes = listing.read_bytes()
    manifest = listing.with_name('source_manifest.json')
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    listing_digest = hashlib.sha256(listing_bytes).hexdigest()
    if listing_digest != metadata['prime_color']['sha256']:
        raise ValueError('Prime listing hash differs from its acquisition manifest')
    prime = prime_listing_record(listing_bytes.decode('utf-8-sig'))
    prime.update(trial='OBP_movement_BaseballThrow_001',
                 source_path=str(listing), source_sha256=listing_digest,
                 source_url=metadata['prime_color']['url'],
                 association_source=str(manifest),
                 association_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 association_scope='Event 13 scalar plus previously audited Drive filename mapping; not per-camera independent speed.',
                 camera_geometry_status='provisional_not_bundle_adjusted',
                 c3d_frame_minus_video_frame=1)
    for camera in (17, 20, 21):
        records.append({**prime, 'clip_id': f'prime{camera}', 'camera': camera})
    records.append(dict(clip_id='driveline_four_seam_demo', recorded_speed_kmh=None,
                        rpm=None, ivb_cm=None, hb_cm=None, source_role='Demonstration without same-pitch instrument records'))
    return dict(schema_version=1, purpose='Same-pitch record provenance only; no prediction or effect validation',
                records=records, workbook_path=str(workbook),
                workbook_sha256=hashlib.sha256(data).hexdigest(),
                speed_gap_kmh=abs(records[0]['recorded_speed_kmh']-records[1]['recorded_speed_kmh']),
                within_1kmh_pair_available=False, same_pitcher_verified=False,
                official_sources={
                    'capture_protocol': 'https://baseballaero.com/services/',
                    'hand_tasks': 'https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker',
                    'legacy_hands': 'https://chuoling.github.io/mediapipe/solutions/hands.html'},
                hand_coordinate_contract={
                    'world_landmarks': 'Meters, predicted coordinates relative to hand geometric center; not calibrated ball/body world frame.',
                    'normalized_depth': 'Wrist-origin depth, not the world-landmark origin.',
                    'handedness': 'Record API/version/flip; current Tasks and legacy selfie assumptions differ.',
                    'accuracy': 'Detection or crop agreement does not establish anatomical accuracy or contact.',
                    'negative_control': 'Score hands hallucinated on post-release ball-only frames separately.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evidence = build_evidence(args.lab)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'records': len(evidence['records']), 'speed_gap_kmh': evidence['speed_gap_kmh'],
                      'within_1kmh_pair_available': evidence['within_1kmh_pair_available']}))


if __name__ == '__main__':
    main()
