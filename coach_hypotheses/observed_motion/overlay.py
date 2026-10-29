"""Replay two measured skeletons in fixed lab axes and unwarped elapsed time."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import ezc3d
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..observed_tradeoffs.analysis import file_digest

PROTOCOL_COMMIT = '4664982b8a5638c76d03927a2e1a94c56a2a3e8f'
CASES = (('830', '1615_1', '1615_2'), ('774', '1097_2', '1097_3'))
EDGES = (('rear_hip', 'lead_hip'), ('shoulder_jc', 'glove_shoulder_jc'),
         ('rear_hip', 'shoulder_jc'), ('lead_hip', 'glove_shoulder_jc'),
         ('rear_hip', 'rear_knee_jc'), ('rear_knee_jc', 'rear_ankle_jc'),
         ('lead_hip', 'lead_knee_jc'), ('lead_knee_jc', 'lead_ankle_jc'),
         ('shoulder_jc', 'elbow_jc'), ('elbow_jc', 'wrist_jc'), ('wrist_jc', 'hand_jc'),
         ('glove_shoulder_jc', 'glove_elbow_jc'), ('glove_elbow_jc', 'glove_wrist_jc'),
         ('glove_wrist_jc', 'glove_hand_jc'))
JOINTS = tuple(sorted({name for edge in EDGES for name in edge}))
ANGLES = ('lead_knee_angle_x', 'torso_pelvis_angle_z', 'elbow_angle_x')
COLORS = ((255, 187, 84), (93, 193, 255))
SIZE = (1600, 1040)
FPS, SLOWDOWN = 30, 12
SOURCE_HASHES = {
    'metadata.csv': '52e9f0a966447f358405c20a46920a9a11944a366e21fb5a1e50581e0b4783a0',
    'full_sig/landmarks.csv': 'bddd0bfab9da2a02fb4235d400d446ce08452dc1a43f47a8f5d44f41a2002370',
    'full_sig/joint_angles.csv': '0cc950eb86d17ff2d7ce25b57ee7bff8da402d0d38922754be1a4ef2db2de2ce',
}


def interpolate(time, values, query):
    time, values, query = np.asarray(time, float), np.asarray(values, float), np.asarray(query, float)
    if (time.ndim != 1 or len(time) < 2 or values.shape[0] != len(time) or
            not all(np.isfinite(x).all() for x in (time, values, query)) or np.any(np.diff(time) <= 0)):
        raise ValueError('invalid_interpolation_support')
    if np.any(query < time[0]) or np.any(query > time[-1]):
        raise ValueError('extrapolation_prohibited')
    columns = values.reshape(len(time), -1)
    result = np.stack([np.interp(query.ravel(), time, column) for column in columns.T], axis=-1)
    return result.reshape(query.shape + values.shape[1:])


def prepare_track(landmarks, angles, speed):
    """Subtract one FP10 horizontal hip origin; preserve height and later motion."""
    if not np.array_equal(landmarks.time.to_numpy(), angles.time.to_numpy()):
        raise ValueError('landmark_angle_time_mismatch')
    time = np.arange(len(landmarks)) / 360.
    if np.max(np.abs(landmarks.time.to_numpy(float) - time)) > .000051:
        raise ValueError('noncanonical_time_grid')
    events = {}
    for key in ('fp_10_time', 'BR_time'):
        if (landmarks[key].nunique() != 1 or angles[key].nunique() != 1 or
                landmarks[key].iloc[0] != angles[key].iloc[0]):
            raise ValueError('nonconstant_or_mismatched_event')
        events[key] = float(landmarks[key].iloc[0])
    fp, br = events['fp_10_time'], events['BR_time']
    if not time[0] < fp < br < time[-1] or not np.isfinite(speed) or speed <= 0:
        raise ValueError('invalid_events_or_speed')
    raw = np.stack([landmarks[[name + '_' + axis for axis in 'xyz']].to_numpy(float) for name in JOINTS], axis=1)
    angle_values = angles[list(ANGLES)].to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(angle_values).all():
        raise ValueError('nonfinite_pose_no_imputation')
    angle_values = np.rad2deg(np.unwrap(np.deg2rad(angle_values), axis=0))
    hips = (raw[:, JOINTS.index('rear_hip')] + raw[:, JOINTS.index('lead_hip')]) / 2
    origin = interpolate(time, hips, fp)
    origin[2] = 0.
    return dict(time=time, xyz=raw - origin, raw_xyz=raw, origin=origin, angles=angle_values,
                fp=fp, br=br, speed=float(speed), pitch=str(landmarks.session_pitch.iloc[0]))


def load_tracks(data_dir, metrics_file, sensitivity_file):
    data_dir = Path(data_dir)
    for name, expected in SOURCE_HASHES.items():
        if file_digest(data_dir / name) != expected:
            raise ValueError('changed_source:' + name)
    if file_digest(metrics_file) != 'c555d826ce78fc0ca20bb6a2be017ffc6df8fa84e8911001a3250ac3ea4b04f4':
        raise ValueError('changed_metrics')
    if file_digest(sensitivity_file) != '32ca2146b29a31ddf212988711705832e05366a0300300999f4535ed5fa0e490':
        raise ValueError('changed_sensitivity_result')
    sensitivity = json.loads(Path(sensitivity_file).read_text())
    metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={'user': str}).set_index('session_pitch')
    metrics = pd.read_csv(metrics_file)
    metrics = metrics[metrics.cutoff_hz.eq(20)].set_index('session_pitch')
    pitches = {pitch for _, fast, slow in CASES for pitch in (fast, slow)}
    columns = [name + '_' + axis for name in JOINTS for axis in 'xyz']
    landmarks = pd.read_csv(data_dir / 'full_sig/landmarks.csv', usecols=['session_pitch', 'time', 'fp_10_time', 'BR_time', *columns])
    angles = pd.read_csv(data_dir / 'full_sig/joint_angles.csv', usecols=['session_pitch', 'time', 'fp_10_time', 'BR_time', *ANGLES])
    tracks, audit = {}, {}
    for pitch in sorted(pitches):
        g = landmarks[landmarks.session_pitch.eq(pitch)].sort_values('time')
        a = angles[angles.session_pitch.eq(pitch)].sort_values('time')
        speed = float(metadata.loc[pitch, 'pitch_speed_mph']) * 1.609344
        if not np.isclose(speed, metrics.loc[pitch, 'pitch_speed_kmh'], atol=1e-6, rtol=0):
            raise ValueError('speed_identity_mismatch')
        tracks[pitch] = prepare_track(g, a, speed)
        candidates = list((data_dir / 'c3d').rglob(metadata.loc[pitch, 'filename_new']))
        if len(candidates) != 1:
            raise ValueError('ambiguous_raw_c3d')
        source = candidates[0]
        c3d = ezc3d.c3d(str(source))
        point = c3d['parameters']['POINT']
        raw = c3d['data']['points']
        if (point['UNITS']['value'][0].strip().lower() != 'm' or point['RATE']['value'][0] != 360 or
                c3d['header']['points']['first_frame'] != 0 or raw.shape[-1] != len(g)):
            raise ValueError('raw_processed_time_units_mismatch')
        labels = {name.strip(): i for i, name in enumerate(point['LABELS']['value'])}
        midpoint = (raw[:3, labels['RWRA'], :].T + raw[:3, labels['RWRB'], :].T) / 2
        delta = np.linalg.norm(midpoint - tracks[pitch]['raw_xyz'][:, JOINTS.index('wrist_jc')], axis=1) * 1000
        if not np.isfinite(delta).all():
            raise ValueError('invalid_raw_wrist_support')
        audit[pitch] = dict(c3d_path=str(source.resolve()), c3d_sha256=file_digest(source), frames=len(g),
                            csv_time_rounding_max_seconds=float(np.max(abs(g.time.to_numpy() - tracks[pitch]['time']))),
                            raw_marker_midpoint_processed_wrist_median_distance_mm=float(np.median(delta)),
                            fp10_to_br_ms=float((tracks[pitch]['br'] - tracks[pitch]['fp']) * 1000),
                            fixed_xy_origin_mm=(tracks[pitch]['origin'] * 1000).tolist(),
                            marker_mapping='POINT LABELS names, not shared array indices',
                            raw_residual_zero_is_not_accuracy=True)
    cases = []
    for user, fast, slow in CASES:
        if any(str(metadata.loc[p, 'user']) != user for p in (fast, slow)):
            raise ValueError('wrong_case_player')
        row = next(r for r in sensitivity['pair_records'] if r['faster_pitch'] == fast and r['slower_pitch'] == slow)
        cases.append(dict(user=user, fast=fast, slow=slow, sensitivity=row,
                          peak_loads={p: {k: float(metrics.loc[p, k]) for k in ('elbow_varus_peak_nm', 'shoulder_resultant_peak_n')} for p in (fast, slow)}))
    return tracks, cases, audit


def aligned_times(tracks, anchor):
    if anchor not in ('fp', 'br') or len(tracks) != 2:
        raise ValueError('need_two_tracks_and_known_anchor')
    requested = (-.12, .30) if anchor == 'fp' else (-.30, .08)
    lo = max(requested[0], *(track['time'][0] - track[anchor] for track in tracks))
    hi = min(requested[1], *(track['time'][-1] - track[anchor] for track in tracks))
    if not lo < hi:
        raise ValueError('no_shared_observed_support')
    return lo + np.arange(int(np.floor((hi - lo) * FPS * SLOWDOWN)) + 1) / (FPS * SLOWDOWN)


def sample_pair(tracks, elapsed, anchor):
    points, angles, states = [], [], []
    for track in tracks:
        time = float(elapsed + track[anchor])
        points.append(interpolate(track['time'], track['xyz'], time))
        angles.append(interpolate(track['time'], track['angles'], time))
        states.append(dict(source_time=time, br_remaining_ms=(track['br'] - time) * 1000,
                           phase='FP10 전' if time < track['fp'] else ('FP10→BR' if time < track['br'] else 'BR 후')))
    return np.array(points), np.array(angles), states


def scene_bounds(tracks):
    points = []
    for anchor in ('fp', 'br'):
        for q in aligned_times(tracks, anchor):
            points.append(sample_pair(tracks, q, anchor)[0])
    all_points = np.concatenate(points).reshape(-1, 3)
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    lower[2] = min(0., lower[2])
    return lower, upper


def draw_frame(tracks, case, elapsed, anchor, bounds, fonts):
    image = Image.new('RGB', SIZE, '#101923')
    draw = ImageDraw.Draw(image)
    draw.text((26, 14), f'선수 {case["user"]} · 실제 두 투구의 전신 겹침', font=fonts[32], fill='white')
    for index, track in enumerate(tracks):
        role = '빠른 기록' if index == 0 else '느린 기록'
        draw.text((26 + index * 790, 66), f'{role}  {track["pitch"]}  |  {track["speed"]:.3f} km/h', font=fonts[28], fill=COLORS[index])
    title = 'FP10 착지 정렬 · 실제 경과시간' if anchor == 'fp' else '제공 BR 이벤트 정렬 · 별도 진단'
    draw.text((26, 112), f'{title}  {elapsed * 1000:+.1f} ms   |   1/{SLOWDOWN}배 재생 · 시간 압축/늘임 없음', font=fonts[24], fill='white')
    duration = [(t['br'] - t['fp']) * 1000 for t in tracks]
    draw.text((26, 148), f'착지→이탈: 빠른 기록 {duration[0]:.1f} ms / 느린 기록 {duration[1]:.1f} ms', font=fonts[21], fill='#b6c6d6')
    points, angles, states = sample_pair(tracks, elapsed, anchor)
    lo, hi = bounds
    scale = min(680 / max(hi[0] - lo[0], hi[1] - lo[1], .1), 470 / max(hi[2] - lo[2], .1))
    for view, axis in enumerate((0, 1)):
        left, top, width, height = 24 + view * 792, 186, 760, 510
        draw.rounded_rectangle((left, top, left + width, top + height), radius=10, outline='#34495c', width=2)
        label = '실험실 X–Z · 홈 방향 →' if axis == 0 else '실험실 Y–Z · 1루 방향 →'
        draw.text((left + 14, top + 10), label, font=fonts[21], fill='#bbcad8')
        center = np.array([(lo[axis] + hi[axis]) / 2, (lo[2] + hi[2]) / 2])
        projected = points[:, :, [axis, 2]].copy()
        projected = (projected - center) * np.array([scale, -scale]) + np.array([left + width / 2, top + height / 2 + 15])
        ground_y = top + height / 2 + 15 + center[1] * scale
        draw.line([(left + 12, ground_y), (left + width - 12, ground_y)], fill='#253c4d', width=2)
        for index in (1, 0):
            for first, second in EDGES:
                draw.line([tuple(projected[index, JOINTS.index(first)]), tuple(projected[index, JOINTS.index(second)])], fill=COLORS[index], width=5 if index else 4)
            for x, y in projected[index]:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS[index])
        for joint in ('wrist_jc', 'lead_knee_jc'):
            start, end = projected[:, JOINTS.index(joint)]
            draw.line([tuple(start), tuple(end)], fill='#edf3f8', width=2)
    labels = ('앞무릎 X', '골반–몸통 분리 Z', '팔꿈치 X')
    for index, label in enumerate(labels):
        delta = angles[0, index] - angles[1, index]
        draw.text((26 + index * 520, 716), f'{label}: {angles[1,index]:.1f}° → {angles[0,index]:.1f}°  (Δ {delta:+.1f}°)', font=fonts[22], fill='white')
    distances = [np.linalg.norm(points[0, JOINTS.index(j)] - points[1, JOINTS.index(j)]) * 1000 for j in ('wrist_jc', 'lead_knee_jc')]
    draw.text((26, 755), f'고정 XY 정렬 후 위치 차이: 손목 {distances[0]:.1f} mm / 앞무릎 {distances[1]:.1f} mm  ·  교정량/정확도 아님', font=fonts[23], fill='white')
    for index, (track, state) in enumerate(zip(tracks, states)):
        draw.text((26 + index * 790, 797), f'{track["pitch"]}: {state["phase"]}  |  BR까지 {state["br_remaining_ms"]:+.1f} ms', font=fonts[23], fill=COLORS[index])
    loads = case['peak_loads']
    f, s = loads[case['fast']], loads[case['slow']]
    draw.text((26, 839), f'추가20Hz 피크 차이(빠른−느린): 팔꿈치 {f["elbow_varus_peak_nm"]-s["elbow_varus_peak_nm"]:+.2f} N·m / 어깨 합력 {f["shoulder_resultant_peak_n"]-s["shoulder_resultant_peak_n"]:+.2f} N', font=fonts[23], fill='#dce7ef')
    passed = case['sensitivity']['profiles'][1]['independent_event']
    draw.text((26, 878), '사전 공학 오차(±0.25km/h·0.5N·m·5N)+이벤트 이동: ' + ('유지' if passed else '유지 안 됨') + '  |  실제 오차 검증 아님', font=fonts[22], fill='#c6d4e1')
    origins = [t['origin'] * 1000 for t in tracks]
    draw.text((26, 916), f'FP10 고정 XY 원점: 빠른({origins[0][0]:.1f},{origins[0][1]:.1f}) / 느린({origins[1][0]:.1f},{origins[1][1]:.1f}) mm · 높이/이후 이동 보존', font=fonts[20], fill='#aec0ce')
    draw.text((26, 951), '관측 관절 재생(원 처리20Hz) · 자세 변경/구속 예측 아님 · 동일 근력/일 미검증 · 공/손가락 표시 없음', font=fonts[21], fill='#cedce8')
    draw.text((26, 993), 'OpenBiomechanics / Driveline Baseball · derived visualization CC BY-NC-SA 4.0', font=fonts[18], fill='#8199ad')
    return image


def export_case(tracks, case, output, fonts):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ('player_' + case['user'] + '_observed_overlay.mp4')
    bounds = scene_bounds(tracks)
    writer = imageio_ffmpeg.write_frames(str(path), SIZE, fps=FPS, codec='libx264', pix_fmt_out='yuv420p', output_params=['-crf', '19', '-movflags', '+faststart'])
    writer.send(None)
    segments, count, previews = [], 0, []
    try:
        for anchor in ('fp', 'br'):
            queries = aligned_times(tracks, anchor)
            targets = {int(np.argmin(abs(queries))): anchor + '_zero'}
            if anchor == 'fp':
                targets[int(np.argmin(abs(queries - (tracks[0]['br'] - tracks[0]['fp']))))] = 'fp_fast_br'
            segment = dict(anchor=anchor, frame_start=count, frames=len(queries),
                           elapsed_start=float(queries[0]), elapsed_end=float(queries[-1]))
            for index, q in enumerate(queries):
                frame = draw_frame(tracks, case, q, anchor, bounds, fonts)
                writer.send(np.asarray(frame))
                count += 1
                if index in targets:
                    preview = output / ('player_' + case['user'] + '_' + targets[index] + '.png')
                    frame.save(preview)
                    previews.append(str(preview.resolve()))
            segments.append(segment)
    finally:
        writer.close()
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostats', '-progress', 'pipe:1', '-i', str(path), '-f', 'null', '-'], capture_output=True, text=True, check=True)
    frames = re.findall(r'^frame=(\d+)', decoded.stdout, flags=re.MULTILINE)
    if not frames or int(frames[-1]) != count or decoded.stderr.strip():
        raise ValueError('video_full_decode_or_frame_count_failed')
    return dict(path=str(path.resolve()), sha256=file_digest(path), frames=count, fps=FPS, slowdown=SLOWDOWN,
                full_decode_verified=True, segments=segments, previews=previews,
                lower_bound_m=bounds[0].tolist(), upper_bound_m=bounds[1].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--metrics-file', required=True, type=Path)
    parser.add_argument('--sensitivity-file', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('artifacts'))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must remain inside observed_motion package')
    tracks, cases, audit = load_tracks(args.data_dir, args.metrics_file, args.sensitivity_file)
    fonts = {size: ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', size) for size in (18, 20, 21, 22, 23, 24, 28, 32)}
    videos = [export_case([tracks[c['fast']], tracks[c['slow']]], c, args.output, fonts) for c in cases]
    result = dict(protocol_commit=PROTOCOL_COMMIT, protocol_sha256=file_digest(Path(__file__).with_name('PROTOCOL.md')),
                  code_sha256=file_digest(__file__), source_hashes=SOURCE_HASHES, input_audit=audit, videos=videos,
                  metric_source=dict(path=str(args.metrics_file.resolve()), sha256=file_digest(args.metrics_file)),
                  sensitivity_source=dict(path=str(args.sensitivity_file.resolve()), sha256=file_digest(args.sensitivity_file)),
                  colors=dict(faster_recorded='orange', slower_recorded='blue'),
                  pose_processing='provided 20Hz, no additional filtering', angle_processing='provided angles unwrapped in degrees for linear sampling',
                  alignment='FP10 primary + BR diagnostic, real elapsed time, fixed FP10 XY translations only',
                  camera_views='fixed lab XZ and YZ, same meter/pixel scale, no rotation fit',
                  speed_values='recorded, not predicted', same_capacity_validated=False, intervention_effect_validated=False,
                  coaching_recommendation_ready=False, data_license='CC BY-NC-SA 4.0')
    results = Path(__file__).with_name('results')
    results.mkdir(exist_ok=True)
    (results / 'overlay_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(videos, indent=2))


if __name__ == '__main__':
    main()
