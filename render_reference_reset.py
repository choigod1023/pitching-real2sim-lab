"""Real baseball motion, same-camera body overlays, and free-ball comparisons."""
import hashlib
import json
from pathlib import Path

import ezc3d
import imageio_ffmpeg as ff
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from covariant_release import LAB, OUT, ROOT
from multiview_release_validation import camera_models, project

COLORS = {'A': '#55d7ff', 'B': '#ff9a51', 'C': '#c5a1ff'}
FONT = 'C:/Windows/Fonts/malgun.ttf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(draw, xy, message, size=21, fill='white'):
    draw.text(xy, message, font=ImageFont.truetype(FONT, size), fill=fill)


def run():
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    metrics = result['primary']['metrics']
    obs = json.loads(Path(result['sources']['observation']['path']).read_text(encoding='utf-8'))
    primary = next(r for r in obs['conditions'] if r['threshold'] == 80 and r['c3d_frame_offset'] == 1)
    balls = {r['video_frame']: np.array(r['ball_rig_m']) for r in primary['positions']}
    align = json.loads(Path(obs['source_sha256']['alignment']['path']).read_text(encoding='utf-8'))
    body = ezc3d.c3d(obs['source_sha256']['body']['path'])
    labels = body['parameters']['ROTATION']['LABELS']['value']
    names = ['pelvis', 'torso', 'head', 'l_uarm', 'l_larm', 'l_hand', 'r_uarm', 'r_larm', 'r_hand',
             'l_thigh', 'l_shank', 'l_foot', 'r_thigh', 'r_shank', 'r_foot']
    edges = [('pelvis', 'torso'), ('torso', 'head'), ('torso', 'l_uarm'), ('l_uarm', 'l_larm'), ('l_larm', 'l_hand'),
             ('torso', 'r_uarm'), ('r_uarm', 'r_larm'), ('r_larm', 'r_hand'), ('pelvis', 'l_thigh'), ('l_thigh', 'l_shank'),
             ('l_shank', 'l_foot'), ('pelvis', 'r_thigh'), ('r_thigh', 'r_shank'), ('r_shank', 'r_foot')]
    basis, shift = np.array(align['rotation']), np.array(align['translation'])
    sequence = list(range(880, 973))+[f for f in range(973, 980) for _ in range(20)]+[979]*60
    joints = {f: {name: align['scale']*(basis@body['data']['rotations'][:3, 3, labels.index(name+'_4X4'), f+1])+shift
                  for name in names} for f in set(sequence)}
    cams = camera_models(); raw_images = {}; sources = {}
    for camera in (20, 21):
        path = LAB/f'speed-journey/artifacts/release-observation/camera{camera}.mp4'
        sources[str(path)] = sha(path)
        reader = ff.read_frames(str(path), pix_fmt='rgb24'); meta = next(reader)
        raw_images[camera] = {}
        for f, pixels in enumerate(reader):
            if f in joints:
                raw_images[camera][f] = Image.frombytes('RGB', meta['size'], pixels)
            if f >= 979:
                break
        reader.close()
    output = OUT/'baseball_multiview_reference_overlay.mp4'
    writer = ff.write_frames(str(output), (1280, 960), fps=30, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                             output_params=['-crf', '18', '-movflags', '+faststart'])
    writer.send(None)
    preview_indices = {60, 93+20*2+10, 93+20*5+10}
    saved = []
    for index, f in enumerate(sequence):
        canvas = Image.new('RGB', (1280, 960), '#101923'); d = ImageDraw.Draw(canvas)
        text(d, (20, 12), '새 기준 투구 · 같은 야구 동작에서 손–공 연결 비교', 28)
        text(d, (20, 52), '청록 A 고정 손 그립 / 주황 B 상대운동 / 보라 C 공 단독 대조 / 흰색 관측 공 후보', 20)
        for col, camera in enumerate((20, 21)):
            raw = raw_images[camera][f].copy(); rd = ImageDraw.Draw(raw)
            jp = {name: project(value, cams[camera])[0] for name, value in joints[f].items()}
            for a, b in edges:
                rd.line([tuple(jp[a]), tuple(jp[b])], fill='#7db9c4', width=2)
            for name in ('l_uarm', 'l_larm', 'l_hand'):
                x, y = jp[name]; rd.ellipse((x-4, y-4, x+4, y+4), fill='#55d7ff')
            if f in balls:
                x, y = project(balls[f], cams[camera])[0]
                rd.ellipse((x-10, y-10, x+10, y+10), outline='white', width=2)
            if f >= 977:
                for name in ('A', 'B', 'C'):
                    positions = np.array(metrics[name]['predicted_positions_m'])[:f-976]
                    positions = np.vstack([metrics[name]['launch_position_m'], positions])
                    pp = project(positions, cams[camera]); rd.line([tuple(x) for x in pp], fill=COLORS[name], width=3)
                    x, y = pp[-1]; rd.ellipse((x-6, y-6, x+6, y+6), outline=COLORS[name], width=3)
            canvas.paste(raw.resize((640, 360)), (col*640, 90))
            center = project(balls[f] if f in balls else joints[f]['l_hand'], cams[camera])[0]
            x, y = center
            # Native-pixel overlays are drawn before the common zoom transform.
            crop = raw.crop((int(x)-160, int(y)-70, int(x)+160, int(y)+70)).resize((640, 280))
            canvas.paste(crop, (col*640, 450))
            text(d, (col*640+12, 98), f'camera {camera} · decoded frame {f}', 19)
        text(d, (20, 740), '같은 투구 평가 평균 구속: '+ ' / '.join(f'{name} {metrics[name]["mean_predicted_speed_kmh"]:.2f}' for name in ('A', 'B', 'C'))+f' km/h | 영상 후보 {metrics["B"]["mean_candidate_speed_kmh"]:.2f}', 20)
        text(d, (20, 776), f'구속 차이 MAE: 고정 그립 {metrics["A"]["mean_absolute_speed_discrepancy_kmh"]:.2f} → 상대운동 {metrics["B"]["mean_absolute_speed_discrepancy_kmh"]:.2f} km/h / 공 단독 {metrics["C"]["mean_absolute_speed_discrepancy_kmh"]:.2f}', 23)
        text(d, (20, 815), f'추가 정규화 D/E: {metrics["D"]["mean_absolute_speed_discrepancy_kmh"]:.2f} km/h — 미채택 | 기록 {result["source_label_kmh"]:.2f} km/h는 시점 미확인', 20)
        text(d, (20, 851), '전후 자세 수정 0° / 0mm · 힘·일 조건 미검증 · 구속 향상 후보가 아닌 관측 연결 비교', 20, '#ffcb8b')
        phase = '학습 973–976 / 평가 977–979 · 976은 계산 경계이며 실제 이탈 시각 미확정'
        text(d, (20, 888), phase, 20, '#c1d1dc')
        text(d, (20, 922), 'Prime BaseballThrow_001 · 제공 촬영률 360Hz · 느린 재생/추가 정지 · 원영상 OpenBiomechanics / Driveline', 17, '#9aaebf')
        writer.send(np.asarray(canvas).tobytes())
        if index in preview_indices:
            p = OUT/f'overlay_frame_{f}.png'; canvas.save(p); saved.append(str(p))
    writer.close()
    reader = ff.read_frames(str(output), pix_fmt='rgb24'); meta = next(reader)
    count = sum(1 for _ in reader); reader.close()
    if count != len(sequence) or meta['size'] != (1280, 960):
        raise RuntimeError('Full output decode failed')
    manifest = {'video': str(output.resolve()), 'sha256': sha(output), 'frames': count, 'fps': 30,
                'seconds': count/30., 'size': list(meta['size']), 'all_frames_decoded': True,
                'source_frames': [880, 979], 'sources': sources, 'results_sha256': sha(OUT/'results.json'),
                'previews': saved, 'body_change_deg': 0., 'body_change_mm': 0., 'force_work_validated': False,
                'performance_gain_demonstrated': False, 'reference': result['trial']}
    (OUT/'video_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'video': manifest['video'], 'frames': count, 'duration': count/30.}))


if __name__ == '__main__':
    run()
