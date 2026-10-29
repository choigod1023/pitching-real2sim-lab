"""Baseball image/model timing sensitivity, explicitly not a posture intervention."""
import json
from pathlib import Path
import imageio_ffmpeg as ff
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prime17_hand_alignment import ROOT, LAB, OUT, Source, project, sha


def label(draw, xy, text, size=21, color='white'):
    draw.text(xy, text, font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', size), fill=color)


def run():
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    source = Source(); states = result['compatible_time_only_states']
    chosen = [('기존 +1.00', 1., '#55d7ff'), ('양립 +0.50', states[0]['offset_frames'], '#ff9a51'), ('양립 +2.25', states[-1]['offset_frames'], '#c5a1ff')]
    sequence = list(range(940, 971))+[f for f in range(971, 980) for _ in range(12)]+[979]*60
    inputs = {17: LAB/'release-window-validation-20261028/artifacts/release-window/camera17.mp4',
              21: LAB/'speed-journey/artifacts/release-observation/camera21.mp4'}
    images = {}; source_hashes = {}
    for camera, path in inputs.items():
        source_hashes[str(path)] = sha(path)
        reader = ff.read_frames(str(path), pix_fmt='rgb24'); meta = next(reader)
        images[camera] = {}
        for f, raw in enumerate(reader):
            if f in sequence:
                images[camera][f] = Image.frombytes('RGB', meta['size'], raw)
            if f >= 979:
                break
        reader.close()
    output = OUT/'prime17_alignment_sensitivity.mp4'
    writer = ff.write_frames(str(output), (1280, 940), fps=30, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                             output_params=['-crf', '18', '-movflags', '+faststart']); writer.send(None)
    regions = json.loads((ROOT/'prime17_regions.json').read_text(encoding='utf-8'))['observations']
    speeds = [s['B_ball_evaluation']['mean_predicted_speed_kmh'] for s in states]
    for index, f in enumerate(sequence):
        canvas = Image.new('RGB', (1280, 940), '#101923'); d = ImageDraw.Draw(canvas)
        label(d, (20, 12), '야구 투구 · 손·팔 시간 정렬을 영상으로 구별할 수 있는가', 28)
        label(d, (20, 53), '청록 기존 +1.00 / 주황 양립 +0.50 / 보라 양립 +2.25 프레임 · 노랑 관찰 영역', 20)
        for column, camera in enumerate((17, 21)):
            raw = images[camera][f].copy(); rd = ImageDraw.Draw(raw)
            if camera == 17:
                for r in regions:
                    if r['frame'] == f:
                        rd.rectangle(r['box'], outline='#ffd34f', width=2)
            for name, offset, color in chosen:
                pts = [source.origins[s](f+offset) for s in ['l_uarm', 'l_larm', 'l_hand']]
                pixels = project(np.array(pts), source.cameras[camera])
                rd.line([tuple(p) for p in pixels], fill=color, width=2)
                for x, y in pixels:
                    rd.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
            canvas.paste(raw.resize((640, 360)), (column*640, 92))
            center = project(source.origins['l_hand'](f+1), source.cameras[camera])[0]
            x, y = map(int, center)
            crop = raw.crop((x-120, y-70, x+120, y+70)).resize((640, 280))
            canvas.paste(crop, (column*640, 452))
            label(d, (column*640+12, 101), f'camera {camera} · decoded {f}', 19)
        label(d, (20, 741), f'채택 보정 없음: 구속 MAE 1.324 → 1.324 km/h | 평가 20영역 중 기존15영역 안', 22)
        label(d, (20, 779), f'같은 적합 영역과 양립하는 8상태의 예측 구속 {min(speeds):.2f}~{max(speeds):.2f} km/h', 22)
        label(d, (20, 817), '표본 대응 차이일 뿐 선수 자세 변경이 아님 · 실제 오차범위/95% 구간이 아님', 20, '#ffcb8b')
        label(d, (20, 853), '영역 안=정확한 관절중심 아님 · 손가락/근육/힘·일 검증 전 · 구속 향상 후보 아님', 20)
        label(d, (20, 891), 'Prime BaseballThrow_001 / OpenBiomechanics · 제공360Hz · 느린 재생/정지 · camera17은 새 손 관측', 17, '#adc0ca')
        writer.send(np.asarray(canvas).tobytes())
        if index == 31+7*12+6:
            canvas.save(OUT/'alignment_preview_978.png')
    writer.close()
    reader = ff.read_frames(str(output), pix_fmt='rgb24'); metadata = next(reader); count = sum(1 for _ in reader); reader.close()
    if count != len(sequence):
        raise RuntimeError('Output decode mismatch')
    manifest = {'video': str(output.resolve()), 'sha256': sha(output), 'frames': count, 'fps': 30, 'seconds': count/30.,
                'size': list(metadata['size']), 'all_frames_decoded': True, 'input_sha256': source_hashes,
                'results_sha256': sha(OUT/'results.json'), 'displayed_time_states': chosen,
                'calibration_adopted': False, 'athlete_motion_modified': False, 'actual_performance_gain': False}
    (OUT/'video_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'video': str(output), 'frames': count}))


if __name__ == '__main__':
    run()
