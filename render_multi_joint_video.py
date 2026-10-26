"""Render saved RK4 states; never substitute measured ball speed for model speed."""
import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline

from coupled_payload import model_for
from pilot import regular_time, vector
from render_speed_journey import ARM, BONES, imageio_ffmpeg

ROOT = Path(__file__).resolve().parent


def state_points(path):
    with np.load(path) as f:
        q, v, lengths, dt = f['qpos'], f['qvel'], f['lengths'], float(f['dt'])
    model = model_for(lengths, dt)
    data = mujoco.MjData(model)
    points, speeds = [], []
    for pos, vel in zip(q, v):
        data.qpos[:] = pos
        data.qvel[:] = vel
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, model.site('ball_center').id)
        speeds.append(float(np.linalg.norm(jp @ vel) * 3.6))
        points.append(dict(shoulder_jc=data.xpos[model.body('upper').id].copy(),
                           elbow_jc=data.xpos[model.body('forearm').id].copy(),
                           wrist_jc=data.xpos[model.body('hand').id].copy(),
                           hand_jc=data.site_xpos[model.site('ball_center').id].copy()))
    return points, speeds, dt


def main(case_id=None):
    source = ROOT / 'artifacts/multi-joint-timing'
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    selected = case_id or report['selected_case_id']
    baseline = report.get('baseline_case_id', 's0_e0_w0')
    cases = [baseline, selected]
    runs = [next(r for r in report['runs'] if r['case_id'] == c and r['substeps'] == 2) for c in cases]
    states = [state_points(source / f'state-{c}-sub2.npz') for c in cases]
    for run, (_, speeds, _) in zip(runs, states):
        np.testing.assert_allclose(speeds[-1], run['speed_kmh'], atol=1e-9)
    dt = states[0][2]
    n = len(states[0][0])
    assert n == len(states[1][0]) and dt == states[1][2]
    out = ROOT / 'artifacts/multi-joint-video'
    if case_id:
        out = out / case_id
    out.mkdir(parents=True, exist_ok=True)
    dataset = ROOT.parent / 'openbiomechanics/baseball_pitching/data'
    landmarks = pd.read_csv(dataset / 'full_sig/landmarks.csv')
    g = landmarks[landmarks.session_pitch.eq('3034_1')].sort_values('time')
    times = float(g.fp_10_time.iloc[0]) + np.arange(n) * dt
    end = float(g.BR_time.iloc[0])
    np.testing.assert_allclose(times[-1], end, atol=1e-9)
    names = sorted(set(sum(([a, b] for a, b in BONES + ARM), [])))
    observed = {name: CubicSpline(regular_time(g), vector(g, name))(times) for name in names}
    meta = pd.read_csv(dataset / 'metadata.csv')
    recorded = float(meta[meta.session_pitch.eq('3034_1')].pitch_speed_mph.iloc[0]) * 1.609344
    passed = runs[1]['budget_gate']['strict_model_budget_pass'] and runs[1]['small_motion_pass']
    fine = [r for r in report['runs'] if r['substeps'] == 2 and r['case_id'] != baseline]
    accepted = sum(r['budget_gate']['strict_model_budget_pass'] and r['small_motion_pass'] and r['speed_difference_from_baseline_kmh'] > 0 for r in fine)
    work_excess = np.asarray(runs[1]['positive_work_per_dof_j']) - np.asarray(runs[0]['positive_work_per_dof_j'])
    max_excess = float(work_excess.max())
    total_delta = runs[1]['total_positive_component_work_j'] - runs[0]['total_positive_component_work_j']
    scene_model = mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="420"/><headlight ambient=".5 .5 .5" diffuse=".8 .8 .8"/></visual><worldbody><light pos="0 -3 5"/><geom type="plane" size="5 5 .1" rgba=".10 .13 .18 1"/></worldbody></mujoco>')
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)
    renderer = mujoco.Renderer(scene_model, height=420, width=640)
    colors = [[.18, .78, .93, 1], [1., .50, .19, 1]]

    def draw_body(points, side, close, focus):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation = (125 if close else 145), -15
        camera.lookat[:] = focus if close else [.8, 0, .85]
        camera.distance = 2.0 if close else 3.8
        renderer.update_scene(scene_data, camera=camera)
        scene = renderer.scene
        gray = [.35, .43, .53, 1]

        def geom(kind, size, pos, color):
            item = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(item, kind, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(color, np.float32))
            scene.ngeom += 1
            return item

        for a, b in BONES + ARM:
            arm = (a, b) in ARM
            item = geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [.022, .1, 0], [0, 0, 0], colors[side] if arm else gray)
            mujoco.mjv_connector(item, mujoco.mjtGeom.mjGEOM_CAPSULE, .027 if arm else .033, points[a], points[b])
        for name in names:
            if name != 'hand_jc':
                geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.038]*3, points[name], colors[side] if name in ('shoulder_jc', 'elbow_jc', 'wrist_jc') else gray)
        geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.0364]*3, points['hand_jc'], [1, 1, .92, 1])
        return Image.fromarray(renderer.render().copy())

    fonts = {s: ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', s) for s in (16, 18, 20, 22, 24, 26)}
    indices = np.rint(np.linspace(0, n-1, 168)).astype(int).tolist() + [n-1]*72
    video = out / 'baseball_multi_joint_timing_3034_1.mp4'
    writer = imageio_ffmpeg.write_frames(str(video), (1280, 720), fps=24, codec='libx264', pix_fmt_out='yuv420p', output_params=['-crf', '18', '-movflags', '+faststart'])
    writer.send(None)
    shifts = np.asarray(runs[1]['shifts_samples']) * dt * 1000
    for close in (False, True):
        for j, i in enumerate(indices):
            measured = {name: observed[name][i] for name in names}
            focus = (measured['shoulder_jc'] + measured['wrist_jc']) / 2
            im = Image.new('RGB', (1280, 720), '#101724')
            for side in range(2):
                im.paste(draw_body({**measured, **states[side][0][i]}, side, close, focus), (640*side, 144))
            draw = ImageDraw.Draw(im)

            def text(x, y, value, size=20, color='#eaf2ff'):
                draw.text((x, y), value, font=fonts[size], fill=color)

            text(24, 8, '야구 · 선수 ID 2 / 3034_1 · ' + ('팔 확대' if close else '여러 관절 입력 타이밍 비교'), 26)
            for side in range(2):
                x, color = 24+640*side, '#44cce9' if side == 0 else '#ffab5b'
                text(x, 46, '기준 시뮬레이션' if side == 0 else ('제약 통과 후보' if passed else '제약 탈락 예시 · 향상 권고 아님'), 22, color)
                text(x, 78, f"현재 공 속도 {states[side][1][i]:.2f} km/h", 24, color)
                text(x, 113, f"모델 BR 구속 {states[side][1][-1]:.3f} km/h" + (f" · 변화 {runs[1]['speed_difference_from_baseline_kmh']:+.3f}" if side else ''), 18)
            draw.rectangle((0, 564, 1280, 720), fill='#101724')
            text(24, 570, f'BR 기준 {(times[i]-end)*1000:+.1f} ms · 느린 재생', 20)
            text(664, 570, f'타이밍(ms) 어깨 {shifts[0]:+.3f} / 팔꿈치 {shifts[1]:+.3f} / 손목 {shifts[2]:+.3f}', 18)
            text(24, 602, f'구속 증가 + 제약 통과 {accepted}개 · 최대 양의 일 성분 증가 {max_excess:+.3f} J', 20)
            text(24, 633, f'총 양의 성분 일 변화 {total_delta:+.3f} J · 회색 몸통·다리: 관찰 동작', 18)
            text(24, 660, f'오차 0 아님 · 기록 {recorded:.2f} - 기준 모델 {states[0][1][-1]:.2f} = {recorded-states[0][1][-1]:.2f} km/h', 22, '#ffd083')
            text(24, 694, '기록 측정 시점 미확인 / 15cm 그립·일반 관성 / 기계적 제약 검사이며 실제 근력·구속 향상 검증 아님', 16, '#b6c4d5')
            writer.send(np.asarray(im))
            if close and j == 167:
                im.save(out / 'baseball_release.png')
            if not close and j == 0:
                im.save(out / 'baseball_start.png')
    writer.close()
    renderer.close()
    decode = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video), '-f', 'null', '-'], capture_output=True, text=True)
    assert decode.returncode == 0 and not decode.stderr, decode.stderr
    manifest = dict(video=str(video), frames=len(indices)*2, fps=24, duration_s=20, selected_case_id=selected, selected_pass=passed, accepted_speed_gain_cases=accepted, baseline_speed_kmh=states[0][1][-1], selected_speed_kmh=states[1][1][-1], recorded_speed_kmh=recorded, conditional_baseline_gap_kmh=recorded-states[0][1][-1], zero_error_claim=False, source_report=str(source/'report.json'), decode_exit_code=decode.returncode, model_state_indices=indices*2)
    (out/'video_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'model_state_indices'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', help='Render an explicitly named diagnostic case without changing search selection.')
    main(parser.parse_args().case)
