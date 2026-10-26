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


def main():
    source = ROOT / 'artifacts/constrained-motion-search'
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    baseline = report['baseline_fine']
    best = report['selected_fine']
    best_gain = best['speed_kmh'] - baseline['speed_kmh']
    selected = 'selected' if best_gain > 1e-8 else 'diagnostic'
    runs = [baseline, report[f'{selected}_fine']]
    states = [state_points(source / f'state-{c}-sub2.npz') for c in ['baseline', selected]]
    for run, (_, speeds, _) in zip(runs, states):
        np.testing.assert_allclose(speeds[-1], run['speed_kmh'], atol=1e-8)
    dt = states[0][2]
    n = len(states[0][0])
    assert n == len(states[1][0]) and dt == states[1][2]
    out = ROOT / 'artifacts/motion-search-video'
    out.mkdir(parents=True, exist_ok=True)
    dataset = ROOT.parent / 'openbiomechanics/baseball_pitching/data'
    landmarks = pd.read_csv(dataset / 'full_sig/landmarks.csv')
    g = landmarks[landmarks.session_pitch.eq('3034_4')].sort_values('time')
    times = float(g.fp_10_time.iloc[0]) + np.arange(n) * dt
    end = float(g.BR_time.iloc[0])
    np.testing.assert_allclose(times[-1], end, atol=1e-9)
    names = sorted(set(sum(([a, b] for a, b in BONES + ARM), [])))
    observed = {name: CubicSpline(regular_time(g), vector(g, name))(times) for name in names}
    meta = pd.read_csv(dataset / 'metadata.csv')
    recorded = float(meta[meta.session_pitch.eq('3034_4')].pitch_speed_mph.iloc[0]) * 1.609344
    passed = bool(runs[1]['all_pass'])
    delta = runs[1]['speed_kmh'] - baseline['speed_kmh']
    work_excess = np.asarray(runs[1]['positive_work_per_dof_j']) - np.asarray(baseline['positive_work_per_dof_j'])
    max_excess = float(work_excess.max())
    total_delta = runs[1]['total_positive_component_work_j'] - baseline['total_positive_component_work_j']
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
    video = out / 'baseball_constrained_motion_search_3034_4.mp4'
    writer = imageio_ffmpeg.write_frames(str(video), (1280, 720), fps=24, codec='libx264', pix_fmt_out='yuv420p', output_params=['-crf', '18', '-movflags', '+faststart'])
    writer.send(None)
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

            text(24, 8, '야구 · 선수 ID 2 / 3034_4 · ' + ('팔 확대' if close else '각도·팔 위치·타이밍 동시 조정'), 26)
            for side in range(2):
                x, color = 24+640*side, '#44cce9' if side == 0 else '#ffab5b'
                text(x, 46, '기준 시뮬레이션' if side == 0 else ('제약 통과 동작 · 실제 향상 미검증' if passed else '제약 탈락 예시 · 향상 권고 아님'), 22, color)
                text(x, 78, f"현재 공 속도 {states[side][1][i]:.2f} km/h", 24, color)
                text(x, 113, f"모델 BR 구속 {states[side][1][-1]:.3f} km/h" + (f" · 변화 {delta:+.3f}" if side else ''), 18)
            draw.rectangle((0, 564, 1280, 720), fill='#101724')
            text(24, 570, f'BR 기준 {(times[i]-end)*1000:+.1f} ms · 느린 재생', 20)
            text(664, 570, f"자세 제한 {'통과' if runs[1]['pose_pass'] else '탈락'} · 힘·일 제한 {'통과' if runs[1]['budget_gate']['strict_model_budget_pass'] else '탈락'}", 20)
            text(24, 602, f'제약 통과 최고 증가 {best_gain:+.3f} km/h · 표시 동작 증가 {delta:+.3f} km/h', 22)
            text(24, 633, f"최대 회전 {runs[1]['max_rotation_deg']:.2f}° · 공 중심 이동 {runs[1]['max_ball_displacement_mm']:.1f}mm · 양의 일 변화 {total_delta:+.2f}J", 18)
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
    manifest = dict(video=str(video), frames=len(indices)*2, fps=24, duration_s=20, selected_case_id=selected, selected_pass=passed, best_accepted_speed_gain_kmh=best_gain, baseline_speed_kmh=states[0][1][-1], selected_speed_kmh=states[1][1][-1], recorded_speed_kmh=recorded, conditional_baseline_gap_kmh=recorded-states[0][1][-1], zero_error_claim=False, source_report=str(source/'report.json'), decode_exit_code=decode.returncode, model_state_indices=indices*2)
    manifest.update(physical_parameters=runs[1]['physical_parameters'], max_rotation_deg=runs[1]['max_rotation_deg'], max_ball_displacement_mm=runs[1]['max_ball_displacement_mm'], release_direction_deg=runs[1]['release_direction_deg'], force_work_pass=runs[1]['budget_gate']['strict_model_budget_pass'], pose_pass=runs[1]['pose_pass'], positive_work_change_j=total_delta, candidate_is_actual_intervention_validated=False)
    (out/'video_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'model_state_indices'}, indent=2))


if __name__ == '__main__':
    main()
