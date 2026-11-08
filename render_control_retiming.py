"""Pitching-body scenes with physically simulated baseline/candidate overlays."""
import json
import subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from coupled_body_search import groups, positions
from render_speed_journey import imageio_ffmpeg
from control_retiming import OUT

BONES = [('rear_hip','lead_hip'), ('rear_hip','rear_knee_jc'), ('rear_knee_jc','rear_ankle_jc'),
         ('lead_hip','lead_knee_jc'), ('lead_knee_jc','lead_ankle_jc'),
         ('rear_hip','shoulder_jc'), ('lead_hip','glove_shoulder_jc'),
         ('shoulder_jc','glove_shoulder_jc'), ('shoulder_jc','elbow_jc'), ('elbow_jc','wrist'),
         ('wrist','ball_center'), ('glove_shoulder_jc','glove_elbow_jc'), ('glove_elbow_jc','glove_wrist_jc')]


def render(output=OUT):
    output = Path(output)
    report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    z = np.load(output/'selected.npz')
    m = mujoco.MjModel.from_xml_path(str(output/'source/model.xml'))
    names, bp = positions(m, z['baseline_q']); _, cp = positions(m, z['candidate_q'])
    gs = groups(m); delta = np.zeros((len(bp), m.nv))
    for i, (a, b) in enumerate(zip(z['baseline_q'], z['candidate_q'])):
        mujoco.mj_differentiatePos(m, delta[i], 1, a, b)
    def speed(qs, vs):
        d = mujoco.MjData(m); out = []
        for q, v in zip(qs, vs):
            d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(m, d)
            jp = np.zeros((3,m.nv)); jr = jp.copy()
            mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
            out.append(np.linalg.norm(jp@v)*3.6)
        return np.array(out)
    bs = speed(z['baseline_q'], z['baseline_v']); cs = speed(z['candidate_q'], z['candidate_v'])
    model = mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="440"/><rgba haze=".06 .08 .12 1"/><headlight ambient=".6 .6 .6" diffuse=".8 .8 .8"/></visual><worldbody><light pos="0 -3 5"/><geom type="plane" size="8 8 .1" rgba=".12 .16 .20 1"/></worldbody></mujoco>')
    data = mujoco.MjData(model); mujoco.mj_forward(model,data)
    renderer = mujoco.Renderer(model,height=440,width=640)
    focus = np.mean(bp.reshape(-1,3),axis=0); focus[2] = .95
    def geom(scene, kind, size, pos, color):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.asarray(size,float),np.asarray(pos,float),np.eye(3).ravel(),np.asarray(color,np.float32))
        scene.ngeom += 1
        return g
    def scene(i, azimuth):
        camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = focus; camera.azimuth = azimuth; camera.elevation = -9; camera.distance = 3.8
        renderer.update_scene(data,camera=camera); sc=renderer.scene
        for points, color, radius in ((z['observed_positions'][i],[.78,.82,.88,.55],.010),
                                      (bp[i],[.12,.72,1,.65],.035),(cp[i],[1,.47,.12,.85],.022)):
            p = dict(zip(names,points))
            for a,b in BONES:
                if not np.isfinite([p[a],p[b]]).all():continue
                g=geom(sc,mujoco.mjtGeom.mjGEOM_CAPSULE,[radius,.1,0],[0,0,0],color)
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,p[a],p[b])
            for name in names:
                if not np.isfinite(p[name]).all():continue
                r=.0364 if name=='ball_center' else radius*1.2
                geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,[r]*3,p[name],color)
            # Head/neck are display-only landmarks inferred from shoulder/hip lines.
            hips=(p['rear_hip']+p['lead_hip'])/2
            shoulders=(p['shoulder_jc']+p['glove_shoulder_jc'])/2
            axis=shoulders-hips; axis/=np.linalg.norm(axis)
            head=shoulders+.22*axis
            g=geom(sc,mujoco.mjtGeom.mjGEOM_CAPSULE,[radius,.1,0],[0,0,0],color)
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,shoulders,head)
            geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,[.085]*3,head,color)
        return Image.fromarray(renderer.render().copy())
    font_cache={}
    def font(size):
        if size not in font_cache:font_cache[size]=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size)
        return font_cache[size]
    selected=report['selected']; base=report['resolutions']['1']['baseline']
    arm=('pitch_shoulder','elbow','marker_roll','wrist_marker_rotation')
    arm_change=sum(selected['costs'][n]['positive_j']-base['costs'][n]['positive_j'] for n in arm)
    indices=np.rint(np.linspace(0,len(bp)-1,192)).astype(int).tolist()+[len(bp)-1]*48
    video=output/'pitching_body_overlay.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=24,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart'])
    writer.send(None)
    try:
        for frame,i in enumerate(indices):
            im=Image.new('RGB',(1280,720),'#101724'); im.paste(scene(i,90),(0,120)); im.paste(scene(i,0),(640,120))
            draw=ImageDraw.Draw(im)
            def text(x,y,t,size=21,color='#edf3ff'):draw.text((x,y),t,font=font(size),fill=color)
            text(25,8,'야구 투구 3034_3 · 앞무릎 토크 입력을 2.78ms 앞당긴 전방 적분',25)
            text(25,48,'회색: 실측 관절',20,'#c7d1e0'); text(295,48,'파랑: 기준 모델',20,'#4ac8ff'); text(570,48,'주황: 타이밍 수정 모델',20,'#ffac61')
            text(25,84,f'기준 순간속도 {bs[i]:.3f} km/h',21,'#4ac8ff')
            text(665,84,f'수정 순간속도 {cs[i]:.3f} km/h',21,'#ffac61')
            text(25,127,'시점 1 · 같은 위치에 겹침',18);text(665,127,'시점 2 · 같은 위치에 겹침',18)
            draw.rectangle((0,560,1280,720),fill='#101724')
            time=(z['time_s'][i]-z['time_s'][-1])*1000
            lead=int(m.joint('lead_knee').qposadr[0]); knee=np.rad2deg(z['candidate_q'][i,lead]-z['baseline_q'][i,lead])
            torso=np.rad2deg(np.linalg.norm(delta[i,gs['torso_rotation']]))
            wrist=np.linalg.norm(cp[i,names.index('wrist')]-bp[i,names.index('wrist')])*1000
            text(25,566,f'BR {time:+.1f}ms · 앞무릎 굽힘 Δ {knee:+.2f}° · 몸통 회전 차이 {torso:.2f}° · 손목 {wrist:.2f}mm',19)
            text(25,598,f'모델 BR: {bs[-1]:.3f} → {cs[-1]:.3f} km/h  ({cs[-1]-bs[-1]:+.3f})',22)
            text(25,631,f'양의 일: 전신 {selected["positive_j"]-base["positive_j"]:+.3f}J · 투구 팔 {arm_change:+.3f}J → 제약 미통과',21,'#ffbf85')
            text(25,662,f'실측 기록 {report["recorded_speed_kmh"]:.3f} km/h · 기준 모델 차이 {report["recorded_speed_kmh"]-bs[-1]:.3f} km/h · 실제 향상 미검증',18)
            text(25,691,'약 50배 느린 재생 · 발 접촉/근육/손가락 이탈 미검증 · 머리/목은 표시용 · 구위/제구 예측 아님',16,'#b1bed2')
            writer.send(np.asarray(im))
            if frame in (0,96,191): im.save(output/f'body_frame_{frame}.png')
    finally:
        writer.close();renderer.close()
    decode=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    if decode.returncode or decode.stderr:raise RuntimeError(decode.stderr)
    manifest={'video':str(video),'fps':24,'frames':len(indices),'duration_s':len(indices)/24,
              'frame_model_indices':indices,'selected_id':report['selected_id'],'graph_frames':0,
              'dynamics': 'Saved forward integration; no animation interpolation of candidate positions',
              'head_neck': 'display-only inferred geometry', 'decode_exit_code':decode.returncode}
    (output/'video_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='frame_model_indices'}))


if __name__=='__main__':render()
