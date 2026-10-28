"""Show measured joints and before/after geometry calibration as pitching scenes."""
import json
import subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from coupled_body_search import positions
from baseline_calibration import endpoint
from render_speed_journey import imageio_ffmpeg
from same_pitch_audit import OUT

BONES=[('rear_hip','lead_hip'),('rear_hip','rear_knee_jc'),('rear_knee_jc','rear_ankle_jc'),
       ('lead_hip','lead_knee_jc'),('lead_knee_jc','lead_ankle_jc'),('rear_hip','shoulder_jc'),
       ('lead_hip','glove_shoulder_jc'),('shoulder_jc','glove_shoulder_jc'),('shoulder_jc','elbow_jc'),
       ('elbow_jc','wrist'),('wrist','ball_center'),('glove_shoulder_jc','glove_elbow_jc'),('glove_elbow_jc','glove_wrist_jc')]


def run(output=OUT):
    output=Path(output);pitch='3034_3';folder=output/pitch
    report=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    row=next(r for r in report['records'] if r['pitch']==pitch)
    models={};states={};speeds={};points={}
    for mode in ('nominal','frozen_skeleton'):
        models[mode]=mujoco.MjModel.from_xml_path(str(folder/(mode+'.xml')))
        states[mode]=np.load(folder/(mode+'.npz'))
        z=states[mode];names,points[mode]=positions(models[mode],z['qpos'])
        speeds[mode]=np.array([np.linalg.norm(endpoint(models[mode],q,v))*3.6 for q,v in zip(z['qpos'],z['qvel'])])
    n=len(points['nominal']);observed=states['nominal']['observed_positions'];times=states['nominal']['time_s']
    b=row['models']['nominal']['resolutions']['1'];a=row['models']['frozen_skeleton']['resolutions']['1']
    sm=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="440"/><headlight ambient=".6 .6 .6" diffuse=".8 .8 .8"/></visual><worldbody><light pos="0 -3 5"/><geom type="plane" size="8 8 .1" rgba=".12 .16 .20 1"/></worldbody></mujoco>')
    sd=mujoco.MjData(sm);mujoco.mj_forward(sm,sd);renderer=mujoco.Renderer(sm,height=440,width=640)
    focus=np.mean(points['nominal'].reshape(-1,3),axis=0);focus[2]=.95
    def add(scene,kind,size,pos,color):
        g=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(g,kind,np.asarray(size,float),np.asarray(pos,float),np.eye(3).ravel(),np.asarray(color,np.float32));return g
    def view(i,azimuth):
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=focus;cam.distance=3.8;cam.azimuth=azimuth;cam.elevation=-9
        renderer.update_scene(sd,camera=cam);scene=renderer.scene
        for ps,col,rad in ((observed[i],[.8,.84,.9,.7],.010),(points['nominal'][i],[.12,.72,1,.6],.035),(points['frozen_skeleton'][i],[1,.47,.12,.85],.022)):
            p=dict(zip(names,ps))
            for x,y in BONES:
                if not np.isfinite([p[x],p[y]]).all():continue
                g=add(scene,mujoco.mjtGeom.mjGEOM_CAPSULE,[rad,.1,0],[0,0,0],col)
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,rad,p[x],p[y])
            for name in names:
                if not np.isfinite(p[name]).all():continue
                r=.0364 if name=='ball_center' else rad*1.1
                add(scene,mujoco.mjtGeom.mjGEOM_SPHERE,[r]*3,p[name],col)
            hip=(p['rear_hip']+p['lead_hip'])/2;shoulder=(p['shoulder_jc']+p['glove_shoulder_jc'])/2
            head=shoulder+.22*(shoulder-hip)/np.linalg.norm(shoulder-hip)
            add(scene,mujoco.mjtGeom.mjGEOM_SPHERE,[.08]*3,head,col)
        return Image.fromarray(renderer.render().copy())
    fonts={}
    def font(size):
        if size not in fonts:fonts[size]=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size)
        return fonts[size]
    indices=np.rint(np.linspace(0,n-1,192)).astype(int).tolist()+[n-1]*48
    video=output/'same_pitch_body_calibration.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=24,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart']);writer.send(None)
    try:
        for frame,i in enumerate(indices):
            im=Image.new('RGB',(1280,720),'#101724');im.paste(view(i,90),(0,120));im.paste(view(i,0),(640,120))
            d=ImageDraw.Draw(im)
            def text(x,y,s,size=21,color='#edf3ff'):d.text((x,y),s,font=font(size),fill=color)
            text(25,8,'동일 투구 3034_3 · 골격 기하 보정 전후 · 구속 향상 개입 아님',25)
            text(25,48,'회색: 실측 관절   파랑: 명목 모델   주황: 첫 투구로 기하를 동결한 모델',21)
            text(25,84,f'보정 전 순간속도 {speeds["nominal"][i]:.3f} km/h',21,'#4ac8ff')
            text(665,84,f'보정 후 순간속도 {speeds["frozen_skeleton"][i]:.3f} km/h',21,'#ffac61')
            d.rectangle((0,560,1280,720),fill='#101724')
            sh=np.linalg.norm(points['frozen_skeleton'][i,names.index('shoulder_jc')]-points['nominal'][i,names.index('shoulder_jc')])*1000
            wr=np.linalg.norm(points['frozen_skeleton'][i,names.index('wrist')]-points['nominal'][i,names.index('wrist')])*1000
            text(25,566,f'BR {(times[i]-times[-1])*1000:+.1f}ms · 어깨 위치 차이 {sh:.2f}mm · 손목 위치 차이 {wr:.2f}mm',20)
            text(25,598,f'모델 BR: {b["speed_kmh"]:.3f} → {a["speed_kmh"]:.3f} km/h · 기록 {row["recorded_speed_kmh"]:.3f} km/h',22)
            text(25,630,f'기록과 차이: {b["record_gap"]["value"]:.3f} → {a["record_gap"]["value"]:.3f} km/h · 아직 큰 차이',21,'#ffbf85')
            text(25,662,f'손목 실측 위치 RMSE {b["observed_joint_rmse_mm"]["wrist"]:.2f} → {a["observed_joint_rmse_mm"]["wrist"]:.2f}mm · 기본 모델 채택 보류',18)
            text(25,691,'약 50배 느린 재생 · 모델별 기준 힘 재계산 · 같은 능력의 향상 비교 아님 · 머리는 표시용',16,'#b1bed2')
            writer.send(np.asarray(im))
            if frame in (0,96,191):im.save(output/f'calibration_frame_{frame}.png')
    finally:writer.close();renderer.close()
    check=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    if check.returncode or check.stderr:raise RuntimeError(check.stderr)
    manifest={'pitch':pitch,'frames':240,'fps':24,'duration_s':10,'graph_frames':0,'frame_indices':indices,
              'actual_athlete_gain_video':False,'geometry_calibration':True,'decode_exit_code':check.returncode}
    (output/'video_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(str(video))


if __name__=='__main__':run()
