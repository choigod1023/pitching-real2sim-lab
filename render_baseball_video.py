"""Baseball-only video of measured joints versus actual force-driven RK4 rollout."""
import json
import sys
import subprocess
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline
from coupled_payload import model_for, inverse_and_rollout
from torque_arm import derivatives
from pilot import vector, regular_time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'comparison-video/artifacts/video-vendor'))
import imageio_ffmpeg

BONES = [('rear_ankle_jc','rear_knee_jc'),('rear_knee_jc','rear_hip'),
         ('lead_ankle_jc','lead_knee_jc'),('lead_knee_jc','lead_hip'),
         ('rear_hip','lead_hip'),('rear_hip','shoulder_jc'),
         ('lead_hip','glove_shoulder_jc'),('shoulder_jc','glove_shoulder_jc'),
         ('glove_shoulder_jc','glove_elbow_jc'),('glove_elbow_jc','glove_wrist_jc')]
ARM = [('shoulder_jc','elbow_jc'),('elbow_jc','wrist_jc'),('wrist_jc','hand_jc')]


def elbow_angle(points):
    a = points['shoulder_jc']-points['elbow_jc']
    b = points['wrist_jc']-points['elbow_jc']
    return float(180-np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))


def main():
    out=ROOT/'artifacts/baseball-video';out.mkdir(parents=True,exist_ok=True)
    source=ROOT.parent/'grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz'
    with np.load(source) as f:
        q=f['qpos'].copy();dt=float(f['dt']);lengths=f['lengths'].copy()
    m=model_for(lengths,dt);m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    reference_result, loads=inverse_and_rollout(m,q,dt)
    velocity,_=derivatives(m,q,dt)
    d=mujoco.MjData(m);d.qpos[:]=q[0];d.qvel[:]=velocity[0]
    states=[];speeds=[];points=[]
    for i in range(len(q)):
        mujoco.mj_forward(m,d)
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id)
        states.append(d.qpos.copy());speeds.append(np.linalg.norm(jp@d.qvel)*3.6)
        points.append(dict(shoulder_jc=d.xpos[m.body('upper').id].copy(),
                           elbow_jc=d.xpos[m.body('forearm').id].copy(),
                           wrist_jc=d.xpos[m.body('hand').id].copy(),
                           hand_jc=d.site_xpos[m.site('ball_center').id].copy()))
        if i<len(q)-1:
            d.qfrc_applied[:]=loads[i];mujoco.mj_step(m,d)
            assert not d.warning.number.any() and np.isfinite(d.qpos).all()
    np.testing.assert_allclose(speeds[-1],reference_result['conditional_release_speed_kmh'],atol=1e-9)
    data=ROOT.parent/'openbiomechanics/baseball_pitching/data'
    all_landmarks=pd.read_csv(data/'full_sig/landmarks.csv')
    g=all_landmarks[all_landmarks.session_pitch.eq('3034_1')].sort_values('time')
    start,end=float(g.fp_10_time.iloc[0]),float(g.BR_time.iloc[0])
    times=start+np.arange(len(q))*dt
    np.testing.assert_allclose(times[-1],end,atol=1e-9)
    names=sorted(set(sum(([a,b] for a,b in BONES+ARM),[])))
    observed={n:CubicSpline(regular_time(g),vector(g,n))(times) for n in names}
    meta=pd.read_csv(data/'metadata.csv');radar=float(meta[meta.session_pitch.eq('3034_1')].pitch_speed_mph.iloc[0])*1.609344
    # A separate scene contains only visual geometry; all simulated coordinates above
    # come from the unmodified physical model, not animation interpolation.
    scene_model=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="560"/><rgba haze="0.06 0.08 0.12 1"/><headlight ambient=".5 .5 .5" diffuse=".8 .8 .8"/></visual><worldbody><light pos="0 -3 5"/><geom type="plane" size="5 5 .1" rgba=".10 .13 .18 1"/></worldbody></mujoco>')
    sd=mujoco.MjData(scene_model);mujoco.mj_forward(scene_model,sd)
    renderer=mujoco.Renderer(scene_model,height=560,width=640)
    def add_geom(scene,kind,size,pos,color):
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,kind,np.asarray(size,float),np.asarray(pos,float),np.eye(3).ravel(),np.asarray(color,np.float32))
        scene.ngeom+=1
        return geom
    def body_image(p,sim,close,focus):
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth=125 if close else 145;cam.elevation=-15
        cam.lookat[:]=focus if close else [.8,0,.85]
        cam.distance=1.8 if close else 3.3
        renderer.update_scene(sd,camera=cam)
        sc=renderer.scene
        gray=[.35,.43,.53,1];color=[1.,.50,.19,1] if sim else [.18,.78,.93,1]
        for a,b in BONES+ARM:
            col=color if (a,b) in ARM else gray
            geom=add_geom(sc,mujoco.mjtGeom.mjGEOM_CAPSULE,[.022,.1,0],[0,0,0],col)
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.027 if (a,b) in ARM else .033,p[a],p[b])
        for n in names:
            if sim and n == "hand_jc":continue
            add_geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,[.038]*3,p[n],color if n in ('shoulder_jc','elbow_jc','wrist_jc') else gray)
        if sim:add_geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,[.0364]*3,p['hand_jc'],[1,1,.92,1])
        return Image.fromarray(renderer.render().copy())
    font=lambda s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s)
    fps=24;frames=[];indices=np.rint(np.linspace(0,len(q)-1,144)).astype(int).tolist()+[len(q)-1]*36
    video=out/'baseball_3034_1.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=fps,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart'])
    writer.send(None)
    for close in (False,True):
        for j,i in enumerate(indices):
            p={n:observed[n][i].copy() for n in names};s={**p,**points[i]}
            focus=(p['shoulder_jc']+p['wrist_jc'])/2
            im=Image.new('RGB',(1280,720),'#101724');im.paste(body_image(p,False,close,focus),(0,95));im.paste(body_image(s,True,close,focus),(640,95))
            draw=ImageDraw.Draw(im)
            def label(x,y,t,size=24,color='#eaf2ff'):draw.text((x,y),t,font=font(size),fill=color)
            label(28,14,'야구 투구 3034_1 · '+('팔 확대' if close else '관절 동작 비교'),28)
            label(28,57,'실측 관절 동작',24,'#44cce9');label(665,57,'힘 기반 시뮬레이션',24,'#ffab5b')
            label(28,113,f'실측 구속  {radar:.1f} km/h',27)
            label(665,113,f'공 추정 속도  {speeds[i]:.1f} km/h',27)
            label(28,610,f'팔꿈치 굽힘  {elbow_angle(p):.1f}°',22)
            label(665,610,f'팔꿈치 굽힘  {elbow_angle(s):.1f}°',22)
            label(28,660,f'BR 기준  {(times[i]-end)*1000:+.0f} ms  |  느린 재생',19)
            label(665,660,'몸통·다리: 관측  /  팔·공: 계산',19)
            label(28,692,'공 그립 15cm 가정 · 구속 증가 검증 전',16,'#a4b2c6')
            writer.send(np.asarray(im))
            if j in (0,48,96,143):frames.append(im.resize((960,540)))
            if close and j==143:im.save(out/'baseball_release.png')
    writer.close();renderer.close()
    frames[0].save(out/'baseball_preview.gif',save_all=True,append_images=frames[1:],duration=600,loop=0)
    proc=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert proc.returncode==0 and not proc.stderr,proc.stderr
    np.savez(out/'rollout.npz',qpos=np.array(states),speed_kmh=speeds,times=times)
    report={'pitch':'3034_1','observed_speed_kmh':radar,'model_release_speed_kmh':float(speeds[-1]),
            'frames':len(indices)*2,'fps':fps,'duration_s':len(indices)*2/fps,'decode_exit_code':proc.returncode,
            'model':reference_result,'source_reference':str(source),'scope':'Measured body context; actual RK4 arm/hand/attached-ball integration, no state resets. Stops at BR. No optimized motion, capacity match, measured ball trajectory, or validated gain.'}
    (out/'video_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
