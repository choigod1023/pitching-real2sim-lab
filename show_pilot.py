"""Replay an explicitly selected illustrative held-out pitch and save a GIF."""
import argparse,json
import numpy as np
import pandas as pd
import mujoco
from PIL import Image
from kinematics import Trial,Skeleton,draw_reference
from pilot import DATA,OUT,vector,change,regular_time
from preview import audit,launch


def build():
    report=json.loads((OUT/'results.json').read_text())
    results=pd.read_csv(OUT/'heldout.csv')
    # Illustration rule: lowest athlete ID with a nonzero selected edit.
    # Not chosen for largest gain. Main results include abstentions and failures.
    possible=results[(results.method=='personalized')&((results.angle_deg!=0)|(results.delay_ms!=0))]
    row=possible.sort_values(['user','session_pitch']).iloc[0]
    df=pd.read_csv(DATA/'full_sig/landmarks.csv')
    df=df[df.session_pitch==row.session_pitch].sort_values('time')
    meta=pd.read_csv(DATA/'metadata.csv').set_index('session_pitch').loc[row.session_pitch]
    throws=pd.read_csv(DATA/'poi/poi_metrics.csv').set_index('session_pitch').loc[row.session_pitch,'p_throws']
    throwing='R' if str(throws).upper()=='R' else 'L'; other='L' if throwing=='R' else 'R'
    joints={}
    for side,prefix,leg in [(throwing,'','rear'),(other,'glove_','lead')]:
        for joint,col in [('hip',leg+'_hip'),('knee',leg+'_knee_jc'),('ankle',leg+'_ankle_jc'),('shoulder',prefix+'shoulder_jc'),('elbow',prefix+'elbow_jc'),('wrist',prefix+'wrist_jc')]:
            joints[side+joint]=vector(df,col)
    joints['pelvis']=(joints['Lhip']+joints['Rhip'])/2
    joints['chest']=(joints['Lshoulder']+joints['Rshoulder'])/2
    origin=joints['pelvis'][0].copy();origin[2]=0
    original={k:v-origin for k,v in joints.items()}
    t=regular_time(df);rate=360.
    baseline=Trial(DATA/str(meta.filename_new),rate,original.copy(),original,0)
    cw,ch=change(t,vector(df,'shoulder_jc'),vector(df,'elbow_jc'),vector(df,'wrist_jc'),vector(df,'hand_jc'),float(df.BR_time.iloc[0]),float(row.angle_deg),float(row.delay_ms))
    edited={k:v.copy() for k,v in original.items()};edited[throwing+'wrist']=cw-origin
    candidate=Trial(baseline.path,rate,edited.copy(),edited,0)
    details=row.to_dict();details['illustration_selection_rule']='lowest athlete ID with nonzero selection; first heldout pitch'
    details['observed_original_ball_speed_mph']=float(meta.pitch_speed_mph)
    details['release_frame']=int(np.argmin(np.abs(t-float(df.BR_time.iloc[0]))))
    (OUT/'illustration.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    return baseline,Skeleton(candidate),details


def movie(reference,skeleton):
    camera=mujoco.MjvCamera();camera.lookat[:]=[.5,0,1.]
    camera.distance=5.4;camera.azimuth=0;camera.elevation=-15
    refscene=mujoco.MjvScene(skeleton.model,maxgeom=200)
    frames=[]
    with mujoco.Renderer(skeleton.model,360,640) as renderer:
        for i in np.linspace(0,reference.count-1,90,dtype=int):
            skeleton.pose(i,1.3);renderer.update_scene(skeleton.data,camera=camera)
            draw_reference(refscene,reference,i)
            for k in range(refscene.ngeom):
                src=refscene.geoms[k];dst=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(dst,src.type,src.size.astype(float),src.pos.astype(float),src.mat.astype(float).ravel(),src.rgba)
                dst.size[:]=src.size;dst.emission=.25;renderer.scene.ngeom+=1
            image=Image.fromarray(renderer.render())
            frames.append(image)
    frames[0].save(OUT/'comparison.gif',save_all=True,append_images=frames[1:],duration=67,loop=0)
    frames[65].save(OUT/'comparison.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');p.add_argument('--smoke-seconds',type=float,default=0);args=p.parse_args()
    baseline,skeleton,details=build();movie(baseline,skeleton)
    print(json.dumps(details,indent=2),flush=True)
    if not args.check_only:
        caption=f'예시: 선수 {details["user"]} / 투구 {details["session_pitch"]}\n방향 변경 {details["angle_deg"]:+.1f}도 · 타이밍 지연 {details["delay_ms"]:+.0f}ms\n계산된 손 속도 변화 {details["delta_hand_speed_m_s"]:+.2f}m/s — 구속 예측 아님\n이는 평균 결과를 대표하거나 실제 개선을 입증하는 사례가 아닙니다.'
        launch(skeleton,audit(skeleton),args.smoke_seconds,reference=baseline,caption=caption,autoplay=True)


if __name__=='__main__':main()
