"""Observed body context + actual simulated arm frames, no invented anatomy."""
import json,hashlib,sys
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from coupled_payload import model_for
from trace_error_audit import kinematics
from pilot import regular_time,vector
from render_speed_journey import BONES,ARM,imageio_ffmpeg

ROOT=Path(__file__).resolve().parent
FONT='C:/Windows/Fonts/malgun.ttf'

def relative_angle(quaternions):
    q=np.asarray(quaternions)
    if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all():raise ValueError('Expected finite Nx4 wxyz')
    r=Rotation.from_quat(q[:,[1,2,3,0]])
    return np.degrees((r[0].inv()*r).magnitude())

def main():
    out=ROOT/'artifacts/joint-detail';out.mkdir(parents=True,exist_ok=True)
    source=ROOT.parent/'trace-error-audit/artifacts/trace-error/3034_4-46080-pad1-foh-sub1.npz'
    dataset=ROOT.parent/'openbiomechanics/baseball_pitching/data'
    with np.load(source) as z:
        ids=np.rint(np.linspace(0,len(z['qpos'])-1,241)).astype(int)
        q=z['qpos'][ids];v=z['qvel'][ids];reference=z['qreference'][ids];reference_v=z['vreference'][ids];dt=float(z['dt']);m=model_for(z['lengths'],dt)
    p,vel,hand=kinematics(m,q,v);speed=np.linalg.norm(vel[:,3],axis=1)*3.6
    rp,rv,rhand=kinematics(m,reference,reference_v)
    position_delta=np.linalg.norm(p-rp,axis=2)*1000
    def rotation_delta(a,b):
        return np.degrees((Rotation.from_quat(a[:,[1,2,3,0]]).inv()*Rotation.from_quat(b[:,[1,2,3,0]])).magnitude())
    angle_delta=np.column_stack([rotation_delta(reference[:,3:7],q[:,3:7]),np.degrees(q[:,7]-reference[:,7]),np.degrees(q[:,8]-reference[:,8]),rotation_delta(reference[:,9:13],q[:,9:13])])
    assert abs(speed[-1]-113.6466512019)<1e-6
    data=mujoco.MjData(m);frames=[]
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_forward(m,data)
        frames.append([data.xmat[m.body(n).id].reshape(3,3).copy() for n in ('upper','forearm','hand')])
    frames=np.array(frames)
    landmarks=pd.read_csv(dataset/'full_sig/landmarks.csv');g=landmarks[landmarks.session_pitch.eq('3034_4')].sort_values('time')
    times=float(g.fp_10_time.iloc[0])+ids*dt
    np.testing.assert_allclose(times[-1],float(g.BR_time.iloc[0]),atol=1e-9)
    names=sorted(set(sum(([a,b] for a,b in BONES+ARM),[])))
    observed={name:CubicSpline(regular_time(g),vector(g,name))(times) for name in names}
    angles=pd.read_csv(dataset/'full_sig/joint_angles.csv');a=angles[angles.session_pitch.eq('3034_4')].sort_values('time')
    bodyangles={name:CubicSpline(regular_time(a),a[[name+'_angle_'+axis for axis in 'xyz']].to_numpy())(times) for name in ('pelvis','torso','shoulder','elbow','wrist')}
    channels=np.column_stack([relative_angle(q[:,3:7]),np.degrees(q[:,7]),np.degrees(q[:,8]-q[0,8]),relative_angle(q[:,9:13])])
    labels=['상완 회전 차이','팔꿈치 힌지각 차이','전완 축 회전 차이','손목 상대회전 차이']
    basis=np.array([[.70710678,-.70710678,0],[.24184476,.24184476,-.93969262]])
    bodyall=np.concatenate([v for v in observed.values()]);bodycenter=(bodyall.max(axis=0)+bodyall.min(axis=0))/2
    armcenter=(p.reshape(-1,3).max(axis=0)+p.reshape(-1,3).min(axis=0))/2
    colors=['#ffbd71','#52d7de','#bba7fa'];white='#eaf1f7';gray='#92a4b9'
    def text(d,xy,s,size=20,color=white):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill=color)
    def proj(x,center,scale,offset):return (np.asarray(x)-center)@basis.T*scale+offset
    def line(d,points,center,scale,offset,color,width):d.line([tuple(x) for x in proj(points,center,scale,offset)],fill=color,width=width)
    writer=imageio_ffmpeg.write_frames(str(out/'baseball_joint_detail.mp4'),(1440,912),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for f in range(600):
        i=min(f%300,240);im=Image.new('RGB',(1440,912),'#101a29');d=ImageDraw.Draw(im)
        text(d,(30,18),'야구 투구: 원래 기준 동작과 모델을 겹쳐 비교',32)
        text(d,(30,63),'선수 ID 2 / 3034_4 | 같은 투구 2회 느린 재생 | 발 착지~BR | 실사 영상 아님',20,gray)
        text(d,(30,110),'전신: 관찰 좌표 재생',24)
        text(d,(480,110),'회색 기준 / 색 선 실제 적분',24)
        text(d,(1020,110),'같은 시각의 기준 대비 차이',24)
        for aa,bb in BONES+ARM:line(d,[observed[aa][i],observed[bb][i]],bodycenter,235,[230,390],'#667b92',6)
        line(d,p[i],bodycenter,235,[230,390],'#52d7de',5)
        line(d,rp[i],armcenter,330,[700,355],'#bcc8d5',17)
        for k in range(3):
            line(d,p[i,k:k+2],armcenter,330,[700,355],colors[k],11 if k<2 else 6)
            at=proj(p[i,k],armcenter,330,[700,355]);d.ellipse((*tuple(at-8),*tuple(at+8)),fill=colors[k])
            text(d,tuple(at+[12,5]),['어깨','팔꿈치','손목'][k],18,colors[k])
            # Small segment-local triads, showing twist that a centerline alone hides.
            for axis,c in enumerate(['#f47180','#92dc94','#79a7ff']):
                line(d,[p[i,k],p[i,k]+frames[i,k,:,axis]*.09],armcenter,330,[700,355],c,3)
        corners=np.array([[x,y,z] for x in (0,.16) for y in (-.04,.04) for z in (-.015,.015)])
        hc=p[i,2]+corners@hand[i].T
        rhc=rp[i,2]+corners@rhand[i].T
        for j in range(8):
            for k in range(j+1,8):
                if np.count_nonzero(corners[j]!=corners[k])==1:
                    line(d,rhc[[j,k]],armcenter,330,[700,355],'#bcc8d5',4)
                    line(d,hc[[j,k]],armcenter,330,[700,355],'#bba7fa',2)
        ball=proj(p[i,3],armcenter,330,[700,355]);radius=.0364*330
        d.ellipse((*tuple(ball-radius),*tuple(ball+radius)),fill='#f4f3d5',outline='#baac7c',width=2)
        for j,label in enumerate(labels):
            y=170+j*92;text(d,(1020,y),label,20,colors[min(j,2)])
            text(d,(1020,y+28),f'{angle_delta[i,j]:+.6f} 도',27)
            text(d,(1020,y+65),f'구간 최대 |차이| {abs(angle_delta[:,j]).max():.6f} 도',16,gray)
        text(d,(30,610),f'모델 공 속력 {speed[i]:.2f} km/h',28,'#52d7de')
        text(d,(30,653),'BR 모델 113.65 / 기록 128.43 km/h',22)
        text(d,(30,691),'구속 오차는 아직 해결되지 않음',21,'#ffbd71')
        text(d,(485,610),'같은 시각의 위치 차이 (mm)',22)
        for j,name in enumerate(['어깨','팔꿈치','손목','가정 공 중심']):
            text(d,(485,650+j*31),f'{name}  {position_delta[i,j]:.6f}  / 최대 {position_delta[:,j].max():.6f}',18,gray)
        text(d,(1020,610),'의도적 자세 수정: 없음',23,'#ffbd71')
        text(d,(1020,650),'원래 동작을 재현하는 검증',19,gray)
        text(d,(1020,685),'최적화 후보의 변경량이 아님',19,gray)
        text(d,(30,805),'기준 = 관찰에서 구성한 모델 자세. 겹침과 차이 수치는 동일 기준·동일 시각·동일 축척입니다.',20)
        text(d,(30,843),'각도: 팔꿈치/전완은 부호 있는 차이, 상완/손목은 회전 차이 크기. 위치는 3D 거리이며 오차를 확대하지 않았습니다.',17,gray)
        if f in (120,240):im.save(out/f'preview-{f}.png')
        writer.send(np.asarray(im))
    writer.close()
    report=dict(frames=600,fps=30,duration_s=20,physical_interval_s=float(times[-1]-times[0]),final_model_speed_kmh=float(speed[-1]),channels=labels,channel_min_deg=angle_delta.min(axis=0).tolist(),channel_max_deg=angle_delta.max(axis=0).tolist(),position_difference_max_mm=position_delta.max(axis=0).tolist(),intentional_pose_change=False,body_scope='observed landmarks only',arm_scope='saved forward integrated states; no physics change',anatomical_calibration=False,sources_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [source,dataset/'full_sig/landmarks.csv',dataset/'full_sig/joint_angles.csv']})
    (out/'metadata.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
