"""Render saved physical rollout states, without rerunning or exaggerating motion."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from coupled_payload import model_for
from trace_error_audit import kinematics

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'comparison-video/artifacts/video-vendor'))
import imageio_ffmpeg

SOURCE = ROOT.parent/'trace-error-audit/artifacts/trace-error'
OUT = ROOT/'artifacts/trace-video'
FONT = 'C:/Windows/Fonts/malgun.ttf'
BG = '#101a29'
WHITE = '#edf4fa'
GRAY = '#93a5b8'
COLORS = ['#ffad66', '#55d8e3']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = ['3034_4-11520-pad0-zoh-sub2', '3034_4-46080-pad1-foh-sub1']
    sequences = []
    for name in names:
        with np.load(SOURCE/(name+'.npz')) as z:
            ids = np.rint(np.linspace(0, len(z['qpos'])-1, 241)).astype(int)
            m = model_for(z['lengths'], float(z['dt']))
            p, v, r = kinematics(m, z['qpos'][ids], z['qvel'][ids])
            ref, _, _ = kinematics(m, z['qreference'][ids], z['vreference'][ids])
            # marker_roll is the unwrapped scalar coordinate after elbow (qpos index 8).
            error = np.abs(np.degrees(z['qpos'][ids,8]-z['qreference'][ids,8]))
            all_error = np.abs(np.degrees(z['qpos'][:,8]-z['qreference'][:,8]))
            sequences.append(dict(p=p, r=r, ref=ref, speed=np.linalg.norm(v[:,3],axis=1)*3.6,
                error=error, max_error=float(all_error.max()), duration=(len(z['qpos'])-1)*float(z['dt'])))
    assert abs(sequences[0]['speed'][-1]-113.8207409429)<1e-6
    assert abs(sequences[1]['speed'][-1]-113.6466512019)<1e-6
    assert abs(sequences[0]['duration']-sequences[1]['duration'])<1e-10
    # Fixed orthographic camera and identical physical scale in both panels.
    basis = np.array([[.70710678,-.70710678,0],[.24184476,.24184476,-.93969262]])
    allp = np.concatenate([s['p'].reshape(-1,3) for s in sequences])
    center = (allp.min(axis=0)+allp.max(axis=0))/2
    def project(p, cx):
        return (p-center)@basis.T*290 + np.array([cx,345])
    def text(d,xy,value,size=22,color=WHITE):
        d.text(xy,value,font=ImageFont.truetype(FONT,size),fill=color)
    writer = imageio_ffmpeg.write_frames(str(OUT/'pitch_trace_before_after.mp4'),(1280,800),fps=30,
        codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
    writer.send(None)
    for frame in range(600):
        # Eight-second slow replay, four-second release hold, then eight-second result page.
        i=min(frame,240)
        im=Image.new('RGB',(1280,800),BG);d=ImageDraw.Draw(im)
        text(d,(35,20),'같은 야구 투구 · 수정 전 / 후 시뮬레이션',32)
        text(d,(35,66),'선수 ID 2 · 투구 3034_4 | 실사 영상이 아닌 저장된 물리 적분 상태',20,GRAY)
        if frame<360:
            for side,s in enumerate(sequences):
                x=side*640; color=COLORS[side]
                text(d,(x+35,110),'수정 전: 구간별 일정한 힘' if side==0 else '수정 후: 힘 보간 + 끝점 미분 수정',25,color)
                p=project(s['p'][i],x+320); ref=project(s['ref'][i],x+320)
                d.line([tuple(a) for a in ref],fill='#536177',width=12)
                d.line([tuple(a) for a in p],fill=color,width=5)
                for k,(a,b) in enumerate(p):
                    d.ellipse((a-7,b-7,a+7,b+7),fill=WHITE if k==3 else color)
                    if k<3:text(d,(a+12,b+8),['어깨','팔꿈치','손목'][k],17,GRAY)
                # Hand local frame shows orientation, at its actual scale.
                for axis,c in enumerate(['#f97985','#a0e786','#8bafff']):
                    endpoint=s['p'][i,2]+s['r'][i,:,axis]*.09
                    e=project(endpoint,x+320)
                    d.line([tuple(p[2]),tuple(e)],fill=c,width=3)
                text(d,(x+35,475),f"현재 가정 공 속력  {s['speed'][i]:.2f} km/h",25,color)
                text(d,(x+35,515),f"회전 재현 오차  {s['error'][i]:.6f}°",22)
                text(d,(x+35,550),f"구간 최대 오차  {s['max_error']:.6f}°",20,GRAY)
                # Same axes, no visual magnification: 0..4 degrees in both plots.
                x0=x+40;y0=693;w=550;h=95
                d.line([(x0,y0-h),(x0,y0),(x0+w,y0)],fill=GRAY,width=1)
                points=[(x0+j/240*w,y0-s['error'][j]/4*h) for j in range(i+1)]
                if len(points)>1:d.line(points,fill=color,width=3)
                text(d,(x0,585),'회전 오차: 두 그래프 모두 0~4°',17,GRAY)
                text(d,(x0,y0+5),'발 착지',17,GRAY);text(d,(x0+w-85,y0+5),'BR 시각',17,GRAY)
            text(d,(35,742),'회색: 관찰 기반 모델 기준 자세  |  색 선: 실제 적분 결과  |  흰 점: 15cm 그립을 가정한 공 중심',19)
            text(d,(35,772),'느린 재생 · 동작/오차 크기 과장 없음 · 회전 오차는 해부학적 관절각 오차가 아닙니다.',16,GRAY)
        else:
            text(d,(50,130),'줄어든 것은 자세의 수치 재현 오차입니다.',32,COLORS[1])
            text(d,(55,195),'회전 재현 오차 최대     3.564607°  →  0.000306°',28)
            text(d,(55,260),'이 투구의 실제 기록 구속                  128.426 km/h',28)
            text(d,(55,310),'모델 이탈 속력                    113.821  →  113.647 km/h',28)
            text(d,(55,365),'실제 구속과의 차이                14.605  →  14.779 km/h',28,COLORS[0])
            text(d,(55,445),'같은 선수: 빠른 투구 - 느린 투구',26)
            text(d,(55,495),'실제 +2.575 km/h  |  모델 +0.155 → +0.213 km/h',28)
            text(d,(55,580),'실제 구속 예측 오차는 아직 해결되지 않았습니다.',29,COLORS[0])
            text(d,(55,635),'다음: 같은 투구의 손·공 좌표, 시간 정렬, 이탈 순간의 상대운동 검증',23)
            text(d,(55,705),'자세 변경이나 구속 향상 훈련의 효과를 입증한 영상이 아닙니다.',21,GRAY)
        if frame in (190,359,480):im.save(OUT/f'preview-{frame}.png')
        writer.send(np.asarray(im))
    writer.close()
    metadata=dict(frames=600,fps=30,duration_s=20,sources=names,pitch='3034_4',
        physical_duration_s=sequences[0]['duration'],end_speeds_kmh=[float(s['speed'][-1]) for s in sequences],
        max_rotation_errors_deg=[s['max_error'] for s in sequences],motion_scale='same physical scale; no error magnification')
    (OUT/'video_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps(metadata))

if __name__=='__main__':main()
