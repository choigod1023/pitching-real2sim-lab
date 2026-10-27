"""Real two-view footage with observed/estimated hand and free-flight overlays."""
import json
import sys
from pathlib import Path
import numpy as np
import ezc3d
from PIL import Image, ImageDraw, ImageFont
from multiview_release_validation import ROOT, LAB, OUT, camera_models, project
sys.path.insert(0,str(LAB/'comparison-video/artifacts/video-vendor'))
import imageio_ffmpeg

FONT='C:/Windows/Fonts/malgun.ttf'
COLORS={'A':'#ff975b','B':'#53efac','C':'#d0a0ff'}


def text(draw,xy,value,size=22,color='white'):
    draw.text(xy,value,font=ImageFont.truetype(FONT,size),fill=color)


def run():
    report=json.loads((OUT/'report.json').read_text());models=report['nominal']['models'];cams=camera_models()
    obs=json.loads((LAB/'hand-ball-link/artifacts/hand-ball/observation.json').read_text())
    selected=next(x for x in obs['conditions'] if x['threshold']==80 and x['c3d_frame_offset']==1)
    balls={x['video_frame']:np.array(x['ball_rig_m']) for x in selected['positions']}
    align=json.loads(Path(obs['source_sha256']['alignment']['path']).read_text())
    c=ezc3d.c3d(obs['source_sha256']['body']['path']);labels=c['parameters']['ROTATION']['LABELS']['value']
    names=['l_uarm_4X4','l_larm_4X4','l_hand_4X4']
    q=np.array(align['rotation']);shift=np.array(align['translation'])
    joints={f:np.array([align['scale']*(q@c['data']['rotations'][:3,3,labels.index(n),f+1])+shift for n in names]) for f in range(900,986)}
    wanted=set(range(900,973,3))|set(range(973,986));images={}
    for cam in [20,21]:
        reader=imageio_ffmpeg.read_frames(str(LAB/f'speed-journey/artifacts/release-observation/camera{cam}.mp4'),pix_fmt='rgb24')
        meta=next(reader);w,h=meta['size'];images[cam]={}
        for f,data in enumerate(reader):
            if f in wanted:images[cam][f]=Image.frombytes('RGB',(w,h),data)
            if f>=985:break
        reader.close()
    writer=imageio_ffmpeg.write_frames(str(OUT/'multiview_hand_ball_overlay.mp4'),(1280,900),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
    writer.send(None)
    sequence=[min(972,900+3*int(i/180*25)) for i in range(180)]+[973+min(6,int(i/210*7)) for i in range(210)]+[979]*90
    for index,f in enumerate(sequence):
        canvas=Image.new('RGB',(1280,900),'#101923');d=ImageDraw.Draw(canvas)
        text(d,(25,12),'같은 야구 투구: 손–공 연결 수정 전후',30)
        text(d,(25,55),'주황 고정 그립 A · 초록 상대운동 B · 보라 공 추적 C · 흰색 관측 공 후보',21)
        for col,cam in enumerate([20,21]):
            raw=images[cam][f].copy();rd=ImageDraw.Draw(raw)
            jp=project(joints[f],cams[cam]);rd.line([tuple(x) for x in jp],fill='#55d7ff',width=3)
            for x,y in jp:rd.ellipse((x-4,y-4,x+4,y+4),fill='#55d7ff')
            if f in balls:
                x,y=project(balls[f],cams[cam])[0];rd.ellipse((x-10,y-10,x+10,y+10),outline='white',width=2)
            if f>=977:
                for name in ['A','B','C']:
                    model=models[name];p=np.array(model['trajectory_m']);frames=model['trajectory_frames']
                    pp=project(p[:frames.index(f)+1],cams[cam]);rd.line([tuple(x) for x in pp],fill=COLORS[name],width=3)
                    x,y=pp[-1];rd.ellipse((x-6,y-6,x+6,y+6),outline=COLORS[name],width=3)
            canvas.paste(raw.resize((640,360)),(col*640,95))
            center=project(balls[f] if f in balls else joints[f][-1],cams[cam])[0]
            x,y=center;crop=raw.crop((int(x)-160,int(y)-65,int(x)+160,int(y)+65)).resize((640,260))
            canvas.paste(crop,(col*640,455))
            text(d,(col*640+15,105),f'camera {cam} | 디코드 {f} / 영상 표기 {f+1}',20)
        text(d,(25,724),'평가 평균 구속: A 77.32 → B 101.55 km/h | 영상 공 후보 102.87 km/h',21)
        phase='손·공 보정 구간 — 접촉 여부는 잔상 때문에 확정 불가' if f<=976 else '평가 구간 — 모델 분리는 976 가정; 978에서 두 시야의 분리 윤곽 확인'
        text(d,(25,758),phase,21)
        text(d,(25,795),'영상 구속 차이 A 25.55 → B 1.32 km/h | 공 추적 C 0.80 km/h',25)
        text(d,(25,834),'단일 개발 투구·느린 재생 | 실제 이탈/근육 검증 아님 | 3034의 오차 감소 아님',20,'#ffca82')
        writer.send(np.asarray(canvas).tobytes())
        if index==360:canvas.save(OUT/'overlay_preview.png')
    writer.close()
    print(OUT/'multiview_hand_ball_overlay.mp4')


if __name__=='__main__':run()
