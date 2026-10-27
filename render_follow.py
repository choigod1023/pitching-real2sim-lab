import json,subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg
from render_fine_timing import points

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/grip-follow'
FONT={n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n) for n in [18,20,22,26,28]}


def make_writer(path):
    w=imageio_ffmpeg.write_frames(str(path),(1280,720),fps=24,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart']);w.send(None);return w


def verify(path):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True)


def run():
    rows=json.loads((ROOT/'acceleration_follow_results.json').read_text())['rows']
    r=[next(x for x in rows if x['axis']=='z' and x['acceleration_m_s2']==50 and x['follow']==f) for f in [False,True]]
    data=[np.load(OUT/f'acc-z-50-{f}.npz')['states'] for f in [0,1]]
    path=OUT/'acceleration_follow.mp4';w=make_writer(path)
    for frame in range(192):
        im=Image.new('RGB',(1280,720),'#101925');d=ImageDraw.Draw(im)
        def text(x,y,s,size=20,color='#eef6ff'):d.text((x,y),s,font=FONT[size],fill=color)
        text(24,16,'손 z축 가속도 50m/s² | 접촉 경로 변경 전후',28)
        text(24,60,'동일 압박·마찰·강성 / 손 기준 좌표로 표시 / 30ms를 느리게 재생',20)
        for j in [0,1]:
            left=16+632*j;d.rounded_rectangle((left,110,left+616,470),12,fill='#1b2a3e')
            text(left+15,123,['고정 접촉점','공을 따라가는 접촉점'][j],22)
            s=data[j][min(round(min(frame/160,1)*(len(data[j])-1)),len(data[j])-1)]
            center=np.array([0,0,.5+.5*50*s[0]**2])
            def pt(p):
                p=p-center;return(left+308+2500*(p[0]+.35*p[1]),280-2500*(p[2]+.2*p[1]))
            b=pt(s[1:4]);rr=.0364*2500;d.ellipse((b[0]-rr,b[1]-rr,b[0]+rr,b[1]+rr),fill='#e7edf8',outline='#fff',width=2)
            for p,force in zip(s[10:19].reshape(3,3),s[19:22]):
                q=pt(p);rad=.008*2500;d.ellipse((q[0]-rad,q[1]-rad,q[0]+rad,q[1]+rad),fill='#3ad3df' if force>1e-7 else '#65718a',outline='#fff')
            text(left+15,423,f't={s[0]*1000:.2f}ms / 최대 패드 힘 {max(s[19:22]):.3f}N',20)
            label='28.50ms에 접촉 상실' if j==0 else '30ms까지 접촉 남음 · 손에서 뒤처짐'
            text(left+15,492,label,22,'#ffb778')
            text(left+15,530,f"손–공 상대 이동 최대 {r[j]['max_ball_hand_offset_mm']:.3f}mm",20)
            text(left+15,566,f"전달 양의 일 합 {sum(r[j]['work_positive_j']):.6f}J",20)
        text(24,620,'접촉 지속 ≠ 공을 손과 함께 가속하는 그립 성공 · 양의 일 증가로 동등 예산 개선 아님',20,'#ffb778')
        text(24,665,'공학적 패드 대리 모델 / 실제 손가락 측정 또는 선수 능력 검증이 아닙니다.',20)
        w.send(np.asarray(im))
        if frame==160:im.save(OUT/'acc-preview.png')
    w.close();verify(path)
    before=points(OUT/'pitch-0-16.npz');after=points(OUT/'pitch-1-16.npz');n=len(before[0]);allp=np.concatenate([before[0],after[0]])
    rows=json.loads((ROOT/'follow_pitch_results.json').read_text())['rows'];old,new=rows[1],rows[3]
    dq=np.zeros(11);angles=[];wrists=[]
    for qa,qb,pa,pb in zip(before[1],after[1],before[0],after[0]):
        mujoco.mj_differentiatePos(after[2],dq,1.,qa,qb);angles.append(np.rad2deg(max(np.linalg.norm(dq[3:6]),abs(dq[6]),abs(dq[7]),np.linalg.norm(dq[8:11]))));wrists.append(np.linalg.norm(pa[2]-pb[2])*1000)
    path=OUT/'follow_pitch_overlay.mp4';w=make_writer(path)
    for frame in range(240):
        i=min(round(min(frame/192,1)*(n-1)),n-1);im=Image.new('RGB',(1280,720),'#101925');d=ImageDraw.Draw(im)
        def text(x,y,s,size=20,color='#eef6ff'):d.text((x,y),s,font=FONT[size],fill=color)
        text(24,14,'3034_4 | 고정 접촉점 vs 공 추종 경로 · 같은 팔 입력',28)
        text(24,58,f"회색 끝 구속 {old['speed_kmh']:.3f}km/h → 청록 {new['speed_kmh']:.3f}km/h / 증가 아님",22)
        for panel,axes in enumerate([(0,2),(1,2)]):
            left=16+632*panel;d.rounded_rectangle((left,102,left+616,440),12,fill='#1b2a3e')
            xy=allp[:,:,list(axes)];lo=xy.min(axis=(0,1));hi=xy.max(axis=(0,1));mid=(lo+hi)/2;scale=min(520/(hi[0]-lo[0]),250/(hi[1]-lo[1]))
            def pt(p):
                a=(p[list(axes)]-mid)*scale;return(left+308+a[0],285-a[1])
            for state,col,width in [(before,'#98a6b7',8),(after,'#38dce3',3)]:
                ps=[pt(p) for p in state[0][i]];d.line(ps[:4],fill=col,width=width);b=ps[4];rr=.0364*scale;d.ellipse((b[0]-rr,b[1]-rr,b[0]+rr,b[1]+rr),outline=col,width=3)
            text(left+15,115,['홈 방향–높이','좌우–높이'][panel],18)
        text(24,460,f'현재 부위 회전 차이 {angles[i]:.3f}° / 손목 위치 차이 {wrists[i]:.3f}mm',22)
        text(24,500,'마지막 접촉 13.587 → 13.977ms (+0.391ms) / 접촉점 이동 최대 8.660mm',20)
        text(24,538,'손–공 최대 합력 136.905 → 141.995N / 관통 최대 35.097 → 36.376mm',20)
        text(24,576,'추종 구동: 양의 일 0.120989J · 흡수한 일 0.187103J (별도 계상)',20)
        text(24,614,'힘·일 증가와 과도한 관통으로 탈락 / 구속 변화 -0.246km/h',22,'#ffb778')
        text(24,665,'실측 영상 아님 · 이상적인 즉시 추종 가정 · 실제 선수에게 권할 향상 동작 아님',20)
        w.send(np.asarray(im))
        if frame==192:im.save(OUT/'pitch-preview.png')
    w.close();verify(path)
    (OUT/'video_metrics.json').write_text(json.dumps(dict(max_rotation_deg=max(angles),max_wrist_mm=max(wrists)),indent=2))


if __name__=='__main__':run()
