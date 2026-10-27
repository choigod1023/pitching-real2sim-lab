import json,subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg
from render_fine_timing import points
from curl_geometry import path
from grip_bench import DIR,RADIUS,PAD_RADIUS,DEPTH

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/contact-curl'
FONT={n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n) for n in [16,18,20,22,26,28]}


def run():
    report=json.loads((ROOT/'curl_pitch_results.json').read_text());b=report['rows'][6];r=report['selected']
    old=points(OUT/'state-0-0-16.npz');new=points(OUT/f"state-1-{int(r['curl_deg'])}-16.npz");allp=np.concatenate([old[0],new[0]]);n=len(new[0]);dq=np.zeros(11);angles=[];wrists=[]
    for qa,qb,pa,pb in zip(old[1],new[1],old[0],new[0]):
        mujoco.mj_differentiatePos(new[2],dq,1.,qa,qb);angles.append(np.rad2deg(max(np.linalg.norm(dq[3:6]),abs(dq[6]),abs(dq[7]),np.linalg.norm(dq[8:11]))));wrists.append(np.linalg.norm(pa[2]-pb[2])*1000)
    dt=float(np.load(OUT/'state-0-0-16.npz')['dt']);video=OUT/'contact_curl_overlay.mp4'
    w=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=24,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart']);w.send(None)
    for frame in range(240):
        i=min(round(min(frame/192,1)*(n-1)),n-1);t=(i+1)*dt;im=Image.new('RGB',(1280,720),'#101925');d=ImageDraw.Draw(im)
        def text(x,y,s,size=20,color='#eff5ff'):d.text((x,y),s,font=FONT[size],fill=color)
        text(22,12,'3034_4 | 접촉 재배치 + 굽힘 경로 비교 · 제약 통과 0개',28)
        text(22,57,f"회색 기존 끝 구속 {b['speed_kmh']:.3f} → 청록 {r['speed_kmh']:.3f}km/h / 추가 일로 탈락",22)
        for panel,axes in enumerate([(0,2),(1,2)]):
            left=14+454*panel;d.rounded_rectangle((left,100,left+440,450),12,fill='#1c2b3f')
            xy=allp[:,:,list(axes)];lo=xy.min(axis=(0,1));hi=xy.max(axis=(0,1));mid=(lo+hi)/2;scale=min(370/(hi[0]-lo[0]),270/(hi[1]-lo[1]))
            def pt(p):
                a=(p[list(axes)]-mid)*scale;return(left+220+a[0],285-a[1])
            for st,col,wide in [(old,'#97a7bb',7),(new,'#41dce6',3)]:
                ps=[pt(p) for p in st[0][i]];d.line(ps[:4],fill=col,width=wide);a=ps[4];rr=RADIUS*scale;d.ellipse((a[0]-rr,a[1]-rr,a[0]+rr,a[1]+rr),outline=col,width=3)
            text(left+12,112,['홈 방향–높이','좌우–높이'][panel],18)
        d.rounded_rectangle((930,100,1266,450),12,fill='#1c2b3f');text(943,113,'손좌표의 접촉점 경로',20)
        def local(p):return(1098+1700*(p[0]+.35*p[1]),279-1700*(p[2]+.2*p[1]))
        axis=np.array(r['curl_axis']);dirs=np.array(r['initial_directions']);radius=RADIUS+PAD_RADIUS-DEPTH
        for j in range(3):
            curve=[local(radius*path(a,dirs,axis,r['curl_deg'])[0][j]) for a in np.linspace(0,.020,40)]
            d.line(curve,fill='#41dce6',width=3)
            for p,col in [(radius*DIR[j],'#97a7bb'),(radius*path(min(t,.020),dirs,axis,r['curl_deg'])[0][j],'#41dce6')]:
                a=local(p);d.ellipse((a[0]-8,a[1]-8,a[0]+8,a[1]+8),fill=col)
        a=local(np.zeros(3));d.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3),fill='#fff')
        text(944,375,'흰 점: 공칭 중심 / +10°',18)
        text(944,405,'설정 경로 · 열림은 생략',18)
        text(22,472,f'현재 팔 부위 회전 차이 {angles[i]:.2f}° / 손목 위치 차이 {wrists[i]:.2f}mm',22)
        text(22,511,f"마지막 접촉 {b['last_contact_ms']:.3f} → {r['last_contact_ms']:.3f}ms / 더 일찍 분리",20)
        text(22,548,f"최대 합력 {b['peak_net_force_n']:.3f} → {r['peak_net_force_n']:.3f}N / 관통 35.097 → 14.625mm",20)
        text(22,585,f"팔 양의 일 {sum(b['work_positive_j']):.3f} → {sum(r['work_positive_j']):.3f}J / 패드 추가 양의 일 {sum(r['finger_work_positive_j']):.6f}J",20)
        text(22,628,'같은 힘 파형이어도 일이 증가 · 관통 2mm 기준 초과 · 선수 향상 후보 아님',22,'#ffb77a')
        text(22,674,'실제 적분 상태 / 해부학적 손가락 복원 아님 / 관통 최소 조건을 진단용으로 표시',18)
        w.send(np.asarray(im))
        if frame==192:im.save(OUT/'preview.png')
    w.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
    (OUT/'video_metrics.json').write_text(json.dumps(dict(max_rotation_deg=max(angles),max_wrist_mm=max(wrists)),indent=2))


if __name__=='__main__':run()
