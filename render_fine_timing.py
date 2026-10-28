"""Overlay actual integrated arm/hand/free-ball states in two lab projections."""
import json,subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from coupled_payload import model_for
from render_speed_journey import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent

def points(path):
    f=np.load(path);m=model_for(f['lengths'],float(f['dt']),ball_mass=0.);d=mujoco.MjData(m);out=[]
    for q,b in zip(f['qpos'],f['ball']):
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        out.append(np.array([d.xpos[m.body('upper').id],d.xpos[m.body('forearm').id],d.xpos[m.body('hand').id],d.site_xpos[m.site('ball_center').id],b]))
    return np.array(out),f['qpos'],m

def run():
    directory=ROOT/'artifacts/fine-timing';report=json.loads((ROOT/'fine_timing_results.json').read_text());unc=json.loads((ROOT/'fine_uncertainty_results.json').read_text())
    states=[points(directory/f'{s}.npz') for s in ['baseline4','original','selected']]
    p=[s[0] for s in states];n=len(p[0]);allp=np.concatenate(p);colors=['#8794a7','#ffb454','#48deec'];names=['기준','기존 후보','세밀 조정']
    fonts={s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s) for s in [16,18,20,22,24,28]}
    video=directory/'fine_timing_overlay.mp4';writer=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=24,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart']);writer.send(None)
    indices=np.r_[np.rint(np.linspace(0,n-1,192)).astype(int),np.full(48,n-1)]
    base=report['baseline_sub4'];candidate=report['selected_sub4'];original=unc['original_sub4'];dq=np.zeros(11)
    for fi,i in enumerate(indices):
        im=Image.new('RGB',(1280,720),'#101724');draw=ImageDraw.Draw(im)
        def text(x,y,s,size=20,color='#eaf2ff'):draw.text((x,y),s,font=fonts[size],fill=color)
        text(24,12,'야구 투구 팔·손·공 | 같은 힘·일 제한에서 타이밍 세부 조정',28)
        text(24,54,'익명 선수 2 / 3034_4 · 말기 30ms 느린 재생 · 기준과 후보를 같은 좌표에 겹침',18)
        for j,(name,color) in enumerate(zip(names,colors)):
            speed=[base,original,candidate][j]['speed_kmh'];text(24+j*420,88,f'{name}: 끝 구속 {speed:.3f} km/h',22,color)
        for view,axes in enumerate([(0,2),(1,2)]):
            low=allp[:,:,axes].reshape(-1,2).min(axis=0)-.08;high=allp[:,:,axes].reshape(-1,2).max(axis=0)+.08
            scale=min(570/(high[0]-low[0]),340/(high[1]-low[1]));center=(low+high)/2
            def project(point):
                xy=(point[list(axes)]-center)*scale
                return (int(320+view*640+xy[0]),int(320-xy[1]))
            draw.rounded_rectangle((16+640*view,128,624+640*view,507),radius=12,fill='#182335')
            text(32+640*view,136,'홈 방향–높이(X–Z)' if view==0 else '좌우–높이(Y–Z)',18)
            for j in range(3):
                ps=[project(v) for v in p[j][i]];width=[10,6,3][j]
                draw.line(ps[:4],fill=colors[j],width=width,joint='curve')
                for x,y in ps[:3]:draw.ellipse((x-5,y-5,x+5,y+5),outline=colors[j],width=2)
                a,b=np.array(ps[3]),np.array(ps[4])
                for t in np.arange(0,1,.2):draw.line([tuple(a+(b-a)*t),tuple(a+(b-a)*min(t+.1,1))],fill=colors[j],width=2)
                x,y=ps[4];radius=max(3,int(.0364*scale));draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=colors[j],width=2)
            text(32+640*view,478,'실선: 팔·손 / 점선: 탄성 연결 / 원: 자유 공',16,'#b6c4d5')
        mujoco.mj_differentiatePos(states[0][2],dq,1.,states[0][1][i],states[2][1][i])
        angle=np.rad2deg(max(np.linalg.norm(dq[3:6]),abs(dq[6]),abs(dq[7]),np.linalg.norm(dq[8:11])))
        mm=np.linalg.norm(p[2][i,2]-p[0][i,2])*1000
        text(24,520,f'세부 입력: 어깨 {candidate["shifts_ms"][0]:+.3f}ms / 팔꿈치 {candidate["shifts_ms"][1]:+.3f}ms / 손목 {candidate["shifts_ms"][2]:+.3f}ms',20)
        text(24,552,f'현재 기준 대비 회전 차이 {angle:.3f}° / 손목 위치 차이 {mm:.3f}mm',20,'#48deec')
        text(24,584,f'구속 +{candidate["gain_kmh"]:.3f}km/h · 최대 부위 일 +{candidate["violations"]["work_positive_j"]:.3f}J · 손–공 힘 +{candidate["violations"]["coupling_peak_force_n"]:.3f}N',20)
        text(24,617,'제약 통과 0개: 후보 탈락 · 실제 선수에게 권고할 향상 동작 아님',22,'#ffb454')
        text(24,653,'실측 영상 아님 / 손가락 접촉 미구현 / 실측 구속 예측 오차범위 미확정',18,'#bac6d5')
        text(24,683,'기준 실측 차이 약0.539km/h는 한 사례 · 연결 가정 변경 시 기준 구속119.97~135.29km/h',16,'#bac6d5')
        writer.send(np.asarray(im))
        if fi in (96,191):im.save(directory/f'preview-{fi}.png')
    writer.close()
    check=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    if check.returncode or check.stderr:raise RuntimeError(check.stderr)
    manifest=dict(video=str(video),frames=len(indices),fps=24,duration_s=len(indices)/24,decode_exit_code=check.returncode,accepted=False,baseline_speed_kmh=base['speed_kmh'],candidate_speed_kmh=candidate['speed_kmh'],state_indices=indices.tolist())
    (directory/'video_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(str(video))

if __name__=='__main__':run()
