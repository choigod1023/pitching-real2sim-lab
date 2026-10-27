import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg
from render_fine_timing import points
import mujoco

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/grip-bench'
F={s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s) for s in [16,18,20,22,26,28]}


def writer(path):
    w=imageio_ffmpeg.write_frames(str(path),(1280,720),fps=24,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart']);w.send(None);return w


def check(path):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True)


def render():
    rows=json.loads((ROOT/'grip_bench_results.json').read_text())['rows'];kinds=['static','motion','release']
    data=[np.load(OUT/f'{k}.npz')['states'] for k in kinds]
    path=OUT/'grip_bench_checks.mp4';w=writer(path)
    for frame in range(192):
        im=Image.new('RGB',(1280,720),'#111925');d=ImageDraw.Draw(im)
        def txt(x,y,s,size=20,color='#eef4ff'):d.text((x,y),s,font=F[size],fill=color)
        txt(24,16,'초기 그립 → 손 운동 → 패드 열림 | 공학적 검사 3개 통과',28)
        txt(24,60,'공통: 패드 3개 · 초기 압박 1mm · 저장 탄성 에너지 0.005J · 실제 손가락 복원 아님',20)
        for j,kind in enumerate(kinds):
            left=16+j*420;d.rounded_rectangle((left,110,left+408,515),12,fill='#1d2a3e')
            txt(left+14,124,['정적 유지','손 이동 + 회전','패드 열림으로 해제'][j],22)
            s=data[j][min(round(min(frame/160,1)*(len(data[j])-1)),len(data[j])-1)]
            def point(p):return (left+204+2300*(p[0]+.35*p[1]),300-2300*(p[2]-.5+.2*p[1]))
            p=point(s[1:4]);rad=2300*.0364;d.ellipse((p[0]-rad,p[1]-rad,p[0]+rad,p[1]+rad),fill='#e9eff7',outline='#fff',width=2)
            for pad,force in zip(s[10:19].reshape(3,3),s[19:22]):
                a=point(pad);rr=2300*.008;d.ellipse((a[0]-rr,a[1]-rr,a[0]+rr,a[1]+rr),fill='#37c9d4' if force>1e-7 else '#647084',outline='#efffff')
            txt(left+14,165,f't = {s[0]*1000:.1f}ms / 최대 패드 힘 {max(s[19:22]):.3f}N',18)
            r=rows[j]['fine'];txt(left+10,534,f"양의 일 {sum(r['work_positive_per_pad_j']):.6f}J",18)
            txt(left+10,564,f"흡수한 일 {sum(r['work_negative_per_pad_j']):.6f}J",18)
            txt(left+10,594,f"유지 중 상대 이동 {r['max_relative_offset_mm']:.3f}mm",18)
        txt(24,640,'정지 중 기계적 일 0J는 사람의 유지 노력·대사 비용 0을 뜻하지 않습니다.',20)
        txt(24,676,'회색 공 / 청록 접촉 패드 / 짙은 회색 분리 패드 · 느린 재생 · 실제 투구 검증과 구분',18)
        w.send(np.asarray(im))
        if frame==160:im.save(OUT/'bench-preview.png')
    w.close();check(path)
    if not (ROOT/'grip_pitch_results.json').exists():return
    new=points(OUT/'pitch16.npz');old=points(ROOT.parent/'fine-compliant-timing/artifacts/fine-timing/baseline4.npz')
    r=json.loads((ROOT/'grip_pitch_results.json').read_text())['rows'][-1]
    n=len(new[0]);allp=np.concatenate([old[0],new[0]]);path=OUT/'grip_pitch_overlay.mp4';w=writer(path)
    dq=np.zeros(11);wrists=[];angles=[]
    for a,b,pa,pb in zip(old[1],new[1],old[0],new[0]):
        mujoco.mj_differentiatePos(new[2],dq,1.,a,b)
        angles.append(np.rad2deg(max(np.linalg.norm(dq[3:6]),abs(dq[6]),abs(dq[7]),np.linalg.norm(dq[8:11]))));wrists.append(np.linalg.norm(pa[2]-pb[2])*1000)
    for frame in range(240):
        i=min(round(min(frame/192,1)*(n-1)),n-1);im=Image.new('RGB',(1280,720),'#111925');d=ImageDraw.Draw(im)
        def txt(x,y,s,size=20,color='#eef4ff'):d.text((x,y),s,font=F[size],fill=color)
        txt(24,14,'3034_4 투구 연결 검사 | 조기 분리·큰 관통으로 실패',28)
        txt(24,57,'회색: 기존 중심 스프링 127.887km/h / 청록: 초기 그립 + 열림 84.224km/h (끝 구속)',20)
        for panel,axes in enumerate([(0,2),(1,2)]):
            left=16+panel*632;d.rounded_rectangle((left,100,left+616,466),12,fill='#1d2a3e')
            xy=allp[:,:,list(axes)];lo=xy.min(axis=(0,1));hi=xy.max(axis=(0,1));scale=min(520/(hi[0]-lo[0]),280/(hi[1]-lo[1]));mid=(lo+hi)/2
            def pt(p):
                a=(p[list(axes)]-mid)*scale;return (left+308+a[0],287-a[1])
            for state,color,width in [(old,'#95a1b1',7),(new,'#44dee6',3)]:
                ps=[pt(p) for p in state[0][i]];d.line(ps[:4],fill=color,width=width);p=ps[4];rr=.0364*scale;d.ellipse((p[0]-rr,p[1]-rr,p[0]+rr,p[1]+rr),outline=color,width=3)
            txt(left+12,110,['홈 방향–높이','좌우–높이'][panel],18)
        txt(24,486,f'현재 기준 대비 최대 부위 회전 차이 {angles[i]:.2f}° / 손목 위치 차이 {wrists[i]:.2f}mm',22)
        txt(24,524,f"마지막 접촉 {r['last_contact_ms']:.3f}ms / 공칭 종료 30.027ms / 관통 최대 {r['max_penetration_mm']:.2f}mm",20,'#ffb874')
        txt(24,562,f"손–공 최대 합력 {r['peak_net_force_n']:.2f}N / 팔 양의 일 {sum(r['work_positive_j']):.3f}J",20)
        txt(24,600,'추가 초기 탄성 에너지 0.005J / 열림 일 0J: 열기 전에 이미 분리됨',20)
        txt(24,642,'정적·단순 운동 통과 ≠ 투구에서 그립 유지 성공 · 구속 향상 후보로 채택하지 않음',20,'#ffb874')
        txt(24,680,'실측 영상 아님 / 같은 원래 팔 힘 입력 / 총 일·선수 능력이 같다는 뜻은 아님',18)
        w.send(np.asarray(im))
        if frame==192:im.save(OUT/'pitch-preview.png')
    w.close();check(path)
    (OUT/'video_metrics.json').write_text(json.dumps(dict(max_rotation_difference_deg=max(angles),max_wrist_difference_mm=max(wrists)),indent=2))


if __name__=='__main__':render()
