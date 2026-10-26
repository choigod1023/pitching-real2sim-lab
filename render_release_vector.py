"""Export saved-state baseball diagnostic; no athlete improvement claim."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from coupled_payload import model_for
from trace_error_audit import kinematics
from render_speed_journey import imageio_ffmpeg


def run():
    out=Path('artifacts/release-vector');out.mkdir(parents=True,exist_ok=True)
    src=Path('../trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz')
    z=np.load(src);ids=np.rint(np.linspace(0,len(z['qpos'])-1,181)).astype(int)
    model=model_for(z['lengths'],float(z['dt']))
    p,v,_=kinematics(model,z['qpos'][ids],z['qvel'][ids])
    rp,rv,_=kinematics(model,z['qreference'][ids],z['vreference'][ids])
    delta=np.linalg.norm(p-rp,axis=2)*1000
    q=z['qpos'][ids];ref=z['qreference'][ids]
    angles=[]
    for a,b in [(q[:,[4,5,6,3]],ref[:,[4,5,6,3]]),(q[:,[10,11,12,9]],ref[:,[10,11,12,9]])]:
        angles.append(np.degrees((Rotation.from_quat(b).inv()*Rotation.from_quat(a)).magnitude()))
    angles.extend([np.degrees(abs(q[:,7]-ref[:,7])),np.degrees(abs(q[:,8]-ref[:,8]))])
    maxang=np.max(angles,axis=0)
    report=json.loads((out/'report.json').read_text())
    row=next(r for r in report['rows'] if r['pitch']=='3034_4' and r['cutoff_hz']==20 and r['br_shift_frames']==0)
    metrics=row['metrics'];origin=p[-1,3]
    font=lambda size:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size)
    colors=['#c0c5cd','#35bfff'];bg='#101924';white='#f2f5f8'
    def text(d,x,y,s,size=22,color=white):d.text((x,y),s,font=font(size),fill=color)
    center=(p.reshape(-1,3).min(0)+p.reshape(-1,3).max(0))/2
    def project(a,front=False):
        if front:return np.c_[a[:,1]-center[1],-(a[:,2]-center[2])]*260+[950,320]
        return np.c_[a[:,0]-center[0],-(a[:,2]-center[2])]*260+[320,320]
    video=out/'release_vector_audit.mp4';writer=imageio_ffmpeg.write_frames(str(video),(1280,800),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for frame in range(360):
        i=min(frame,180);im=Image.new('RGB',(1280,800),bg);d=ImageDraw.Draw(im)
        text(d,28,18,'야구 투구 이탈 방향 검증 · 선수 성능 향상 영상 아님',30)
        text(d,28,66,'선수 ID 2 / 3034_4 | 회색: 모델 기준 · 파랑: 실제 전방 적분 | 의도적 자세 수정 없음',21)
        text(d,60,115,'측면: 홈 방향(+x) / 위(+z)');text(d,720,115,'정면: 1루 방향(+y) / 위(+z)')
        for front in (False,True):
            for points,color,width in [(rp[i],colors[0],8),(p[i],colors[1],3)]:
                pp=project(points,front)
                d.line([tuple(x) for x in pp],fill=color,width=width)
                for x,y in pp:d.ellipse((x-5,y-5,x+5,y+5),fill=color)
            if frame>=180:
                for name,col in [('fixed_length_wrist','#ffce66'),('assumed_ball','#fc738c')]:
                    vel=np.array(metrics[name]['velocity_m_s'])
                    pp=project(np.array([origin,origin+vel*.02]),front)
                    d.line([tuple(x) for x in pp],fill=col,width=4)
                    d.ellipse((pp[1,0]-5,pp[1,1]-5,pp[1,0]+5,pp[1,1]+5),fill=col)
        text(d,28,500,f"현재 공 가정점 속도 {np.linalg.norm(v[i,3])*3.6:.3f} km/h | 이탈 {np.linalg.norm(v[-1,3])*3.6:.3f} km/h | 기록 128.426 km/h",24)
        text(d,28,543,f'기준 대비 수치 재현 차이: 위치 최대 {delta[i].max():.4f} mm / 관절 최대 {maxang[i]:.5f}°',21)
        text(d,28,582,'이탈 방향: 손목 -17.92° → 고정 그립 공 -21.68° (아래) / 홈 방향 대비 옆으로 +22.50°',23,'#fc738c')
        text(d,28,626,'노랑: 손목 속도 · 분홍: 가정 공 속도 | 속도 화살표는 같은 20ms 축척',21)
        text(d,28,666,'관절 수정 0° / 타이밍 수정 0ms / 입력 힘 변경 없음 · 실제 공 궤적 정답은 이 자료에 없음',21)
        text(d,28,710,'현재 결론: 동작 처방 보류. 같은 투구의 실제 공 이탈 속도 벡터 검증이 먼저 필요합니다.',23)
        writer.send(np.asarray(im))
        if frame==240:im.save(out/'preview.png')
    writer.close()
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    meta=dict(frames=360,fps=30,duration_s=12,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        kind='diagnostic, no intervention or performance gain',final_speed_kmh=float(np.linalg.norm(v[-1,3])*3.6),
        max_replay_difference_mm=float(delta.max()),max_replay_rotation_deg=float(maxang.max()),decoder_pass=True)
    (out/'video.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta))


if __name__=='__main__':run()
