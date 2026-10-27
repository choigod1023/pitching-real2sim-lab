"""Overlay actual integrated baseline/rejected body candidate and observed context."""
import json,subprocess
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from coupled_body_search import OUT,positions,groups,build
from render_speed_journey import imageio_ffmpeg
F='C:/Windows/Fonts/malgun.ttf'
def font(n):return ImageFont.truetype(F,n)
BONES=[('rear_hip','rear_knee_jc'),('rear_knee_jc','rear_ankle_jc'),('lead_hip','lead_knee_jc'),('lead_knee_jc','lead_ankle_jc'),('rear_hip','lead_hip'),('rear_hip','shoulder_jc'),('lead_hip','glove_shoulder_jc'),('shoulder_jc','glove_shoulder_jc'),('shoulder_jc','elbow_jc'),('elbow_jc','wrist'),('wrist','ball_center'),('glove_shoulder_jc','glove_elbow_jc'),('glove_elbow_jc','glove_wrist_jc')]

def main():
    report=json.loads((OUT/'report.json').read_text());r=report['best_diagnostic'];base=report['baseline'];work0=sum(x['positive_j'] for x in base['costs'].values());work1=sum(x['positive_j'] for x in r['costs'].values());torque_delta=r['costs']['rear_knee']['peak']-base['costs']['rear_knee']['peak'];z=np.load(OUT/'overlay_states.npz')
    m,_,_,_,observed=build(float(z['dt']));names,b=positions(m,z['qbaseline']);_,c=positions(m,z['qcandidate']);times=z['time']-z['time'][0]
    speeds=[];d=mujoco.MjData(m)
    for q,v in [(z['qbaseline'],z['vbaseline']),(z['qcandidate'],z['vcandidate'])]:
        s=[]
        for qq,vv in zip(q,v):
            d.qpos[:]=qq;d.qvel[:]=vv;mujoco.mj_forward(m,d);jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,m.site('ball_center').id);s.append(np.linalg.norm(jp@vv)*3.6)
        speeds.append(s)
    views=[np.array([[1.,0,0],[0,0,-1]]),np.array([[0,1.,0],[0,0,-1]])];limits=[]
    for v in views:
        p=np.concatenate([b.reshape(-1,3),c.reshape(-1,3)])@v.T;limits.append((p.min(0),p.max(0)))
    writer=imageio_ffmpeg.write_frames(str(OUT/'coupled_body_rejected_overlay.mp4'),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    ids=np.r_[np.rint(np.linspace(0,len(b)-1,300)).astype(int),np.full(60,len(b)-1)]
    knee=int(m.joint('rear_knee').qposadr[0]);torso=groups(m)['torso_rotation'];dq=np.zeros(m.nv)
    for fi,i in enumerate(ids):
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        d.text((28,16),'뒤무릎 + 골반–몸통 변경 | 전신 연결 시험 모델',font=font(30),fill='white')
        d.text((28,63),'제한 미통과 — 구속 향상 후보로 채택하지 않음',font=font(29),fill=(255,115,100))
        d.text((28,108),f'기준 {report["baseline"]["speed_kmh"]:.3f} → 변경 {r["speed_kmh"]:.3f} km/h  (+{r["gain_kmh"]:.3f})',font=font(26),fill='white')
        d.text((28,149),'파랑: 기준 적분 / 주황: 변경 적분 / 회색: 관찰 하체·몸통 좌표',font=font(23),fill=(200,210,220))
        for vi,v in enumerate(views):
            left=25+vi*700;top=195;w=675;h=420;mn,mx=limits[vi];scale=min((w-50)/max(mx[0]-mn[0],.1),(h-25)/max(mx[1]-mn[1],.1));center=(mn+mx)/2
            def proj(p):return tuple((p@v.T-center)*scale+np.array([left+w/2,top+h/2]))
            d.rectangle((left,top,left+w,top+h),outline=(60,70,90),width=2)
            for a,bb in BONES:
                if a in observed and bb in observed:d.line([proj(observed[a][i]),proj(observed[bb][i])],fill=(100,110,125),width=2)
            for points,color in [(b[i],(75,180,255)),(c[i],(255,165,70))]:
                for a,bb in BONES:d.line([proj(points[names.index(a)]),proj(points[names.index(bb)])],fill=color,width=4)
                for p in points:
                    xx,yy=proj(p);d.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color)
        mujoco.mj_differentiatePos(m,dq,1,z['qbaseline'][i],z['qcandidate'][i]);kd=np.degrees(z['qcandidate'][i,knee]-z['qbaseline'][i,knee]);td=np.degrees(np.linalg.norm(dq[torso]))
        d.text((28,632),f'FP 후 {times[i]*1000:.1f} ms | 뒤무릎 각도차 {kd:+.2f}° | 몸통 상대회전 차이 {td:.2f}°',font=font(24),fill='white')
        d.text((28,674),f'현재 공 중심 속력 {speeds[0][i]:.2f} → {speeds[1][i]:.2f} km/h | 무릎 피크 실제 이동 +{r["actual_knee_peak_shift_ms"]:.2f} ms',font=font(23),fill='white')
        d.text((28,716),f'전체 양의 일 {work0:.2f} → {work1:.2f} J  /  뒤무릎 최대 토크 {torque_delta:+.2f} N·m',font=font(24),fill=(255,150,125))
        d.text((28,758),f'최대 관절 변경 {r["max_joint_change_deg"]:.2f}° (한도 3°) / 발목 변위 {r["max_ankle_change_mm"]:.2f} mm (한도 20 mm)',font=font(23),fill=(255,150,125))
        d.text((28,810),f'실측 {report["measured_speed_kmh"]:.2f} km/h와 기준 모델 차이 {report["measured_speed_kmh"]-base["speed_kmh"]:.2f} km/h — 실제 선수 효과로 해석 불가',font=font(23),fill='white')
        d.text((28,853),'지면 접촉 대신 외부 골반 힘·토크 사용 / 근육 미검증 / 착지 후 분리각 변경',font=font(22),fill=(195,205,220))
        d.text((28,895),f'입력 시간 간격 절반: 증가량 {report["input_sampling_refinement"]["diagnostic"]["gain_kmh"]:+.3f} km/h, 여전히 제한 미통과',font=font(22),fill=(195,205,220))
        writer.send(np.asarray(im))
        if fi==260:im.save(OUT/'overlay_preview.png')
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(OUT/'coupled_body_rejected_overlay.mp4'),'-f','null','-'],check=True)
    print(OUT/'coupled_body_rejected_overlay.mp4')
if __name__=='__main__':main()
