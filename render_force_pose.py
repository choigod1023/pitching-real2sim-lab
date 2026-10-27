"""Measured-arm context and simulated whole-body overlays, not source video."""
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from force_pose_calibration import OUT
from render_coupled_body_search import BONES
from render_speed_journey import imageio_ffmpeg

def main():
    report=json.loads((OUT/'3034_4.json').read_text());b,c=report['rows']
    z=np.load(OUT/'3034_4_overlay.npz');names=list(z['names']);t=z['time']
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    views=[np.array([[1,0,0],[0,0,-1]]),np.array([[0,1,0],[0,0,-1]])]
    arrays=[z['before'],z['after'],z['after_ground_only']]
    colors=[(90,175,255),(75,230,160),(255,155,75)]
    writer=imageio_ffmpeg.write_frames(str(OUT/'force_pose_overlay.mp4'),(1440,960),fps=30,
        codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    frames=np.r_[np.rint(np.linspace(0,len(t)-1,240)).astype(int),np.full(60,len(t)-1)]
    for fi,i in enumerate(frames):
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        texts=[('실측 지면 반력 + 자세 보정 | 3034_4',30,'white'),
            (f'실측 {b["measured_speed_kmh"]:.2f} km/h | 보조 힘 포함 모델 {b["speed_kmh"]:.2f} → {c["speed_kmh"]:.2f} km/h',26,'white'),
            ('파랑: 보정 전 / 초록: 보정 후 / 주황: 보정 후 보조 힘 제거 / 흰색: 관측 팔',22,'white'),
            ('원본 촬영 영상이 아닌 3D 관절 재구성입니다. 구속 향상 동작을 뜻하지 않습니다.',22,(255,190,110))]
        for j,(s,size,col) in enumerate(texts):d.text((25,18+j*43),s,font=font(size),fill=col)
        for vi,view in enumerate(views):
            left=25+vi*710;top=205;width=680;height=400
            cloud=np.concatenate([a.reshape(-1,3) for a in arrays])@view.T
            mn,mx=cloud.min(0),cloud.max(0);center=(mn+mx)/2
            scale=min((width-45)/max(mx[0]-mn[0],.1),(height-40)/max(mx[1]-mn[1],.1))
            def proj(p):return tuple((p@view.T-center)*scale+[left+width/2,top+height/2])
            d.rectangle((left,top,left+width,top+height),outline=(70,80,95),width=2)
            d.text((left+12,top+10),'X–Z 투영' if vi==0 else 'Y–Z 투영',font=font(18),fill='white')
            for a,color in zip(arrays,colors):
                for u,v in BONES:d.line([proj(a[i,names.index(u)]),proj(a[i,names.index(v)])],fill=color,width=3)
            d.line([proj(p) for p in z['observed_arm'][i]],fill='white',width=3)
        delta=z['delta'][i]
        lines=[f'FP 후 {(t[i]-t[0])*1000:.1f} ms | 골반 이동 보정 {np.linalg.norm(delta[:3])*1000:.2f} mm / 회전 {np.rad2deg(np.linalg.norm(delta[3:])):.3f}°',
            f'전신 기준 동작 대비 RMSE {b["pose_change_rmse_mm"]:.2f} → {c["pose_change_rmse_mm"]:.2f} mm (보조 힘 포함)',
            f'관측 팔 위치 RMSE {b["observed_arm_rmse_mm"]:.2f} → {c["observed_arm_rmse_mm"]:.2f} mm',
            f'남은 보조 힘 RMS {b["residual_force_rms_n"]:.0f} → {c["residual_force_rms_n"]:.0f} N / 토크 {b["residual_torque_rms_nm"]:.0f} → {c["residual_torque_rms_nm"]:.0f} N·m',
            f'보조 힘 제거: {c["ground_only_speed_kmh"]:.2f} km/h / 원래 동작 대비 {c["ground_only_original_pose_rmse_mm"]:.2f} mm',
            '내부 관절 각도는 유지하고 골반 위치·방향만 보정 / 근육 및 발 접촉 검증 아님',
            '구속과 자세를 함께 만족하는 보정은 아직 미통과']
        for j,line in enumerate(lines):d.text((25,625+j*43),line,font=font(22),fill='white' if j<6 else (255,190,110))
        writer.send(np.asarray(im))
        if fi==230:im.save(OUT/'preview.png')
    writer.close()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(OUT/'force_pose_overlay.mp4'),'-f','null','-'],check=True)

if __name__=='__main__':main()
