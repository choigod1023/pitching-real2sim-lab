import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg
OUT=Path(__file__).resolve().parent/'artifacts/shoulder-marker-stability'

def main():
    z=np.load(OUT/'overlay.npz');r=next(x for x in json.loads((OUT/'summary.json').read_text())['rows'] if x['pitch']=='3034_4')
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    views=[np.array([[1,0,0],[0,-1,0]]),np.array([[0,0,1],[0,-1,0]])]
    arrays=[z['before'],z['after'],z['observed']];colors=[(90,170,255),(70,220,150),(230,230,235)]
    rotations=[z['before_rotation'],z['after_rotation'],z['target_rotation']]
    writer=imageio_ffmpeg.write_frames(str(OUT/'marker_stability_overlay.mp4'),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for fi,i in enumerate(np.r_[np.rint(np.linspace(0,len(z['time'])-1,240)).astype(int),np.full(60,len(z['time'])-1)]):
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        headlines=['전완·손 방향 마커 적용 전후 | 3034_4',
            '파랑: 수정 전 / 초록: 방향 마커 + 구간 전체 적합 / 흰색: 관측 기준',
            '짧은 선: 손 방향 비교 | 원본 영상이 아닌 관절 재구성 · 근육 구동 시뮬레이션 아님']
        for j,s in enumerate(headlines):d.text((25,22+48*j),s,font=font(27 if j==0 else 23),fill='white')
        for vi,view in enumerate(views):
            left=25+710*vi;top=185;w=680;h=420
            cloud=np.concatenate([a.reshape(-1,3) for a in arrays])@view.T;mn,mx=cloud.min(0)-.15,cloud.max(0)+.15
            scale=min((w-40)/(mx[0]-mn[0]),(h-40)/(mx[1]-mn[1]));center=(mn+mx)/2
            def proj(p):return tuple((p@view.T-center)*scale+[left+w/2,top+h/2])
            d.rectangle((left,top,left+w,top+h),outline=(70,80,95),width=2)
            for a,col,rot in zip(arrays,colors,rotations):
                d.line([proj(a[i,j]) for j in [0,1,2,0]],fill=col,width=2)
                d.line([proj(a[i,j]) for j in [0,3,4,5]],fill=col,width=3)
                # Same body-local axis for both model and registered skin-frame target.
                d.line([proj(a[i,5]),proj(a[i,5]+rot[i,2,:,0]*.12)],fill=col,width=5)
        lines=[f'앞발 착지 후 {(z["time"][i]-z["time"][0])*1000:.1f} ms',
            f'팔 위치 RMSE {r["before_arm_rmse_mm"]:.2f} → {r["after_arm_rmse_mm"]:.2f} mm',
            f'전완 방향 RMSE {r["before_orientation_rmse_deg"]["radius"]:.2f} → {r["after_orientation_rmse_deg"]["radius"]:.2f}°',
            f'손 방향 RMSE {r["before_orientation_rmse_deg"]["hand"]:.2f} → {r["after_orientation_rmse_deg"]["hand"]:.2f}°',
            f'오류 수정 기준 힘 배분 잔차 {r["bugfix_only"]["hill_residual_rms_nm"]:.1f} → {r["direction_markers"]["hill_residual_rms_nm"]:.1f} N·m',
            f'활성화 끝점 평형의 최대 장력 {r["bugfix_only"]["max_endpoint_tendon_force_n"]/1000:.2f} → {r["direction_markers"]["max_endpoint_tendon_force_n"]/1000:.2f} kN',
            '손목 고정 모델과 관측 손 회전이 충돌함 / 근육·이탈 검증 미통과',
            '공 구속 예측·성능 향상 후보 아님 / 관절 범위와 근육 최대 힘은 변경하지 않음']
        for j,s in enumerate(lines):d.text((25,625+j*39),s,font=font(22),fill='white' if j<5 else (255,190,110))
        writer.send(np.asarray(im))
        if fi==230:im.save(OUT/'preview.png')
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(OUT/'marker_stability_overlay.mp4'),'-f','null','-'],check=True)

if __name__=='__main__':main()
