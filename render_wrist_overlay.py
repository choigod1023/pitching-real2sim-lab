import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg
OUT=Path(__file__).resolve().parent/'artifacts/wrist-muscle'

def main():
    z=np.load(OUT/'overlay.npz');r=next(x for x in json.loads((OUT/'comparison.json').read_text())['rows'] if x['pitch']=='3034_4')
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    views=[np.array([[1,0,0],[0,-1,0]]),np.array([[0,0,1],[0,-1,0]])]
    arrays=[z['before'],z['after'],z['observed']];colors=[(90,170,255),(70,220,150),(230,230,235)]
    rotations=[z['before_rotation'],z['after_rotation'],z['target_rotation']]
    writer=imageio_ffmpeg.write_frames(str(OUT/'wrist_overlay.mp4'),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for fi,i in enumerate(np.r_[np.rint(np.linspace(0,len(z['time'])-1,240)).astype(int),np.full(60,len(z['time'])-1)]):
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        headlines=['손목 자유도·근육 6개 연결 전후 | 3034_4',
            '파랑: 수정 전 / 초록: 손목 연동 모델 / 흰색: 관측 기준',
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
        a=r['before'];b=r['after'];bm=r['before_muscle'];am=r['after_muscle']
        delta=np.rad2deg(np.arccos(np.clip((np.trace(z['before_rotation'][i,2].T@z['after_rotation'][i,2])-1)/2,-1,1)))
        displacement=np.linalg.norm(z['after'][i,5]-z['before'][i,5])*1000
        lines=[f'현재 프레임 손 방향 수정 {delta:.1f}° / 손목 위치 수정 {displacement:.2f} mm',
            f'팔 위치 RMSE {a["arm_rmse_mm"]:.2f} → {b["arm_rmse_mm"]:.2f} mm',
            f'손 방향 RMSE {a["orientation_rmse_deg"]["hand"]:.2f} → {b["orientation_rmse_deg"]["hand"]:.2f}°',
            f'근육 힘 배분 잔차 {bm["residual_nm"]:.1f} → {am["residual_nm"]:.1f} N·m',
            f'평형 끝점 최대 장력 {bm["max_endpoint_n"]/1000:.2f} → {am["max_endpoint_n"]/1000:.2f} kN',
            '최대 장력은 선수의 실측 힘이 아닌 활성화 끝점 진단값',
            '실측 구속 128.426 km/h / 이번 모델의 공 구속 예측은 아직 없음',
            '힘·일 및 공 이탈 검증 미통과: 구속 향상 후보 탐색 보류']
        for j,s in enumerate(lines):d.text((25,625+j*39),s,font=font(22),fill='white' if j<5 else (255,190,110))
        writer.send(np.asarray(im))
        if fi==230:im.save(OUT/'preview.png')
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(OUT/'wrist_overlay.mp4'),'-f','null','-'],check=True)

if __name__=='__main__':main()
