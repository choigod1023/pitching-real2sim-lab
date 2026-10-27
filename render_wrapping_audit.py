"""Scientific diagnostic video, not an athlete performance simulation."""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg
OUT=Path(__file__).resolve().parent/'artifacts/wrapping-consistency'

def main():
    z=np.load(OUT/'wrapping_plot.npz');q=z['q_deg'];a=z['original_mm'];b=z['local_mm'];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    writer=imageio_ffmpeg.write_frames(str(OUT/'wrapping_consistency.mp4'),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for frame,idx in enumerate(np.r_[np.rint(np.linspace(0,len(q)-1,180)).astype(int),np.full(60,len(q)//2)]):
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        d.text((30,25),'견갑하근 경로 검산 | 선수 자세·구속 향상 영상이 아닙니다',font=font(30),fill='white')
        d.text((30,78),'파랑: 기존 경로 / 초록: 좁은 범위의 길이 기반 경로 / 점: 현재 검사 각도의 길이',font=font(24),fill='white')
        left,top,w,h=95,180,1210,420;lo=float(a.min())-.5;hi=float(a.max())+.5
        def point(x,y):return (left+(x-q[0])/(q[-1]-q[0])*w,top+h-(y-lo)/(hi-lo)*h)
        d.rectangle((left,top,left+w,top+h),outline=(100,110,125),width=2)
        for k in range(6):
            y=lo+(hi-lo)*k/5;d.text((25,point(q[0],y)[1]-13),f'{y:.1f}',font=font(18),fill='white')
        d.line([point(x,y) for x,y in zip(q,a)],fill=(90,170,255),width=3)
        mask=np.isfinite(b);d.line([point(x,y) for x,y in zip(q[mask],b[mask])],fill=(70,220,150),width=6)
        x,y=point(q[idx],a[idx]);d.ellipse((x-7,y-7,x+7,y+7),fill='white')
        for k in range(5):
            value=q[0]+(q[-1]-q[0])*k/4;d.text((point(value,lo)[0]-30,top+h+10),f'{value:.2f}°',font=font(20),fill='white')
        lines=['세로축: 근육 경로 길이(mm) / 가로축: 모델 어깨면 각도',
            '기존 경로: 0.0057° 변화에 길이 7.31 mm 급변하는 구간 확인',
            '국소 범위 ±0.115°: 길이 미분 불일치 0.00984 → 0.000000039 m/s',
            '넓은 범위 대안: 원래 길이 보존 실패 / 전체 투구에 적용하지 않음',
            '실제 힘 적용을 반영한 잔차: 134.36 → 137.61 N·m / 근육 검증 미통과',
            '구속 향상 후보 탐색 보류: 첫 계산 문제를 분리·수정하는 연구 단계']
        for k,text in enumerate(lines):d.text((30,665+k*42),text,font=font(24),fill='white' if k<3 else (255,190,110))
        writer.send(np.asarray(im))
        if frame==210:im.save(OUT/'preview.png')
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(OUT/'wrapping_consistency.mp4'),'-f','null','-'],check=True)

if __name__=='__main__':main()
