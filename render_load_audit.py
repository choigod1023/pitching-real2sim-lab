"""Overlay identical poses and expose model torque residuals before/after."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/shoulder-load-audit'

def main():
    z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');b=next(x for x in json.loads((OUT/'results.json').read_text())['reports'] if x['pitch']=='3034_4');a=next(x for x in json.loads((OUT/'corrected_results.json').read_text())['reports'] if x['pitch']=='3034_4')
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    writer=imageio_ffmpeg.write_frames(str(OUT/'mass_correction_overlay.mp4'),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    groups=[('girdle','견갑·쇄골'),('shoulder','어깨'),('elbow_forearm','팔꿈치·전완'),('wrist','손목')]
    maximum=max(r['groups'][g]['residual'] for q in [a,b] for r in q['rows'] for g,_ in groups)
    for frame,j in enumerate(np.r_[np.rint(np.linspace(0,len(a['rows'])-1,240)).astype(int),np.full(60,len(a['rows'])-1)]):
        ra,rb=a['rows'][j],b['rows'][j];i=ra['sample'];pos=z['predicted'][i]
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        for y,text in [(24,'질량 축척 오류 수정 전후 | 선수 3034_4'),(75,'파랑: 기존 계산 / 초록: 원본 질량 보존 계산'),(120,'관절 재구성 겹침: 각도·위치 변화 0 | 근육으로 재생한 전방 시뮬레이션 아님')]:d.text((25,y),text,font=font(29 if y==24 else 23),fill='white')
        for v,view in enumerate([np.array([[1,0,0],[0,-1,0]]),np.array([[0,0,1],[0,-1,0]])]):
            left=25+v*315;top=200;w=300;h=430;cloud=z['predicted'].reshape(-1,3)@view.T;mn,mx=cloud.min(0)-.12,cloud.max(0)+.12;scale=min((w-20)/(mx[0]-mn[0]),(h-20)/(mx[1]-mn[1]));center=(mn+mx)/2
            def proj(p):return tuple((p@view.T-center)*scale+[left+w/2,top+h/2])
            d.rectangle((left,top,left+w,top+h),outline=(75,85,100),width=2)
            for col,width in [((90,170,255),7),((70,220,150),3)]:
                d.line([proj(pos[k]) for k in [0,1,2,0]],fill=col,width=width)
                d.line([proj(pos[k]) for k in [0,3,4,5]],fill=col,width=width)
        for k,(g,label) in enumerate(groups):
            y=210+k*102;d.text((690,y),label,font=font(24),fill='white')
            for offset,r,color in [(36,rb,(90,170,255)),(60,ra,(70,220,150))]:
                length=r['groups'][g]['residual']/maximum*480;d.rectangle((690,y+offset,690+length,y+offset+16),fill=color)
            d.text((1110,y),f'{rb["groups"][g]["residual"]:.1f} → {ra["groups"][g]["residual"]:.1f}',font=font(22),fill='white')
        lines=[f'현재 프레임 잔차 {rb["residual"]:.1f} → {ra["residual"]:.1f} N·m',f'투구 전체 RMS {b["rms"]["residual"]:.2f} → {a["rms"]["residual"]:.2f} N·m',
               '상완골 모델 질량 5.31 → 1.77 kg: 실제 선수의 체중 변화가 아닌 코드 수정',
               '실측 구속 128.426 km/h | 이 모델의 공 구속 예측은 아직 없음',
               '근육 최대 힘·관절 자세 동일 / 선수 개인의 질량·관성 및 힘·일은 미검증',
               '잔차 검증 미통과: 구속 향상 후보 탐색 보류']
        for k,text in enumerate(lines):d.text((25,675+k*42),text,font=font(24),fill='white' if k<3 else (255,190,110))
        writer.send(np.asarray(im))
        if frame==180:im.save(OUT/'preview.png')
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(OUT/'mass_correction_overlay.mp4'),'-f','null','-'],check=True)

if __name__=='__main__':main()
