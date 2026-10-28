"""Overlay conditional baseball flights and paired synthetic endpoint stresses."""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_speed_journey import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/performance-frontier'


def main():
    r=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    by={x['label']:x for x in r['records']};base=by['baseline']
    z=np.load(OUT/'flight_paths.npz');points=z['paths'];indices=z['indices'].tolist()
    m=np.load(OUT/'plot_inputs.npz');ends=m['mc_endpoints']
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    blue=(70,180,255);orange=(255,170,65)
    size=(1440,960);video=OUT/'performance_comparison.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),size,fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for label,title in [('speed_focus','1. 구속 우선 조건'),('movement_focus','2. 수직 움직임 우선 조건'),('robust_joint_release_candidate','3. 종합 조건 | 9개 공학적 민감도 조건 통과')]:
        if label not in by:continue
        case=by[label];a,b=indices.index(base['index']),indices.index(case['index'])
        for frame in range(240):
            im=Image.new('RGB',size,(15,22,34));d=ImageDraw.Draw(im)
            d.text((24,18),'야구 공 비행 가정 실험 | 실제 선수의 신체 교정 효과는 아직 미검증',font=font(27),fill='white')
            d.text((24,63),title,font=font(27),fill=orange)
            d.text((24,105),'파랑: 가상 기준 포심   주황: 변경한 이탈 조건   같은 공 이탈 에너지 99.554 J',font=font(23),fill='white')
            left,top,w,h=40,185,765,345
            def side(p):return (left+p[1]/16.8*w,top+h-p[2]/2.1*h)
            d.rectangle((left,top,left+w,top+h),outline=(70,90,110),width=2)
            d.text((left,top-32),'측면 | 가로: 이탈 기준 거리(m) / 세로: 높이(m)',font=font(20),fill='white')
            for k in (a,b):d.line([side(p) for p in points[:,k,:3]],fill=blue if k==a else orange,width=4)
            t=frame/239*max(base['flight_ms'],case['flight_ms'])/1000
            step=min(int(t/.001),len(points)-1)
            for k,color in ((a,blue),(b,orange)):
                x,y=side(points[step,k,:3]);d.ellipse((x-7,y-7,x+7,y+7),fill=color)
            tx,ty=side([0,16.8,.9]);d.line((tx-10,ty,tx+10,ty),fill='white',width=2)
            for distance in (0,4,8,12,16.8):d.text((left+distance/16.8*w-10,top+h+8),f'{distance:g}',font=font(17),fill='white')
            for height in (0,1,2):d.text((7,side([0,0,height])[1]-10),str(height),font=font(17),fill='white')
            xl,yt,sw,sh=882,185,490,345
            d.rectangle((xl,yt,xl+sw,yt+sh),outline=(70,90,110),width=2)
            d.text((xl,yt-32),'가상 도착점 | 같은 256개 오차 표본',font=font(20),fill='white')
            def endpoint(p):return (xl+sw/2+p[0]/.22*sw/2,yt+sh/2-(p[1]-.9)/.22*sh/2)
            for k,color in ((a,blue),(b,orange)):
                for point in ends[k]:
                    x,y=endpoint(point);d.ellipse((x-2,y-2,x+2,y+2),fill=color)
            x,y=endpoint([0,.9]);d.line((x-10,y,x+10,y),fill='white',width=2);d.line((x,y-10,x,y+10),fill='white',width=2)
            d.text((xl+20,top+h+8),'목표 중심 / 표시 범위 좌우·상하 22 cm',font=font(18),fill='white')
            lines=[
                f'구속 {base["speed_kmh"]:.2f} → {case["speed_kmh"]:.2f} km/h  |  수직 움직임 대리값 {base["ivb_proxy_cm"]:.2f} → {case["ivb_proxy_cm"]:.2f} cm',
                f'비행 시간 {base["flight_ms"]:.2f} → {case["flight_ms"]:.2f} ms  |  가상 도착 오차 RMS {base["mc_rms_target_cm"]:.2f} → {case["mc_rms_target_cm"]:.2f} cm',
                f'회전수 {base["rpm"]:.0f} → {case["rpm"]:.0f} rpm / 횡회전 비율 {base["efficiency"]*100:.0f} → {case["efficiency"]*100:.0f}%',
                f'이탈점 변화: 앞 {case["forward_m"]*100:+.0f} cm / 높이 {(case["height_m"]-1.8)*100:+.0f} cm | 평균 조준 방향은 각 조건에서 다시 맞춤',
                '구속 증가는 에너지 재배분 가정에서 계산됨 / 관절·근육 변경에서 예측한 증가량이 아님',
                '움직임은 구위 대리 지표, 가상 도착 오차는 실제 제구가 아님 / 선수 힘·일·부하 미검증']
            for j,text in enumerate(lines):d.text((24,595+j*51),text,font=font(23 if j<4 else 21),fill='white' if j<4 else (255,210,135))
            writer.send(np.asarray(im))
            if label=='robust_joint_release_candidate' and frame==160:im.save(OUT/'performance_preview.png')
    writer.close()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],check=True)
    print(video.resolve())


if __name__=='__main__':main()
