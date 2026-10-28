"""Two-camera overlay of observed ball vs fixed-grip diagnostic; no performance gain claim."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_speed_journey import imageio_ffmpeg
from .camera import load_cameras, project
from .detect import find_video


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--cv-root', type=Path, required=True)
    ap.add_argument('--cal-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    report = json.loads((a.analysis/'relative_motion.json').read_text(encoding='utf-8'))
    nominal = report['nominal_c3d_minus_video_frame']
    z = np.load(a.analysis/f'relative_offset{nominal}.npz')
    frames = z['frames']; last=977
    cameras=(19,22); K,P=load_cameras(a.cal_dir,cameras)
    images={}
    for cam in cameras:
        cap=cv2.VideoCapture(str(find_video(a.cv_root,report['trial'],cam)))
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0]))
        for f in range(int(frames[0]),last+1):
            ok,im=cap.read()
            if not ok:raise RuntimeError(f'Missing camera {cam}, frame {f}')
            images[(cam,f)]=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((704,396))
        cap.release()
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    blue=(60,180,255);yellow=(255,210,65);green=(90,240,170)
    rows={r['frame']:r for r in report['results_by_c3d_offset'][str(nominal)]['frames']}
    end=int(np.where(frames==last)[0][0]); local=z['local_ball_m'][:end+1]*1000
    initial=np.asarray(report['results_by_c3d_offset'][str(nominal)]['summary']['fixed_local_vector_mm'])
    left,top,w,h=82,625,1270,180;low=min(local.min(),initial.min())-8;high=max(local.max(),initial.max())+8
    def pt(f,val):return (left+(f-frames[0])/(last-frames[0])*w,top+h-(val-low)/(high-low)*h)
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/'hand_ball_relative_overlay.mp4'
    writer=imageio_ffmpeg.write_frames(str(path),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
    writer.send(None)
    sequence=np.rint(np.linspace(0,end,240)).astype(int)
    for j,i in enumerate(sequence):
        f=int(frames[i]);row=rows[f]
        im=Image.new('RGB',(1440,960),(15,22,33));d=ImageDraw.Draw(im)
        d.text((24,16),'야구 투구 | 손에 고정한 공 가정과 관측 공의 차이',font=font(29),fill='white')
        d.text((24,62),'파랑: 영상에서 추적한 공   노랑: 초기 손 상대 위치 고정   초록: Theia 추정 손목',font=font(23),fill='white')
        for cam,xoff in zip(cameras,(12,724)):
            im.paste(images[(cam,f)],(xoff,108));d=ImageDraw.Draw(im)
            points={k:project(z[k][i]*1000,K[cam],P[cam])*.55+[xoff,108]
                    for k in ('ball_m','fixed_ball_m','hand_m')}
            d.line([tuple(points['hand_m']),tuple(points['ball_m'])],fill=blue,width=3)
            d.line([tuple(points['hand_m']),tuple(points['fixed_ball_m'])],fill=yellow,width=3)
            for key,color in [('hand_m',green),('fixed_ball_m',yellow),('ball_m',blue)]:
                x,y=points[key];d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3)
            d.rectangle((xoff,108,xoff+270,141),fill=(10,16,23))
            d.text((xoff+8,111),f'카메라 {cam} | 인덱스 {f} (0부터)',font=font(18),fill='white')
        d.text((24,520),f'손목–공 거리 {row["distance_mm"]:.1f} mm   |   고정 가정과 위치 차이 {row["fixed_position_error_mm"]:.1f} mm',font=font(25),fill='white')
        d.text((24,560),'아래: 손 기준 공 좌표(mm). 실선은 관측, 점선은 초기 구간 956–963의 평균.',font=font(22),fill='white')
        d.rectangle((left,top,left+w,top+h),outline=(85,100,120),width=1)
        for axis,color in enumerate((blue,green,(230,125,200))):
            d.line([pt(frame,val) for frame,val in zip(frames[:end+1],local[:,axis])],fill=color,width=3)
            y=pt(frames[0],initial[axis])[1]
            for x in range(left,left+w,18):d.line((x,y,x+8,y),fill=color,width=1)
            d.text((left+axis*280,top-30),f'Theia 축 {axis+1}: {local[i,axis]:+.1f}',font=font(19),fill=color)
        x=pt(f,low)[0];d.line((x,top,x,top+h),fill='white',width=2)
        for val in np.linspace(low,high,4):d.text((8,pt(frames[0],val)[1]-10),f'{val:.0f}',font=font(17),fill='white')
        d.text((24,831),'360 fps 원영상 약 137배 느리게 | 이탈 시점 미확정 | 자세·근력 수정이나 구속 향상 결과가 아닙니다',font=font(22),fill=yellow)
        d.text((24,874),'정렬·시간 오차가 섞여 있어 손가락 효과로 단정할 수 없습니다. 원영상의 화면 카운터는 인덱스+1.',font=font(22),fill='white')
        writer.send(np.asarray(im))
        if j==int(np.argmin(abs(sequence-np.where(frames==974)[0][0]))):im.save(a.out/'relative_preview.png')
    writer.close()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True)
    print(path.resolve())


if __name__=='__main__':main()
