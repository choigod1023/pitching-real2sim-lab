"""Display newly acquired real pitching scenes and observed ball centers."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import imageio_ffmpeg as ff


def run():
    root=Path(__file__).resolve().parent;out=root/'artifacts/paired-source'
    result=json.loads((out/'new_ball_observations.json').read_text());font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
    fits={c['label']:c['fit'] for c in result['cases'] if c['threshold']==160}
    tracks={(r['label'],r['frame']):r for r in result['detections'] if r['threshold']==160 and r['status']=='candidate'}
    caps={s:cv2.VideoCapture(str(out/f'edgertronic_dynamic_baseballthrow{s}mph_3.mp4')) for s in [67,70]}
    anchors={67:375,70:455}
    for s,cap in caps.items():cap.set(cv2.CAP_PROP_POS_FRAMES,anchors[s]-45)
    movie=out/'new_baseball_hand_ball_comparison.mp4'
    writer=ff.write_frames(str(movie),(1280,992),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None);count=0;first=None;last=None
    for relative in range(-45,31):
        sheet=Image.new('RGB',(1280,992),'#101923');d=ImageDraw.Draw(sheet)
        d.text((16,8),'새 공개 야구 투구자료 | 실제 두 투구와 손·공 관측 비교',font=font,fill='white')
        d.text((16,43),'분리가 보이는 프레임을 기준으로 표시 정렬 · 같은 장면의 자세 수정 전후가 아닙니다',font=small,fill='#d1dae5')
        for j,label in enumerate([67,70]):
            frame=anchors[label]+relative;ok,bgr=caps[label].read();assert ok
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(im)
            points=[tracks[(label,f)]['pixel'] for f in range(anchors[label],min(frame,anchors[label]+4)+1) if (label,f) in tracks]
            if len(points)>=2:dr.line([tuple(p) for p in points],fill='#33efbf',width=3)
            if (label,frame) in tracks:
                u,v=tracks[(label,frame)]['pixel'];dr.ellipse((u-10,v-10,u+10,v+10),outline='#ff5774',width=3)
            pic=im.resize((640,512));sheet.paste(pic,(j*640,96))
            crop=im.crop((150,550,1050,950));crop=ImageOps.contain(crop,(640,256));sheet.paste(crop,(j*640+(640-crop.width)//2,608))
            text=f'파일명 {label}mph = {label*1.609344:.2f}km/h (명목 표기)'
            d.rectangle((j*640,96,j*640+640,130),fill='#152331');d.text((j*640+12,100),text,font=small,fill='white')
            d.text((j*640+12,868),f'원본 프레임 {frame} | 공 이동 {fits[label]["speed_px_per_frame"]:.2f}픽셀/프레임',font=small,fill='white')
        d.text((16,907),'추적점은 영상 관측 · 3D 구속·회전수·힘은 미검증 · 파일명 구속과 목록의 대응은 확인 중',font=small,fill='#cdd7e3')
        d.text((16,940),'OBP / Driveline Baseball · CC BY-NC-SA 4.0 및 원자료 사용 제한 적용',font=small,fill='#a5b8ca')
        arr=np.asarray(sheet)
        if first is None:
            first=arr.copy()
            for _ in range(30):writer.send(first);count+=1
        for _ in range(2):writer.send(arr);count+=1
        last=arr
        if relative==1:sheet.save(out/'new_comparison_preview.png')
    for cap in caps.values():cap.release()
    for _ in range(30):writer.send(last);count+=1
    writer.close();reader=ff.read_frames(str(movie),pix_fmt='rgb24');meta=next(reader);decoded=0
    for raw in reader:assert len(raw)==1280*992*3;decoded+=1
    reader.close();assert decoded==count
    manifest=dict(frames=decoded,fps=30,size=[1280,992],duration_s=decoded/30,video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
       scope='Two newly acquired real throws,2D ball observations only. Not an intervention simulation or verified performance gain.',
       source_anchors=anchors,display_alignment='first inspected clearly separated frame; not ground-truth release synchronization')
    (out/'new_video_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest));return manifest


if __name__=='__main__':run()
