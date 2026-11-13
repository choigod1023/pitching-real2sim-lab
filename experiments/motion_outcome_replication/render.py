"""Source-video comparison only: no invented correction or causal animation."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE=(1600,960)
BG=(18,26,32); FG=(236,240,241); MUTED=(174,190,197)
COLORS=((92,212,224),(255,193,96))


def selected_pairs(result):
    rows=[]
    for cohort in ('speed_contrast','matched_speed_spin'):
        seen=set()
        for pair in sorted(result['pairs'],key=lambda p:(p['date'],p['id'])):
            if pair['cohort']!=cohort or pair['game'] in seen:
                continue
            seen.add(pair['game']);rows.append(pair)
            if len(seen)==(2 if cohort=='speed_contrast' else 3):
                break
    return rows


def decode_selected(path, indices):
    need=set(map(int,indices)); images={}
    cap=cv2.VideoCapture(str(path)); index=0
    while index<=max(need):
        ok,frame=cap.read()
        if not ok:break
        if index in need:images[index]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        index+=1
    cap.release()
    if set(images)!=need:raise ValueError('Missing source frame')
    return images


def limbs(points,scores,masks):
    valid=np.isfinite(points).all(-1)&(scores>=.5)
    edges=[]
    if masks['trunk_tilt']:
        edges.extend([(points[a],points[b]) for a,b in [(11,12),(23,24),(11,23),(12,24)]])
        edges.append((points[[23,24]].mean(0),points[[11,12]].mean(0)))
    if masks['lead_knee']:edges.extend([(points[23],points[25]),(points[25],points[27])])
    if masks['throwing_elbow']:edges.extend([(points[12],points[14]),(points[14],points[16])])
    for a,b in [(24,26),(26,28)]:
        if valid[a] and valid[b]:edges.append((points[a],points[b]))
    return edges


def measure_text(pair, feature):
    f=pair['features'][feature]
    if f['nominal'] is None:return '관측 부족'
    lo,hi=f['range']
    return f"{f['nominal']:+.2f}°  ({lo:+.2f} ~ {hi:+.2f}°)"


def render(folder,font_path):
    result=json.loads((folder/'results.private.json').read_text(encoding='utf-8'))
    z=np.load(folder/'mapped-tracks.private.npz',allow_pickle=False)
    pairs=selected_pairs(result)
    fonts={n:ImageFont.truetype(str(font_path),n) for n in (21,23,25,27,30,34)}
    frames_written=0
    master=folder/'comparison-master.mp4'; target=folder/'skenes-motion-outcomes-original-overlay.mp4'
    w=cv2.VideoWriter(str(master),cv2.VideoWriter_fourcc(*'mp4v'),30,SIZE)
    if not w.isOpened():raise RuntimeError('Video writer unavailable')
    captures=[]
    for scene,pair in enumerate(pairs):
        keys=pair['keys'];clips=[result['clips'][k] for k in keys]
        relative_times=np.arange(-233.333,100.001,16.66667)
        rows=[];images=[];scales=[]
        for key,clip in zip(keys,clips):
            ts=z[key+'_times']; first=clip['first_index']; rel=ts-ts[first]
            row=np.abs(rel[:,None]-relative_times[None,:]).argmin(axis=0)
            if np.max(abs(rel[row]-relative_times))>17:raise ValueError('Insufficient phase coverage for replay')
            rows.append(row);images.append(decode_selected(clip['video'],z[key+'_frames'][row]))
            pp=z[key+'_points'];valid=z[key+'_trunk_tilt_mask']
            length=np.linalg.norm(pp[:,[11,12]].mean(1)-pp[:,[23,24]].mean(1),axis=1)
            scales.append(float(np.median(length[valid])) if valid.any() else None)
        for tick,time in enumerate(relative_times):
            im=Image.new('RGB',SIZE,BG);draw=ImageDraw.Draw(im)
            group='빠른 공 / 느린 공' if pair['cohort']=='speed_contrast' else '비슷한 구속 · RPM/수직 차이'
            draw.text((24,14),f"스킨스 포심  |  {pair['date']}  |  {group}",font=fonts[34],fill=FG)
            draw.text((24,64),'실제 두 투구 비교 · 관절 수정 시뮬레이션 아님 · 다른 경기의 반복 관계 미확인',font=fonts[25],fill=MUTED)
            for side,(key,clip,color) in enumerate(zip(keys,clips,COLORS)):
                x=24+side*568;row=int(rows[side][tick]);frame=int(z[key+'_frames'][row]);record=clip['record']
                label=('느린 공','빠른 공')[side] if pair['cohort']=='speed_contrast' else ('낮은 RPM·수직','높은 RPM·수직')[side]
                draw.text((x,106),f"{label}  {record['speed_kmh']:.2f} km/h · {record['rpm']:.0f} RPM",font=fonts[25],fill=color)
                draw.text((x,143),f"수직 {record['vertical_cm']:.2f} cm · 좌우 {record['horizontal_cm']:+.2f} cm",font=fonts[23],fill=FG)
                raw=images[side][frame];height,width=raw.shape[:2]
                cropbox=(int(.28*width),int(.28*height),int(.72*width),int(.98*height))
                cw,ch=cropbox[2]-cropbox[0],cropbox[3]-cropbox[1]
                scale=min(544/cw,500/ch);sw,sh=round(cw*scale),round(ch*scale)
                dx=x+(544-sw)//2;dy=188+(500-sh)//2
                im.paste(Image.fromarray(raw).crop(cropbox).resize((sw,sh),Image.Resampling.BILINEAR),(dx,dy))
                pp=z[key+'_points'][row];ss=z[key+'_scores'][row]
                masks={name:bool(z[key+'_'+name+'_mask'][row]) for name in ('trunk_tilt','lead_knee','throwing_elbow','arm_direction')}
                edge=limbs(pp,ss,masks)
                for a,b in edge:
                    aa=(dx+(a[0]-cropbox[0])*scale,dy+(a[1]-cropbox[1])*scale)
                    bb=(dx+(b[0]-cropbox[0])*scale,dy+(b[1]-cropbox[1])*scale)
                    draw.line([aa,bb],fill=(12,20,25),width=8);draw.line([aa,bb],fill=color,width=4)
                dt=z[key+'_times'][row]-z[key+'_times'][clip['first_index']]
                draw.text((x,698),f"원본 F{frame}  ·  첫 자유 공 대비 {dt:+.1f} ms",font=fonts[23],fill=FG)
                if masks['trunk_tilt'] and scales[side]:
                    pelvis=pp[[23,24]].mean(0);scale2=105/scales[side]
                    for a,b in edge:
                        aa=(1365+(a[0]-pelvis[0])*scale2,460+(a[1]-pelvis[1])*scale2)
                        bb=(1365+(b[0]-pelvis[0])*scale2,460+(b[1]-pelvis[1])*scale2)
                        draw.line([aa,bb],fill=color,width=4)
            draw.text((1180,112),'동작 겹쳐 보기',font=fonts[30],fill=FG)
            draw.text((1180,154),'골반 위치·몸통 길이 정규화',font=fonts[21],fill=MUTED)
            draw.text((1180,190),'2D 표시용 · 실제 이동량 아님',font=fonts[21],fill=MUTED)
            draw.text((1180,688),'안 보이는 팔은 연결하지 않음',font=fonts[21],fill=MUTED)
            delta=pair['delta']
            draw.line((24,745,1576,745),fill=(74,90,99),width=1)
            draw.text((24,764),f"오른쪽 - 왼쪽  |  구속 {delta['speed_kmh']:+.2f} km/h   RPM {delta['rpm']:+.0f}   수직 {delta['vertical_cm']:+.2f} cm   좌우 {delta['horizontal_cm']:+.2f} cm",font=fonts[27],fill=FG)
            draw.text((24,810),'직전 경로의 앞무릎 각도 차이: '+measure_text(pair,'lead_knee_level'),font=fonts[25],fill=FG)
            draw.text((24,851),'직전 경로의 몸통 기울기 차이: '+measure_text(pair,'trunk_tilt_level'),font=fonts[25],fill=FG)
            draw.text((24,911),'구간 -150~-33ms 중앙값 · 괄호는 시점 ±1프레임 민감도 · 공 수치는 실제 기록, 동작의 효과 예측이 아님',font=fonts[21],fill=MUTED)
            if tick==9:
                path=folder/f'comparison-scene-{scene+1}.jpg';im.save(path,quality=93);captures.append(path.name)
            bgr=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
            for _ in range(7):w.write(bgr);frames_written+=1
    w.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(master),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    cap=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if (frame.shape[1],frame.shape[0])!=SIZE:raise ValueError('Unexpected output size')
        decoded+=1
    cap.release()
    if decoded!=frames_written:raise ValueError('Incomplete output decode')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(target),'-f','null','-'],check=True)
    receipt={'pairs':[p['id'] for p in pairs],'frames':decoded,'fps':30,'seconds':decoded/30,'size':SIZE,'file':target.name,'snapshots':captures,'source_images_generated':False}
    (folder/'video-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--font',type=Path,default=Path(os.environ.get('WINDIR','/Windows'))/'Fonts/malgun.ttf')
    args=parser.parse_args();render(args.folder,args.font)
