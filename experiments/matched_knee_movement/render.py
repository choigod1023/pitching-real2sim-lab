"""Original-pitch overlays with observed motion and held-game predictions."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .analyze import read

SIZE=(1600,1050)
COLORS=((92,212,224),(255,193,96))


def predictions(result):
    found={}
    for cohort in ('speed_contrast','matched_speed_spin'):
        model=result['analyses']['pair_camera_only/'+cohort]['models'][0]['nominal']
        for fold in model['folds']:
            for row in fold['predictions']:found[row['pair']]=row
    return found


def selected_pairs(result):
    selected=[]
    for cohort in ('speed_contrast','matched_speed_spin'):
        seen=set()
        for p in sorted(result['rows'],key=lambda p:(p['date'],p['id'])):
            if p['cohort']!=cohort or p['game'] in seen or not(p['context_match']['strict'] and p['pair_camera']['pass']):continue
            if p['features']['lead_knee_change']['nominal'] is None:continue
            selected.append(p);seen.add(p['game'])
    return selected


def source_frames(path,frames):
    need=set(map(int,frames));cap=cv2.VideoCapture(str(path));images={}
    for index in range(max(need)+1):
        ok,frame=cap.read()
        if not ok:break
        if index in need:images[index]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cap.release()
    if set(images)!=need:raise ValueError('Missing source frames')
    return images


def edges(points,scores,masks):
    good=np.isfinite(points).all(-1)&(scores>=.5);out=[]
    if masks['trunk_tilt']:out.extend([(11,12),(23,24),(11,23),(12,24)])
    if masks['lead_knee']:out.extend([(23,25),(25,27)])
    if masks['throwing_elbow']:out.extend([(12,14),(14,16)])
    out.extend((a,b) for a,b in [(24,26),(26,28)] if good[a] and good[b])
    return out


def measure(pair):
    f=pair['features']['lead_knee_change']
    if f['nominal'] is None:return '관측 부족'
    return f"{f['nominal']:+.2f}°  (기준 프레임 ±1: {f['range'][0]:+.2f} ~ {f['range'][1]:+.2f}°)"


def render(lab,folder,font):
    result=read(folder/'results.private.json');pred=predictions(result);pairs=selected_pairs(result)
    tracks=lab/'motion-outcome-replication-20261113/artifacts/motion-outcome-replication-001/mapped-tracks.private.npz'
    z=np.load(tracks,allow_pickle=False);fonts={n:ImageFont.truetype(str(font),n) for n in (20,22,24,26,30,32)}
    bg=(18,26,32);fg=(236,240,241);muted=(174,190,197)
    target=folder/'matched-knee-original-overlay.mp4';master=folder/'matched-knee-master.mp4'
    writer=cv2.VideoWriter(str(master),cv2.VideoWriter_fourcc(*'mp4v'),30,SIZE)
    if not writer.isOpened():raise RuntimeError('Video writer failed')
    total=0;shots=[]
    for scene,pair in enumerate(pairs):
        keys=pair['keys'];clips=[result['source_clips'][k] for k in keys]
        relative=np.arange(-233.333,100.001,16.66667);indices=[];images=[];scales=[]
        for key,clip in zip(keys,clips):
            dt=z[key+'_times']-z[key+'_times'][clip['first_index']]
            index=np.abs(dt[:,None]-relative[None,:]).argmin(axis=0)
            if np.max(abs(dt[index]-relative))>17:raise ValueError('Video phase outside source support')
            indices.append(index);images.append(source_frames(clip['video'],z[key+'_frames'][index]))
            pts=z[key+'_points'];valid=z[key+'_trunk_tilt_mask']
            lengths=np.linalg.norm(pts[:,[11,12]].mean(1)-pts[:,[23,24]].mean(1),axis=1)
            scales.append(float(np.median(lengths[valid])) if valid.any() else None)
        for tick,_ in enumerate(relative):
            im=Image.new('RGB',SIZE,bg);draw=ImageDraw.Draw(im)
            draw.text((24,12),f"조건을 맞춘 스킨스 포심 | {pair['date']}",font=fonts[32],fill=fg)
            a,b=pair['context'];group='느린 공 / 빠른 공' if pair['cohort']=='speed_contrast' else '낮은 RPM / 높은 RPM'
            draw.text((24,54),f"{group} · 같은 타자 / {a['inning']}회 / {a['outs_when_up']}아웃 / 주자 {a['bases']} · 볼카운트 {a['balls']}-{a['strikes']} → {b['balls']}-{b['strikes']}",font=fonts[22],fill=muted)
            for side,(key,clip,color) in enumerate(zip(keys,clips,COLORS)):
                x=24+568*side;row=int(indices[side][tick]);frame=int(z[key+'_frames'][row]);record=clip['record']
                label=('느린 공','빠른 공')[side] if pair['cohort']=='speed_contrast' else ('낮은 RPM','높은 RPM')[side]
                draw.text((x,94),f"{label}  {record['speed_kmh']:.2f} km/h · {record['rpm']:.0f} RPM",font=fonts[24],fill=color)
                draw.text((x,129),f"좌우 {record['horizontal_cm']:+.2f} cm · 수직 {record['vertical_cm']:.2f} cm",font=fonts[22],fill=fg)
                raw=images[side][frame];h,w=raw.shape[:2];crop=(int(.28*w),int(.28*h),int(.72*w),int(.98*h))
                cw,ch=crop[2]-crop[0],crop[3]-crop[1];scale=min(544/cw,510/ch);sw,sh=round(cw*scale),round(ch*scale)
                dx=x+(544-sw)//2;dy=171+(510-sh)//2
                im.paste(Image.fromarray(raw).crop(crop).resize((sw,sh),Image.Resampling.BILINEAR),(dx,dy))
                points=z[key+'_points'][row];scores=z[key+'_scores'][row]
                masks={n:bool(z[key+'_'+n+'_mask'][row]) for n in ('trunk_tilt','lead_knee','throwing_elbow')}
                for j,k in edges(points,scores,masks):
                    aa=(dx+(points[j,0]-crop[0])*scale,dy+(points[j,1]-crop[1])*scale)
                    bb=(dx+(points[k,0]-crop[0])*scale,dy+(points[k,1]-crop[1])*scale)
                    draw.line([aa,bb],fill=bg,width=9);draw.line([aa,bb],fill=color,width=5 if (j,k) in ((23,25),(25,27)) else 3)
                    if masks['trunk_tilt'] and scales[side]:
                        hip=points[[23,24]].mean(0);scale2=110/scales[side]
                        pa=tuple(np.array([1360,450])+(points[j]-hip)*scale2);pb=tuple(np.array([1360,450])+(points[k]-hip)*scale2)
                        draw.line([pa,pb],fill=color,width=4)
                t=z[key+'_times'][row]-z[key+'_times'][clip['first_index']]
                draw.text((x,690),f"원본 F{frame} · 첫 자유 공 대비 {t:+.1f} ms",font=fonts[22],fill=fg)
            draw.text((1180,100),'앞무릎 동작 비교',font=fonts[30],fill=fg)
            draw.text((1180,145),'골반·몸통 길이 정규화',font=fonts[20],fill=muted)
            draw.text((1180,180),'2D 표시 · 3D 교정량 아님',font=fonts[20],fill=muted)
            draw.text((1180,685),'관측하지 못한 팔은 생략',font=fonts[20],fill=muted)
            d=pair['delta'];draw.line((24,734,1576,734),fill=(74,90,99),width=1)
            draw.text((24,749),f"오른쪽 - 왼쪽 | 구속 {d['speed_kmh']:+.2f} km/h · RPM {d['rpm']:+.0f} · 좌우 {d['horizontal_cm']:+.2f} cm · 수직 {d['vertical_cm']:+.2f} cm",font=fonts[26],fill=fg)
            draw.text((24,794),'앞무릎 구간 변화 차이: '+measure(pair),font=fonts[24],fill=fg)
            if pair['id'] in pred:
                p=pred[pair['id']];e=p['estimates']
                text=f"이 경기 제외 학습 | 좌우 차이 예측 M0 {e['m0']:+.2f} cm → +무릎 M1 {e['m1']:+.2f} cm  / 실제 {p['observed']:+.2f} cm"
            else:text='다른 경기 예측: 자료 부족으로 계산하지 않음'
            draw.text((24,835),text,font=fonts[24],fill=fg)
            draw.text((24,878),'M0: 구속·RPM·볼카운트  |  무릎: -150~-100ms → -83~-33ms의 화면각 변화',font=fonts[22],fill=muted)
            draw.text((24,916),'쌍 내부 시야 유사성 확인 · 경기 간 시야 동등성 미확인 · 반복 검증 미통과',font=fonts[22],fill=(255,193,96))
            draw.text((24,957),'실제 두 투구의 관찰 비교입니다. 무릎을 바꾸면 공이 이만큼 움직인다는 시뮬레이션이 아닙니다.',font=fonts[22],fill=muted)
            if tick==9:
                name=f'matched-scene-{scene+1}.jpg';im.save(folder/name,quality=93);shots.append(name)
            bgr=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
            for _ in range(7):writer.write(bgr);total+=1
    writer.release()
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe,'-y','-v','error','-i',str(master),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    cap=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if image.shape[:2]!=(SIZE[1],SIZE[0]):raise ValueError('Wrong output size')
        decoded+=1
    cap.release()
    if decoded!=total:raise ValueError('Incomplete video decode')
    subprocess.run([exe,'-v','error','-i',str(target),'-f','null','-'],check=True)
    receipt={'file':target.name,'pairs':[p['id'] for p in pairs],'frames':decoded,'fps':30,'seconds':decoded/30,
             'size':SIZE,'snapshots':shots,'full_opencv_decode':True,'full_ffmpeg_decode':True}
    (folder/'video-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lab',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--font',type=Path,default=Path(os.environ.get('WINDIR','/Windows'))/'Fonts/malgun.ttf')
    args=parser.parse_args();render(args.lab.resolve(),args.out,args.font)
