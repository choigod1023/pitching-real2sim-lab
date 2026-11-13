"""Original footage plus observed chain values, never a fabricated intervention."""
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

SIZE=(1600,1100)
COLORS=((92,212,224),(255,193,96))


def selected_pairs(result):
    rows=[]
    for cohort in ('speed_contrast','matched_speed_spin'):
        seen=set()
        for p in sorted(result['rows'],key=lambda p:(p['date'],p['id'])):
            if p['cohort']!=cohort or p['game'] in seen or not(p['context_match']['strict'] and p['pair_camera']['pass']):continue
            if p['measurements']['knee'][4] is None:continue
            rows.append(p);seen.add(p['game'])
    return rows


def source_frames(path,indices):
    need=set(map(int,indices));images={};cap=cv2.VideoCapture(str(path))
    for index in range(max(need)+1):
        ok,frame=cap.read()
        if not ok:break
        if index in need:images[index]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cap.release()
    if set(images)!=need:raise ValueError('Missing source frame')
    return images


def edges(points,scores,body,direction):
    out=[];valid=np.isfinite(points).all(-1)&(scores>=.5)
    if body['trunk_tilt']:out.extend([(11,12),(23,24),(11,23),(12,24)])
    if body['lead_knee']:out.extend([(23,25),(25,27)])
    if direction['upperarm']:out.append((12,14))
    if direction['forearm']:out.append((14,16))
    out.extend((a,b) for a,b in ((24,26),(26,28)) if valid[a] and valid[b])
    return out


def observed_text(pair,name,unit):
    value=pair['measurements'][name][4]
    return '관측 부족' if value is None else f'{value:+.2f}{unit}'


def render(lab,folder,font):
    result=read(folder/'results.private.json');pairs=selected_pairs(result)
    z=np.load(lab/'motion-outcome-replication-20261113/artifacts/motion-outcome-replication-001/mapped-tracks.private.npz',allow_pickle=False)
    directions=np.load(folder/'direction-traces.private.npz',allow_pickle=False)
    fonts={n:ImageFont.truetype(str(font),n) for n in (20,22,24,26,30,32)}
    bg=(18,26,32);fg=(236,240,241);muted=(174,190,197)
    target=folder/'knee-release-spin-original-overlay.mp4';master=folder/'comparison-master.mp4'
    writer=cv2.VideoWriter(str(master),cv2.VideoWriter_fourcc(*'mp4v'),30,SIZE)
    if not writer.isOpened():raise RuntimeError('Video writer failed')
    total=0;shots=[]
    for scene,pair in enumerate(pairs):
        keys=pair['keys'];clips=[result['source_clips'][k] for k in keys]
        relative=np.arange(-233.333,100.001,16.66667);indices=[];images=[];scales=[]
        for key,clip in zip(keys,clips):
            dt=z[key+'_times']-z[key+'_times'][clip['first_index']]
            index=np.abs(dt[:,None]-relative[None,:]).argmin(axis=0)
            if np.max(abs(dt[index]-relative))>17:raise ValueError('Unsupported phase replay')
            indices.append(index);images.append(source_frames(clip['video'],z[key+'_frames'][index]))
            p=z[key+'_points'];valid=z[key+'_trunk_tilt_mask'];length=np.linalg.norm(p[:,[11,12]].mean(1)-p[:,[23,24]].mean(1),axis=1)
            scales.append(float(np.median(length[valid])) if valid.any() else None)
        for tick,_ in enumerate(relative):
            im=Image.new('RGB',SIZE,bg);draw=ImageDraw.Draw(im)
            draw.text((24,14),f"스킨스 실제 포심 | 무릎 → 릴리스·팔 방향·회전축 | {pair['date']}",font=fonts[32],fill=fg)
            ctx=pair['context'];draw.text((24,58),f"같은 타자·{ctx[0]['inning']}회·{ctx[0]['outs_when_up']}아웃 · 실제 두 투구 비교 · 신체 개입 시뮬레이션 미실행",font=fonts[22],fill=muted)
            for side,(key,clip,color) in enumerate(zip(keys,clips,COLORS)):
                x=24+568*side;row=int(indices[side][tick]);frame=int(z[key+'_frames'][row]);r=result['records'][key]
                label=('느린 공','빠른 공')[side] if pair['cohort']=='speed_contrast' else ('낮은 RPM','높은 RPM')[side]
                draw.text((x,98),f"{label} {r['speed_kmh']:.2f} km/h · {r['rpm']:.0f} RPM",font=fonts[24],fill=color)
                draw.text((x,132),f"좌우 {r['horizontal_cm']:+.2f}cm · 수직 {r['vertical_cm']:.2f}cm",font=fonts[22],fill=fg)
                raw=images[side][frame];h,w=raw.shape[:2];crop=(int(.28*w),int(.28*h),int(.72*w),int(.98*h))
                cw,ch=crop[2]-crop[0],crop[3]-crop[1];scale=min(544/cw,500/ch);sw,sh=round(cw*scale),round(ch*scale)
                dx=x+(544-sw)//2;dy=173+(500-sh)//2
                im.paste(Image.fromarray(raw).crop(crop).resize((sw,sh),Image.Resampling.BILINEAR),(dx,dy))
                points=z[key+'_points'][row];scores=z[key+'_scores'][row]
                body={n:bool(z[key+'_'+n+'_mask'][row]) for n in ('trunk_tilt','lead_knee')}
                direction={n:bool(directions[key+'_'+n+'_mask'][row]) for n in ('upperarm','forearm')}
                for j,k in edges(points,scores,body,direction):
                    aa=(dx+(points[j,0]-crop[0])*scale,dy+(points[j,1]-crop[1])*scale)
                    bb=(dx+(points[k,0]-crop[0])*scale,dy+(points[k,1]-crop[1])*scale)
                    draw.line([aa,bb],fill=bg,width=8);draw.line([aa,bb],fill=color,width=4)
                    if body['trunk_tilt'] and scales[side]:
                        hip=points[[23,24]].mean(0);s=105/scales[side]
                        draw.line([tuple(np.array([1360,430])+(points[j]-hip)*s),tuple(np.array([1360,430])+(points[k]-hip)*s)],fill=color,width=4)
                dt=z[key+'_times'][row]-z[key+'_times'][clip['first_index']]
                draw.text((x,682),f"원본 F{frame} · 첫 자유 공 대비 {dt:+.1f}ms",font=fonts[22],fill=fg)
                draw.text((x,714),f"기록 릴리스 x {r['release_x']:+.2f} / 높이 {r['release_z']:.2f}cm",font=fonts[22],fill=color)
            draw.text((1180,105),'동작 겹쳐 보기',font=fonts[30],fill=fg)
            draw.text((1180,149),'2D 골반·몸통 정규화',font=fonts[20],fill=muted)
            draw.text((1180,181),'손가락은 원모델 미출력',font=fonts[20],fill=muted)
            a,b=[result['records'][k] for k in keys]
            draw.text((1180,653),f"팔 슬롯 기록 {a['arm_slot']:.1f}° / {b['arm_slot']:.1f}°",font=fonts[22],fill=fg)
            draw.text((1180,688),f"2D 축 기록 {a['spin_axis']:.0f}° / {b['spin_axis']:.0f}°",font=fonts[22],fill=fg)
            draw.text((1180,721),'3D 회전·손 방향과 구분',font=fonts[20],fill=muted)
            draw.line((24,757,1576,757),fill=(74,90,99),width=1)
            text=lambda name,unit:observed_text(pair,name,unit)
            draw.text((24,774),f"오른쪽 - 왼쪽 | 무릎 구간 변화 {text('knee','°')} → 릴리스 x {text('release_x','cm')} / 높이 {text('release_z','cm')}",font=fonts[26],fill=fg)
            draw.text((24,819),f"팔 슬롯 {text('arm_slot','°')} · 전완 화면 방향 {text('forearm','°')} · 2D 회전축 {text('spin_axis','°')}",font=fonts[24],fill=fg)
            d=pair['delta'];draw.text((24,859),f"실제 공 기록 차이 | 구속 {d['speed_kmh']:+.2f} km/h · RPM {d['rpm']:+.0f} · 좌우 {d['horizontal_cm']:+.2f}cm · 수직 {d['vertical_cm']:+.2f}cm",font=fonts[24],fill=fg)
            draw.text((24,907),'화살표는 검증할 연결 가설입니다. 기록 차이는 무릎 수정의 효과가 아닙니다.',font=fonts[22],fill=(255,193,96))
            draw.text((24,945),'첫 자유 공 기준 | 무릎: -150~-100 → -83~-33ms 변화 · 팔: -70~-15ms 방향 · 2D 추정 관절',font=fonts[20],fill=muted)
            draw.text((24,984),'다른 경기·시점 반복 미통과 | 같은 투구의 3D·손 접촉·힘/일 상태 미확보',font=fonts[22],fill=muted)
            if tick==10:
                name=f'chain-scene-{scene+1}.jpg';im.save(folder/name,quality=93);shots.append(name)
            bgr=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
            for _ in range(7):writer.write(bgr);total+=1
    writer.release();exe=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe,'-y','-v','error','-i',str(master),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    cap=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if image.shape[:2]!=(SIZE[1],SIZE[0]):raise ValueError('Wrong video size')
        decoded+=1
    cap.release()
    if decoded!=total:raise ValueError('Incomplete decode')
    subprocess.run([exe,'-v','error','-i',str(target),'-f','null','-'],check=True)
    receipt={'file':target.name,'pairs':[p['id'] for p in pairs],'frames':decoded,'fps':30,'seconds':decoded/30,'size':SIZE,
             'snapshots':shots,'full_opencv_decode':True,'full_ffmpeg_decode':True,'intervention_simulation':False}
    (folder/'video-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lab',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--font',type=Path,default=Path(os.environ.get('WINDIR','/Windows'))/'Fonts/malgun.ttf')
    a=p.parse_args();render(a.lab.resolve(),a.out,a.font)
