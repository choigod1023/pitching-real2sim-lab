"""Render original baseball footage and diagnostic 21-point candidates."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .evaluate import BONES,load,selected

SIZE=(1600,1000)
COLORS=((101,206,219),(255,188,103))


def decode_selected(path,indices):
    wanted=set(indices);images={}
    if path.suffix.lower()=='.gif':
        with Image.open(path) as source:
            for f in sorted(wanted):
                source.seek(f);images[f]=source.convert('RGB').copy()
    else:
        cap=cv2.VideoCapture(str(path))
        for f in range(max(wanted)+1):
            ok,raw=cap.read()
            if not ok:raise ValueError(f'Missing original frame {f}')
            if f in wanted:images[f]=Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
        cap.release()
    return images


def draw_hand(draw,xy,origin,scale,color):
    p=np.asarray(xy,float)*scale+np.asarray(origin)
    if p.shape!=(21,2) or not np.isfinite(p).all():return
    for a,b in BONES:draw.line([tuple(p[a]),tuple(p[b])],fill=color,width=2)
    for i,(x,y) in enumerate(p):
        radius=4 if i in (0,8,12) else 2
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)


def shape_projection(world):
    """Display a model-relative view only; never a field/body alignment."""
    p=np.asarray(world,float)-np.asarray(world,float)[0]
    a=np.deg2rad(30);r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    p=p@r.T;xy=p[:,:2];span=max(float(np.ptp(xy,axis=0).max()),.02)
    return (xy-(xy.min(axis=0)+xy.max(axis=0))/2)/span*210+np.array([139,185])


def render_frame(raw,sample,rows,meta,metrics,font):
    im=Image.new('RGB',SIZE,'#141b20');d=ImageDraw.Draw(im)
    def text(x,y,s,size=24,fill='#dce4e7'):
        d.text((x,y),s,font=ImageFont.truetype(str(font),size),fill=fill)
    is_prime=meta['domain'].startswith('prime')
    title='Prime 001 · 같은 투구의 손–몸 연결 검사' if is_prime else ('Rapsodo 기록 연결 고속 투구' if 'rapsodo' in meta['domain'] else '포심 고속 시연 · 기록 미연결')
    text(24,18,title,32);text(24,66,f"{meta['clip_id']} · 원본 0기반 F{sample['frame']} · 원방향과 손 방향 정규화 비교",23)
    text(24,108,'원영상 전체',23);text(816,108,'손 영역 확대 · 새 픽셀 정보가 생기지는 않습니다',23)
    display=raw.copy();display.thumbnail((752,500));pos=(24,152+(500-display.height)//2);im.paste(display,pos)
    scale=display.width/raw.width
    x,y,w,h=sample['roi_xywh'];crop=raw.crop((x,y,x+w,y+h));crop=crop.resize((420,420),Image.Resampling.BICUBIC)
    crop_draw=ImageDraw.Draw(crop)
    d=ImageDraw.Draw(im)
    for mode,color in zip(('raw','rotated'),COLORS):
        hand=selected(rows[mode])
        if hand:
            draw_hand(d,hand['pixel21'],pos,scale,color)
            xy=np.asarray(hand['pixel21'])-[x,y];draw_hand(crop_draw,xy,(0,0),420/w,color)
    im.paste(crop,(816,176))
    d.rectangle((pos[0]+x*scale,pos[1]+y*scale,pos[0]+(x+w)*scale,pos[1]+(y+h)*scale),outline='#a4afb6',width=1)
    text(1260,142,'손 상대 3D',22);text(1260,172,'단안 모델 추정',19)
    candidate=selected(rows['rotated'])
    if candidate:
        draw_hand(d,shape_projection(candidate['world21']),(1244,210),1,COLORS[1])
    else:text(1270,310,'미검출',25,fill='#a7b1b7')
    text(1252,592,'전신 3D 연결 아님',19)
    for i,(mode,name) in enumerate((('raw','원방향'),('rotated','방향 정규화'))):
        hand=selected(rows[mode]);m=metrics['methods'][mode]
        text(24+i*780,680,f"{name}: {'21점 후보' if hand else '미검출'} · 검출 {m['detected_frames']}/{m['target_frames']}프레임",25,fill=COLORS[i])
    record=meta.get('_record')
    if record and all(record[k] is not None for k in ('recorded_speed_kmh','rpm','ivb_cm','hb_cm')):
        text(24,734,f"기록 구속 {record['recorded_speed_kmh']:.2f} km/h · {record['rpm']:.0f} RPM · IVB {record['ivb_cm']:.2f} cm · HB {record['hb_cm']:.2f} cm",26)
    elif is_prime:text(24,734,'목록 구속 100.91 km/h · RPM / 무브먼트 기록 없음',26)
    else:text(24,734,'동일 투구의 구속 · RPM · 무브먼트 기록 없음',26)
    text(24,786,'손 모양 후보 ≠ 공 접촉 측정 · 기록 수치는 신체 모델 예측이 아닙니다',25,fill='#ffbc67')
    text(24,834,'구속·RPM·무브먼트 예측 오차: 산출하지 못함 · 동역학 개입 미실행',23)
    text(24,880,'손끝을 가리는 공과 흐림은 모델만으로 관측 사실이 되지 않습니다',22,fill='#a7b1b7')
    text(24,950,'원본·후보 구분 / 영상 QA · design guidance: unslop by @heydamir, CC BY 4.0',16,fill='#8b979e')
    return im


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--font',type=Path,default=Path(os.environ.get('WINDIR',''))/'Fonts/malgun.ttf')
    args=parser.parse_args();manifest=load(args.out/'input_manifest.private.json')
    inference=load(args.out/'inference.private.json');evaluation=load(args.out/'evaluation.private.json')
    detections={c['clip_id']:c for c in inference['clips']};scores={c['clip_id']:c for c in evaluation['clips']}
    evidence=load(args.out/'evidence.private.json')
    record_lookup={r['clip_id']:r for r in evidence['records']}
    frames=0;snapshots=[];master=args.out/'hands-original-master.mp4';writer=cv2.VideoWriter(str(master),cv2.VideoWriter_fourcc(*'mp4v'),30,SIZE)
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    try:
        ordered=sorted(manifest['clips'],key=lambda c:0 if 'rapsodo' in c['domain'] else (1 if c['domain'].startswith('prime') else 2))
        for meta in ordered:
            clip_id=meta['clip_id'];samples=[s for s in meta['samples'] if s['role']=='target']
            images=decode_selected(Path(meta['path']),[s['frame'] for s in samples])
            mode_rows={mode:{r['frame']:r for r in detections[clip_id]['methods'][mode]} for mode in ('raw','rotated')}
            meta['_record']=record_lookup.get(clip_id)
            for j,sample in enumerate(samples):
                rows={m:mode_rows[m][sample['frame']] for m in mode_rows}
                frame=render_frame(images[sample['frame']],sample,rows,meta,scores[clip_id],args.font)
                if j==len(samples)//2 or sample['frame'] in (3,5,976):
                    name=f"hands-{clip_id}-{sample['frame']}.jpg";frame.save(args.out/name,quality=92);snapshots.append(name)
                bgr=cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
                repeat=3 if meta['domain'].startswith('prime') else 6
                for _ in range(repeat):writer.write(bgr);frames+=1
    finally:writer.release()
    final=args.out/'observed-finger-original-overlay.mp4';ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-v','error','-y','-i',str(master),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True)
    subprocess.run([ffmpeg,'-v','error','-i',str(final),'-f','null','-'],check=True)
    cap=cv2.VideoCapture(str(final));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    if decoded!=frames:raise ValueError(f'Incomplete video: {decoded}/{frames}')
    receipt=dict(file=final.name,frames=frames,seconds=frames/30,full_decode=True,size=SIZE,snapshots=snapshots,body_intervention_executed=False)
    (args.out/'video-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':main()
