"""Measured full-body overlay, preserving real elapsed time and translations."""
import json, subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from full_body_comparison import run,sample,OUT,ANGLES
from pilot import regular_time,vector
from render_speed_journey import BONES,ARM,imageio_ffmpeg
BONES=BONES+ARM
COLORS=[(76,178,255),(255,160,70)]
FONT='C:/Windows/Fonts/malgun.ttf'
def font(n):return ImageFont.truetype(FONT,n)

def prepare(tables,ids):
    tracks=[]
    for pid in ids:
        g=tables['landmarks'].query('session_pitch == @pid').reset_index(drop=True)
        a=tables['joint_angles'].query('session_pitch == @pid').reset_index(drop=True)
        t=regular_time(g);fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0])
        origin=sample(t,(vector(g,'rear_hip')+vector(g,'lead_hip'))/2,fp);origin[2]=0
        tracks.append(dict(t=t,ta=regular_time(a),fp=fp,br=br,origin=origin,
                           xyz={n:vector(g,n)-origin for n in set(sum(([x,y] for x,y in BONES),[]))},
                           angles={c:a[c].to_numpy() for c in ANGLES}))
    return tracks

def main():
    report,tables=run();ids=[report['slow'],report['fast']];tracks=prepare(tables,ids)
    import pandas as pd
    features=pd.read_csv(OUT/'all_five_pitches.csv').set_index('session_pitch')
    size=(1440,912);fps=30;frames=[];video=OUT/'full_body_overlay.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),size,fps=fps,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    views=[np.array([[1,0,0],[0,0,-1]]),np.array([[0,1,0],[0,0,-1]])]
    metadata=[]
    for anchor in ['fp','br']:
        lo=max(x['t'][0]-x[anchor] for x in tracks);hi=min(x['t'][-1]-x[anchor] for x in tracks)
        start=max(lo,-.35 if anchor=='fp' else -.5);end=min(hi,.2 if anchor=='fp' else .05)
        qs=np.linspace(start,end,240)
        # Fixed scale and view for both pitches and every frame.
        allp=np.concatenate([sample(x['t'],p,qs+x[anchor]) for x in tracks for p in x['xyz'].values()])
        projected=[allp@v.T for v in views];limits=[(p.min(0),p.max(0)) for p in projected]
        for j,q in enumerate(qs):
            im=Image.new('RGB',size,(16,23,35));d=ImageDraw.Draw(im)
            d.text((30,18),'같은 선수 · 빠른 투구와 느린 투구의 전신 비교',font=font(30),fill='white')
            for k,pid in enumerate(ids):d.text((30+680*k,68),f'{"느린" if k==0 else "빠른"} 투구 {pid}  |  실측 {features.loc[pid,"speed_kmh"]:.2f} km/h',font=font(25),fill=COLORS[k])
            label='발 착지 FP10' if anchor=='fp' else '공 이탈 BR'
            d.text((30,110),f'{label} 정렬  |  기준 시각 {q*1000:+.0f} ms  |  실제 시간 간격 유지 · 느리게 재생',font=font(22),fill='white')
            coords=[];angles=[]
            for x in tracks:
                coords.append({n:sample(x['t'],p,q+x[anchor]) for n,p in x['xyz'].items()})
                angles.append({c:float(sample(x['ta'],v,q+x[anchor])) for c,v in x['angles'].items()})
            for vi,v in enumerate(views):
                left=25+vi*690;top=160;w=660;h=470;mn,mx=limits[vi];scale=min((w-60)/max(mx[0]-mn[0],.1),(h-40)/max(mx[1]-mn[1],.1));center=(mn+mx)/2
                def project(p):return tuple((p@v.T-center)*scale+np.array([left+w/2,top+h/2]))
                d.rectangle((left,top,left+w,top+h),outline=(55,67,84),width=2)
                d.text((left+12,top+10),['실험실 X–Z 시점','실험실 Y–Z 시점'][vi],font=font(18),fill=(180,190,205))
                for k,points in enumerate(coords):
                    for a,b in BONES:d.line([project(points[a]),project(points[b])],fill=COLORS[k],width=4)
                    for p in points.values():
                        xx,yy=project(p);d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=COLORS[k])
                # Wrist displacement at matching elapsed time (after fixed XY translation).
                a=project(coords[0]['wrist_jc']);b=project(coords[1]['wrist_jc']);d.line([a,b],fill='white',width=2)
            values=[]
            for c,label in [('lead_knee_angle_x','앞무릎 X'),('pelvis_angle_z','골반 Z'),('torso_angle_z','몸통 Z')]:
                values.append(f'{label}: {angles[0][c]:.1f}° → {angles[1][c]:.1f}° (차이 {angles[1][c]-angles[0][c]:+.1f}°)')
            d.text((30,650),'  |  '.join(values),font=font(21),fill='white')
            gap=np.linalg.norm(coords[1]['wrist_jc']-coords[0]['wrist_jc'])*100
            d.text((30,688),f'같은 경과시간 손목 위치 차이 {gap:.1f} cm  |  골반의 착지 시 수평 원점만 일치시킴',font=font(21),fill=(210,215,225))
            for k,x in enumerate(tracks):
                now=q+x[anchor];state='착지 전' if now<x['fp'] else ('착지 → 이탈' if now<x['br'] else '이탈 후')
                d.text((30+680*k,730),f'{ids[k]}: {state}  /  BR까지 {(x["br"]-now)*1000:+.0f} ms',font=font(22),fill=COLORS[k])
            d.text((30,782),'관찰 관절좌표 재생 · 실사 영상 아님 · 구속 예측/동작 수정 시뮬레이션 아님',font=font(23),fill='white')
            d.text((30,824),'두 투구 차이 +2.57 km/h는 실측값입니다. 이 자세를 따라 하면 증가한다는 뜻은 아닙니다.',font=font(22),fill=(210,215,225))
            writer.send(np.asarray(im))
            if j==160:im.save(OUT/f'overlay_{anchor}.png')
        metadata.append(dict(anchor=anchor,start=start,end=end,frames=len(qs)))
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],check=True)
    (OUT/'video_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(video)

if __name__=='__main__':main()
