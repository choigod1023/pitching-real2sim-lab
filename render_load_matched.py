"""Two-view overlays of actual paired throws. No modified-pose or predicted-gain claims."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

from load_matched_pitching import read_selected
from full_body_comparison import sample
from pilot import regular_time,vector
from render_speed_journey import BONES,ARM,imageio_ffmpeg

EDGES=BONES+ARM
NAMES=sorted({n for edge in EDGES for n in edge})
COLORS=[(72,174,255),(255,170,65)]
VIEWS=[np.array([[1,0,0],[0,0,-1]]),np.array([[0,1,0],[0,0,-1]])]


def make_track(g,angles):
    g=g.sort_values('time');angles=angles.sort_values('time')
    t=regular_time(g);fp=float(g.fp_10_time.iloc[0]);br=float(g.BR_time.iloc[0])
    origin=sample(t,(vector(g,'rear_hip')+vector(g,'lead_hip'))/2,fp);origin[2]=0
    return dict(t=t,ta=regular_time(angles),fp=fp,br=br,
                xyz={n:vector(g,n)-origin for n in NAMES},
                angles={k:angles[k].to_numpy(float) for k in
                        ['lead_knee_angle_x','torso_angle_z','torso_pelvis_angle_z']})


def replay(track,time):
    return {n:sample(track['t'],p,time) for n,p in track['xyz'].items()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    ap.add_argument('--out',type=Path,default=Path('artifacts/load-matched'))
    a=ap.parse_args();report=json.loads((a.out/'analysis.json').read_text(encoding='utf-8'))
    pairs=report['selected_pair_records'];ids={p[k] for p in pairs for k in ('slow','fast')}
    cols=['session_pitch','time','fp_10_time','BR_time',*[n+'_'+axis for n in NAMES for axis in 'xyz']]
    landmarks=read_selected(a.data/'full_sig/landmarks.csv',ids,cols)
    anglecols=['session_pitch','time','lead_knee_angle_x','torso_angle_z','torso_pelvis_angle_z']
    angles=read_selected(a.data/'full_sig/joint_angles.csv',ids,anglecols)
    tracks={pid:make_track(g,angles[angles.session_pitch.eq(pid)]) for pid,g in landmarks.groupby('session_pitch')}
    values=pd.read_csv(a.out/'pitches.csv').set_index('session_pitch')
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    size=(1440,960);fps=24;manifest=[]

    def render(pair,times,limits,label):
        slow,fast=pair['slow'],pair['fast'];ids=[slow,fast];rows=[values.loc[p] for p in ids]
        im=Image.new('RGB',size,(14,22,34));d=ImageDraw.Draw(im)
        d.text((26,16),f'같은 선수 {pair["user"]} | 팔 부담 지표를 제한한 실제 투구 비교',font=font(29),fill='white')
        for k,p in enumerate(ids):
            d.text((26+690*k,64),f'{"느린" if k==0 else "빠른"} {p}  실측 {rows[k].speed_kmh:.2f} km/h',font=font(25),fill=COLORS[k])
        d.text((26,105),label,font=font(21),fill='white')
        positions=[replay(tracks[p],t) for p,t in zip(ids,times)]
        for vi,v in enumerate(VIEWS):
            left,top,w,h=22+vi*700,150,674,432;mn,mx=limits[vi]
            scale=min((w-55)/max(mx[0]-mn[0],.1),(h-40)/max(mx[1]-mn[1],.1));center=(mn+mx)/2
            def point(p):return tuple((p@v.T-center)*scale+[left+w/2,top+h/2])
            d.rectangle((left,top,left+w,top+h),outline=(55,75,95),width=2)
            d.text((left+10,top+8),['X–Z 시점','Y–Z 시점'][vi],font=font(18),fill=(185,200,220))
            for k,p in enumerate(positions):
                for n,m in EDGES:d.line([point(p[n]),point(p[m])],fill=COLORS[k],width=6 if k==0 else 3)
                for xyz in p.values():
                    x,y=point(xyz);r=5 if k==0 else 3;d.ellipse((x-r,y-r,x+r,y+r),fill=COLORS[k])
            d.line([point(positions[0]['wrist_jc']),point(positions[1]['wrist_jc'])],fill='white',width=2)
        live=[]
        for name,label in [('lead_knee_angle_x','앞무릎 X'),('torso_angle_z','몸통 Z'),('torso_pelvis_angle_z','분리각 Z')]:
            z=[float(sample(tracks[p]['ta'],tracks[p]['angles'][name],t)) for p,t in zip(ids,times)]
            live.append(f'{label} {z[0]:.1f}→{z[1]:.1f}° (Δ{z[1]-z[0]:+.1f}°)')
        d.text((26,606),' | '.join(live),font=font(21),fill='white')
        wrist=np.linalg.norm(positions[1]['wrist_jc']-positions[0]['wrist_jc'])*1000
        d.text((26,647),f'손목 위치 차이 {wrist:.1f} mm | 착지 골반 수평 원점만 맞춤 | 실측 구속 차이 +{pair["speed_gain_kmh"]:.2f} km/h',font=font(22),fill='white')
        s,f=rows
        d.text((26,692),f'최대 팔꿈치 {s.elbow_varus_moment:.2f}→{f.elbow_varus_moment:.2f} / 어깨 {s.shoulder_internal_rotation_moment:.2f}→{f.shoulder_internal_rotation_moment:.2f} N·m',font=font(22),fill=(150,220,180))
        d.text((26,737),f'몸통→팔 전달 {s.trunk_to_arm_transfer_j:.2f}→{f.trunk_to_arm_transfer_j:.2f} J | 팔 관절 생성 합 {s.arm_generation_j:.2f}→{f.arm_generation_j:.2f} J',font=font(22),fill='white')
        d.text((26,782),f'앞무릎 생성 {s.lead_knee_generation_j:.2f}→{f.lead_knee_generation_j:.2f} J | 여섯 관절 생성 부분 합 {s.partial_six_joint_generation_j:.2f}→{f.partial_six_joint_generation_j:.2f} J',font=font(22),fill='white')
        d.text((26,840),'관측 관절좌표 재생 | 관절각·위치 차이는 두 실제 투구의 차이 | 수정 동작의 예측 구속이 아닙니다',font=font(22),fill=(255,205,120))
        d.text((26,888),'두 최대 토크 지표 감소는 근육 힘·총일·부상 위험 감소의 증명이 아닙니다.',font=font(22),fill=(205,215,230))
        return im,wrist

    def limits_for(pair,qs):
        points=np.concatenate([sample(tracks[p]['t'],xyz,times) for p,times in qs.items() for xyz in tracks[p]['xyz'].values()])
        if not np.isfinite(points).all():raise ValueError('Nonfinite video trajectory')
        return [(pr.min(axis=0),pr.max(axis=0)) for v in VIEWS for pr in [points@v.T]]

    first=pairs[0];ids=[first['slow'],first['fast']]
    paths=[]
    path=a.out/'observed_load_matched_overlay.mp4';paths.append(path)
    writer=imageio_ffmpeg.write_frames(str(path),size,fps=fps,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for anchor in ('fp','br'):
        start=max(max(tracks[p]['t'][0]-tracks[p][anchor] for p in ids),-.25 if anchor=='fp' else -.4)
        end=min(min(tracks[p]['t'][-1]-tracks[p][anchor] for p in ids),.20 if anchor=='fp' else .03)
        elapsed=np.linspace(start,end,8*fps);qs={p:elapsed+tracks[p][anchor] for p in ids};limits=limits_for(first,qs)
        for i,e in enumerate(elapsed):
            times=[qs[p][i] for p in ids];text=f'{"발 착지 FP10" if anchor=="fp" else "제공 BR"} 정렬 | {e*1000:+.0f} ms | 실제 경과시간 유지 · 느리게 재생'
            im,gap=render(first,times,limits,text);writer.send(np.asarray(im))
            if i==int(np.argmin(abs(elapsed))):im.save(a.out/f'overlay_{anchor}.png')
        manifest.append(dict(file=path.name,anchor=anchor,start_s=start,end_s=end,n_frames=len(elapsed)))
    writer.close()
    path=a.out/'all_30_pairs.mp4';paths.append(path)
    writer=imageio_ffmpeg.write_frames(str(path),size,fps=fps,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','21','-movflags','+faststart']);writer.send(None)
    for number,pair in enumerate(pairs,1):
        ids=[pair['slow'],pair['fast']];phase=np.linspace(0,1,60)
        qs={p:tracks[p]['fp']+phase*(tracks[p]['br']-tracks[p]['fp']) for p in ids};limits=limits_for(pair,qs)
        for i,progress in enumerate(phase):
            times=[qs[p][i] for p in ids]
            im,_=render(pair,times,limits,f'전체 {number}/{len(pairs)}쌍 | FP10→BR 진행률 {progress*100:.0f}% 정렬 | 각 투구의 실제 소요시간은 다릅니다')
            writer.send(np.asarray(im))
        manifest.append(dict(file=path.name,user=pair['user'],slow=pair['slow'],fast=pair['fast'],n_frames=60,phase_aligned=True,
                             slow_duration_ms=(tracks[ids[0]]['br']-tracks[ids[0]]['fp'])*1000,
                             fast_duration_ms=(tracks[ids[1]]['br']-tracks[ids[1]]['fp'])*1000))
    writer.close()
    for path in paths:subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True)
    (a.out/'video_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # Spatial gaps for the fixed example are computed at actual event times, not normalized display time.
    spatial={}
    for event in ('fp','br'):
        s,f=[replay(tracks[p],tracks[p][event]) for p in [first['slow'],first['fast']]]
        spatial[event]={n:float(np.linalg.norm(f[n]-s[n])*1000) for n in ['shoulder_jc','elbow_jc','wrist_jc','lead_ankle_jc']}
    (a.out/'example_spatial_gaps.json').write_text(json.dumps(spatial,indent=2),encoding='utf-8')
    print(json.dumps(spatial));print('\n'.join(str(p.resolve()) for p in paths))


if __name__=='__main__':main()
