"""Pitching-scene video with measured poses and explicit conditional assumptions."""
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .geometry import bat_state,ball_positions
from .scene import JOINTS,EDGES,HITTER_JOINTS,HITTER_EDGES,interpolate
from ..temporal_quality.model import digest

SIZE=(1600,1000)
FPS=30
COLORS=('#66c9ff','#ff80b5')


def project(points, mode):
    p=np.asarray(points,float)
    if mode=='field':return np.stack([180+68*p[...,0]+16*p[...,1],410-65*p[...,2]+12*p[...,1]],axis=-1)
    if mode=='pitcher':return np.stack([270+135*(p[...,0]-18.44)+32*p[...,1],866-155*p[...,2]+15*p[...,1]],axis=-1)
    if mode=='hitter':return np.stack([820+135*p[...,0]+55*p[...,1],866-155*p[...,2]+15*p[...,1]],axis=-1)
    if mode=='contact':return np.stack([1290+660*p[...,0]+80*p[...,1],748-660*p[...,2]],axis=-1)
    raise ValueError('unknown_view')


def draw_skeleton(draw,xyz,names,edges,mode,color,width=4):
    points=project(xyz,mode)
    for a,b in edges:
        aa,bb=points[names.index(a)],points[names.index(b)]
        draw.line([tuple(aa),tuple(bb)],fill=color,width=width)
    for x,y in points:draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    # An unlabeled head circle is a drawing aid, not a measured head center.
    shoulders=[names.index(n) for n in (('shoulder_jc','glove_shoulder_jc') if mode=='pitcher' or names==JOINTS else ('lsjc','rsjc'))]
    neck=points[shoulders].mean(axis=0)
    radius=9 if mode=='field' else 20
    draw.ellipse((neck[0]-radius,neck[1]-2*radius,neck[0]+radius,neck[1]),outline=color,width=width)


def frame(scene,elapsed,geometry,fonts,slowdown=10,replay=False):
    image=Image.new('RGB',SIZE,'#101b29');d=ImageDraw.Draw(image)
    d.text((24,12),'투수–공–타자 · 실측 두 기록을 배치한 합성 대결',font=fonts[30],fill='white')
    d.text((24,55),f'투수 830 / 1615_1   |   타자 {scene["hitter_id"]} / {scene["swing_id"]}   |   같은 실제 대결이 아님',font=fonts[22],fill='#b9c9da')
    d.text((24,87),f'공 초기속력 {scene["speed_kmh"]:.2f} km/h: 기록값 입력 가정   ·   타자 기준 / +12 ms 겹침',font=fonts[23],fill='white')
    role='접촉 구간 다시보기' if replay else '전체 투구'
    d.text((24,120),f'{role}  ·  이탈 기준 {elapsed*1000:+.0f} ms   |   1/{slowdown}배 재생   |   공 반작용·타구 비행 미계산',font=fonts[20],fill='#b9c9da')
    for x0,x1,title in [(18,520,'투수 · 원 실측 자세 재생'),(540,1050,'타자 · 같은 스윙의 시간만 이동'),(1070,1582,'스윗스폿 구간 확대')]:
        d.rounded_rectangle((x0,477,x1,916),radius=12,outline='#34516a',width=2)
        d.text((x0+12,489),title,font=fonts[20],fill='white')
    d.line([(65,410),(1535,410)],fill='#34516a',width=3)
    d.text((170,425),'홈 쪽',font=fonts[20],fill='#b9c9da');d.text((1410,425),'투수 쪽',font=fonts[20],fill='#b9c9da')
    plate=project(np.array([[-.2,-.2,0],[.2,-.2,0],[.2,.2,0],[-.2,.2,0]]),'field')
    d.polygon([tuple(p) for p in plate],outline='#b9c9da')
    pitch_query=np.clip(scene['br']+elapsed,scene['pitch_time'][0],scene['pitch_time'][-1])
    pxyz=interpolate(scene['pitch_time'],scene['pitch_xyz'],pitch_query)
    for view in ('field','pitcher'):draw_skeleton(d,pxyz,JOINTS,EDGES,view,'#ffc36b',4)
    if scene['br']+elapsed>scene['pitch_time'][-1]:
        d.text((30,882),'기록 종료 후 마지막 자세 유지',font=fonts[17],fill='#b9c9da')
    bat_states=[]
    for delay,color in zip((0,12),COLORS):
        query=scene['bat']['contact']+elapsed-scene['arrival']-delay/1000
        hxyz=interpolate(scene['hitter_time'],scene['hitter_xyz'],query)
        grip,sweet,axis=bat_state(scene['bat'],elapsed,scene['arrival'],delay);bat_states.append((grip,sweet,axis))
        for view in ('field','hitter'):
            draw_skeleton(d,hxyz,HITTER_JOINTS,HITTER_EDGES,view,color,3)
            # Grip-to-sweet extension is a schematic bat, not measured full-bat endpoints.
            p=project(np.stack([grip,sweet+.10*axis]),view)
            d.line([tuple(p[0]),tuple(p[1])],fill=color,width=7)
    anchor=interpolate(scene['bat']['time'],scene['bat']['sweet'],scene['bat']['contact'])
    for state,color in zip(bat_states,COLORS):
        _,sweet,axis=state;p=project(np.stack([sweet-.05*axis,sweet+.05*axis])-anchor,'contact')
        visible,a,b=cv2.clipLine((1090,530,470,370),tuple(np.rint(p[0]).astype(int)),tuple(np.rint(p[1]).astype(int)))
        if visible:
            radius=round(.033*660)
            d.line([a,b],fill=color,width=2*radius)
            for x,y in p:
                if 1112<x<1538 and 552<y<878:d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
    if elapsed>=0:
        ball=ball_positions(scene['release'],scene['velocity'],elapsed)
        for view in ('field','hitter'):
            x,y=project(ball,view);radius=5 if view=='field' else 7
            if view=='field' or (540<x<1050 and 530<y<900):d.ellipse((x-radius,y-radius,x+radius,y+radius),fill='white')
        x,y=project(ball-anchor,'contact')
        radius=round(.0365*660)
        if 1114<x<1536 and 554<y<876:d.ellipse((x-radius,y-radius,x+radius,y+radius),fill='white',outline='#263d4d',width=2)
    g0,s0,a0=bat_states[0];g1,s1,a1=bat_states[1]
    angle=np.rad2deg(np.arccos(np.clip(a0@a1,-1,1)));delta=np.linalg.norm(s1-s0)*1000
    d.text((560,854),f'배트 축 차이 {angle:.1f}°  /  점 위치 차이 {delta:.1f} mm',font=fonts[18],fill='white')
    selected=[c for c in geometry['nominal']['cases'] if c['delay_ms'] in (0,12)]
    labels=[]
    for case in selected:
        near='근접' if case['sweet_zone_near'] else '벗어남'
        labels.append(f'{case["delay_ms"]:+.0f} ms: {near} (축 거리 {case["minimum_clearance_mm"]+69.5:.1f} mm)')
    d.text((24,930),'  |  '.join(labels)+'   ·   전체 배트의 실제 컨택/헛스윙 판정이 아님',font=fonts[20],fill='white')
    d.text((24,963),'투수 수정 0° / 0 mm  ·  힘·일 미평가  ·  실제 구위 향상 미검증  ·  OBP / CC BY-NC-SA 4.0',font=fonts[18],fill='#b9c9da')
    return image


def validate_video(path,frames_dir=None):
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.mp4':raise ValueError('missing_pitching_video')
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],capture_output=True)
    if result.returncode:raise ValueError('full_video_decode_failed')
    cap=cv2.VideoCapture(str(path));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    n=0;selected={0,max(0,expected//2),max(0,expected-1)};saved=[]
    if frames_dir is not None:Path(frames_dir).mkdir(parents=True,exist_ok=True)
    while True:
        ok,img=cap.read()
        if not ok:break
        if n in selected and frames_dir is not None:
            output=Path(frames_dir)/f'frame_{n:04d}.png';cv2.imwrite(str(output),img);saved.append(str(output.resolve()))
        n+=1
    cap.release()
    if not n or n!=expected or fps<=0:raise ValueError('frame_count_mismatch')
    return dict(path=str(path.resolve()),sha256=digest(path),decoded_frames=n,fps=fps,representative_frames=saved)


def export(scene,geometry,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fonts={s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s) for s in (17,18,20,22,23,30)}
    begin=max(-.45,scene['arrival']-scene['bat']['contact']+.012)
    end=min(scene['arrival']+.05,scene['arrival']+scene['hitter_time'][-1]-scene['bat']['contact'])
    times=np.linspace(begin,end,int(np.ceil((end-begin)*10*FPS))+1)
    replay_times=np.linspace(max(begin,scene['arrival']-.05),end,int(np.ceil((end-max(begin,scene['arrival']-.05))*30*FPS))+1)
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{SIZE[0]}x{SIZE[1]}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for t in times:proc.stdin.write(frame(scene,float(t),geometry,fonts).tobytes())
        for t in replay_times:proc.stdin.write(frame(scene,float(t),geometry,fonts,30,True).tobytes())
        proc.stdin.close();error=proc.stderr.read();proc.stderr.close();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    except BaseException:
        proc.kill();proc.wait()
        if not proc.stdin.closed:proc.stdin.close()
        if not proc.stderr.closed:proc.stderr.close()
        raise
    info=validate_video(path,path.parent/'frames')
    contact_index=len(times)+int(np.argmin(abs(replay_times-scene['arrival'])))
    capture=cv2.VideoCapture(str(path));capture.set(cv2.CAP_PROP_POS_FRAMES,contact_index)
    ok,pixels=capture.read();capture.release()
    if not ok:raise ValueError('contact_representative_frame_missing')
    contact_path=path.parent/'frames'/'contact.png';cv2.imwrite(str(contact_path),pixels)
    info['representative_frames'].append(str(contact_path.resolve()))
    info.update(scope='separate_observed_pose_records_with_conditional_ball_sweet_zone_demo',
                contains_pitcher_scene=True,pitcher_pose_modified=False,actual_athlete_improvement=False,
                full_pitch_frames=len(times),contact_replay_frames=len(replay_times))
    return info


def success_bundle(success,kind,videos,*,athlete_evidence=None):
    if kind not in ('engineering','prediction','athlete_candidate'):raise ValueError('unknown_success_scope')
    if success and not videos:raise ValueError('success_requires_pitching_video')
    if success and kind=='athlete_candidate':
        raise ValueError('athlete_candidate_gate_not_implemented')
    if success and any(v.get('contains_pitcher_scene') is not True for v in videos):
        raise ValueError('success_requires_pitcher_scene')
    checked=[validate_video(v['path']) for v in videos]
    return dict(success=bool(success),success_kind=kind,videos=checked,
        inline_markdown=['![투수–타자 시뮬레이션]('+v['path'].replace('\\','/')+')' for v in checked],
        actual_athlete_improvement=bool(success and kind=='athlete_candidate'))
