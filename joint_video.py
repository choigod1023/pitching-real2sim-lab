"""Display supplied Theia segment origins beside source baseball video.

This is kinematic replay, not a forward-dynamics intervention prediction.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import ezc3d
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/joint-video'
SOURCE = ROOT.parent / 'ball-coordinate-audit/artifacts/video-audit'
UPSTREAM = ROOT.parent / 'openbiomechanics/computer_vision'
EDGES = [('l_thigh','r_thigh'), ('pelvis','torso'), ('torso','head'),
         ('l_uarm','r_uarm'), ('torso','l_uarm'), ('torso','r_uarm')]
for side in ('l','r'):
    EDGES += [(f'{side}_uarm',f'{side}_larm'), (f'{side}_larm',f'{side}_hand'),
              ('pelvis',f'{side}_thigh'), (f'{side}_thigh',f'{side}_shank'),
              (f'{side}_shank',f'{side}_foot'), (f'{side}_foot',f'{side}_toes')]

def load_joints(path):
    c = ezc3d.c3d(str(path))
    a = c['data']['rotations']
    names = [n.replace('_4X4','') for n in c['parameters']['ROTATION']['LABELS']['value']]
    xyz = a[:3,3].transpose(2,1,0)
    if not np.isfinite(xyz).all():
        raise ValueError('Missing segment origins cannot be silently interpolated')
    return xyz, names, float(c['parameters']['ROTATION']['RATE']['value'][0])

def aligned_index(frame, offset, n):
    i = frame + offset
    if not 0 <= i < n:
        raise ValueError('Video/C3D correspondence out of range')
    return i

def elbow_angle(points, side):
    """Interior angle at the selected elbow; 180 degrees means straight.

    Side is anatomical, not screen-left/right. Never silently choose the
    right arm for a left-arm throw. Segment origins are estimated joint points.
    """
    if side not in ('l', 'r'):
        raise ValueError('Explicit anatomical side l or r required')
    a = np.asarray(points[f'{side}_uarm']) - points[f'{side}_larm']
    b = np.asarray(points[f'{side}_hand']) - points[f'{side}_larm']
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    if not np.isfinite(np.r_[a,b]).all() or norm <= 0:
        raise ValueError('Degenerate segment')
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/norm,-1,1))))

def render_combined():
    """Append a clearly separate observed-pitch forward-dynamics example."""
    import imageio_ffmpeg as ff
    from PIL import ImageOps
    physical=ROOT.parent/'baseball-video/artifacts/baseball-video/baseball_3034_1.mp4'
    current=OUT/'baseball_joints_side_by_side.mp4'
    writer=ff.write_frames(str(OUT/'baseball_joints_and_dynamics.mp4'),(1280,800),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
    writer.send(None); count=0
    r=ff.read_frames(str(current),pix_fmt='rgb24');next(r)
    for raw in r:
        writer.send(raw);count+=1
    r.close()
    F=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',30)
    S=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21)
    card=Image.new('RGB',(1280,800),'#101820');d=ImageDraw.Draw(card)
    for y,line in [(260,'다음은 별도 투구 3034_1'),(340,'힘 기반 팔·손·공 시뮬레이션'),(420,'앞 영상과 다른 투구입니다.')]:
        d.text((140,y),line,font=F,fill='white')
    for _ in range(60):writer.send(np.asarray(card));count+=1
    r=ff.read_frames(str(physical),pix_fmt='rgb24');meta=next(r);w,h=meta['size'];physical_frames=0;emitted=0
    for i,raw in enumerate(r):
        src=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(h,w,3))
        pic=ImageOps.contain(src,(1280,700))
        im=Image.new('RGB',(1280,800),'#101820');im.paste(pic,((1280-pic.width)//2,70+(700-pic.height)//2))
        d=ImageDraw.Draw(im)
        d.text((24,12),'별도 투구 3034_1 · 힘 기반 팔·손·공 시뮬레이션',font=F,fill='white')
        d.text((24,774),'앞 영상과 다른 투구 · 외부 어깨 힘·일 및 일반 관성 가정에 의존',font=S,fill='#b0c5d6')
        # Rate conversion preserves source duration; it is display resampling only.
        repeats=int(np.floor((i+1)*30/meta['fps'])-np.floor(i*30/meta['fps']))
        for _ in range(repeats):writer.send(np.asarray(im));count+=1;emitted+=1
        if i==180:im.save(OUT/'preview_physical.png')
        physical_frames+=1
    r.close();writer.close()
    (OUT/'combined_manifest.json').write_text(json.dumps({'frames':count,'fps':30,'duration_s':count/30,'chapters':['same-trial Theia kinematic replay','2-second different-trial notice','3034_1 forward dynamics'],'physical_source_metadata':meta,'physical_frames_decoded':physical_frames,'physical_frames_emitted':emitted,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [current,physical]},'scope':'two different baseball trials; no joint intervention or validated speed improvement'},indent=2),encoding='utf-8')

def main():
    sys.path.insert(0,str(ROOT.parent/'comparison-video/artifacts/video-vendor'))
    import imageio_ffmpeg as ff
    OUT.mkdir(parents=True,exist_ok=True)
    c3d = OUT/'OBP_movement_BaseballThrow_001.c3d'
    xyz,names,rate = load_joints(c3d)
    alignment_path = UPSTREAM/'4d_reconstruction/theia_alignment.json'
    align = json.loads(alignment_path.read_text())
    offset = int(align['c3d_frame_minus_video_frame'])
    assert rate == 360 and offset == 1
    font = lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    F,S,B=font(24),font(19),font(30)
    cal=json.loads((UPSTREAM/'calibration/results/calibration_results.json').read_text())['calibrations']['optitrack_19']
    R=np.array(align['rotation']);t=np.array(align['translation']);scale=align['scale']
    reader=ff.read_frames(str(SOURCE/'camera19.mp4'),pix_fmt='rgb24')
    metadata=next(reader)
    writer=ff.write_frames(str(OUT/'baseball_joints_side_by_side.mp4'),(1280,800),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
    writer.send(None)
    rows=[]; count=0
    for frame,raw in enumerate(reader):
        if frame < 780: continue
        if frame > 1080: break
        idx=aligned_index(frame,offset,len(xyz)); q=xyz[idx];points=dict(zip(names,q))
        im=Image.new('RGB',(1280,800),'#101820');d=ImageDraw.Draw(im)
        d.text((24,16),'야구 투구 · 실제 영상 + 관절 동작 재구성',font=B,fill='white')
        d.text((24,63),'같은 투구 BaseballThrow_001 · 360 Hz 자료 / 1/12 배속',font=S,fill='#b0c5d6')
        pic=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(720,1280,3)).crop((240,0,1120,720)).resize((640,524))
        im.paste(pic,(0,140))
        # Projection is a supplied, fitted display alignment, NOT calibration validation.
        rig=scale*(q@R.T)+t
        uv=np.column_stack((cal['fx_px']*rig[:,0]/rig[:,2]+cal['cx_px'],cal['fy_px']*rig[:,1]/rig[:,2]+cal['cy_px']))
        overlay=dict(zip(names,(uv-[240,0])*[640/880,524/720]+[0,140]))
        # Oblique orthographic view in the original C3D laboratory coordinates.
        horizontal=(q[:,0]*.8-q[:,1]*.6)
        coords=np.column_stack((970+horizontal*.22,640-q[:,2]*.25))
        skel=dict(zip(names,coords))
        d.line((660,640,1260,640),fill='#344959',width=2)
        for a,b in EDGES:
            color='#ffb44b' if (a,b) in [('r_uarm','r_larm'),('r_larm','r_hand'),('l_uarm','l_larm'),('l_larm','l_hand')] else '#67d8d1'
            d.line([tuple(overlay[a]),tuple(overlay[b])],fill=color,width=2)
            d.line([tuple(skel[a]),tuple(skel[b])],fill=color,width=7)
        for name in set(sum(([a,b] for a,b in EDGES),[])):
            x,y=skel[name];d.ellipse((x-6,y-6,x+6,y+6),fill='white')
        d.text((24,110),'실제 영상 · 카메라 19',font=F,fill='white')
        d.text((685,110),'3D 관절 · 팔과 손목 강조',font=F,fill='white')
        angle=elbow_angle(points, side='l')  # Prime BaseballThrow_001 is left-arm throwing.
        d.text((24,676),'관절 동작 재생 · 힘 기반 예측 아님 · 제공된 시간 정렬 사용',font=S,fill='#b0c5d6')
        d.text((24,730),'OpenBiomechanics / Driveline Baseball · CC BY-NC-SA 4.0 + 사용 제한',font=S,fill='#91a5b5')
        writer.send(np.asarray(im));count+=1
        if frame in (820,900,950,980,1040): im.save(OUT/f'preview_{frame}.png')
        rows.append({'video_frame':frame,'c3d_index':idx,'elbow_side':'l','elbow_interior_deg':angle})
    reader.close();writer.close()
    result={'frames':count,'duration_s':count/30,'source_video_metadata':metadata,'c3d_frames':len(xyz),'c3d_rate':rate,'c3d_sha256':hashlib.sha256(c3d.read_bytes()).hexdigest(),'alignment_sha256':hashlib.sha256(alignment_path.read_bytes()).hexdigest(),'alignment_status':'contributor fit; not independently verified','c3d_frame_minus_video_frame':offset,'claim':'kinematic segment-origin replay, no forward dynamics or speed gain','rows':rows}
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    render_combined()

if __name__=='__main__': main()
