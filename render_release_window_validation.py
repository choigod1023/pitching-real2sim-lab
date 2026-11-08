"""Pitching scene overlay of unchanged and timing-only trial models, not intervention."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg as ff
from release_window_validation import interpolate
from joint_video import load_joints,EDGES
from fixed_alignment_audit import project,similarity


def run():
    root=Path(__file__).resolve().parent;out=root/'artifacts/release-window'
    result=json.loads((out/'alignment_results.json').read_text())
    third=json.loads((out/'third_view_results.json').read_text())
    xyz,names,rate=load_joints(Path(result['inputs']['body']['path']))
    intr=json.loads(Path(result['inputs']['calibration']['path']).read_text())['calibrations']
    poses=json.loads(Path(result['inputs']['rig']['path']).read_text())['camera_poses']
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21)
    small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
    edges=[(names.index(a),names.index(b)) for a,b in EDGES]
    output=out/'baseball_alignment_comparison.mp4'
    writer=ff.write_frames(str(output),(1280,848),fps=30,codec='libx264',pix_fmt_in='rgb24',
                          pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
    writer.send(None);images=[]
    for f in range(970,987):
        sheet=Image.new('RGB',(1280,848),'#101923');d=ImageDraw.Draw(sheet)
        d.text((14,7),'실제 투구 위 정렬 비교 | 청록: 기존 +1프레임 / 주황: 시간만 +0.720프레임',font=font,fill='white')
        d.text((14,37),f'디코드 프레임 {f} · 72배 느리게 · 같은 선수/같은 투구 · 시험용, 채택안 아님',font=small,fill='#c9d1dc')
        for j,c in enumerate([17,19,20,21]):
            im=Image.open(out/f'camera{c}_{f}.png').convert('RGB');dr=ImageDraw.Draw(im)
            for model,color in [('M0','#25d3df'),('M1','#ff9b38')]:
                m=result['models'][model]
                points=similarity(interpolate(xyz,np.array([f+m['offset']]))[0],m['alignment'])
                uv=project(points,poses[str(c)],intr[f'optitrack_{c}'])
                for a,b in edges:
                    dr.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=3)
                for name in ['l_larm','l_hand']:
                    u,v=uv[names.index(name)];dr.ellipse((u-5,v-5,u+5,v+5),outline=color,width=3)
            dr.text((20,15),f'camera {c}',fill='white',font=font)
            if c==17:
                detection=next((r for r in third['detections'] if r['frame']==f and r['threshold']==100 and r['status']=='candidate'),None)
                if detection:
                    u,v=detection['pixel'];dr.ellipse((u-9,v-9,u+9,v+9),outline='#ff4f70',width=2)
            sheet.paste(im.resize((640,360)),((j%2)*640,70+(j//2)*360))
        d.text((14,795),'제공 좌표 대비 손 RMSE 15.64→12.35mm | 실제 정확도·구속 향상 입증 아님',font=small,fill='white')
        d.text((14,818),'공 영상 추정 범위 102.29~103.93km/h · 기록 구속 기준 미확인 · 힘/일 미검증 · 신체 수정 없음',font=small,fill='#d2d8e2')
        images.append(np.asarray(sheet))
        if f==977:sheet.save(out/'overlay_preview.png')
    count=0
    for _ in range(30):writer.send(images[0]);count+=1
    for _ in range(2):
        for im in images:
            for __ in range(6):writer.send(im);count+=1
    for _ in range(30):writer.send(images[-1]);count+=1
    writer.close()
    reader=ff.read_frames(str(output),pix_fmt='rgb24');meta=next(reader);decoded=0
    for raw in reader:
        assert len(raw)==1280*848*3;decoded+=1
    reader.close();assert decoded==count
    manifest=dict(frames=decoded,fps=30,duration_s=decoded/30,size=[1280,848],
       scope='Same-pitch kinematic alignment overlay, no forward dynamics, no posture intervention, no measured performance gain.',
       raw_frames=[970,986],cameras=[17,19,20,21],original_offset=1.,candidate_offset=result['models']['M1']['offset'])
    (out/'video_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':run()
