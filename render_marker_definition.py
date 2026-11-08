"""Same-pitch anatomical-side diagnostic; no posture or speed intervention."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg as ff
from joint_video import load_joints,elbow_angle
from fixed_alignment_audit import similarity,project


def run():
    root=Path(__file__).resolve().parent;lab=root.parent;out=root/'artifacts/marker-definition'
    xyz,names,rate=load_joints(lab/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d')
    up=lab/'openbiomechanics/computer_vision';align=json.loads((up/'4d_reconstruction/theia_alignment.json').read_text())
    intr=json.loads((up/'calibration/results/calibration_results.json').read_text())['calibrations']
    poses=json.loads((up/'calibration/results/optitrack_rig_provisional.json').read_text())['camera_poses']
    rawroot=lab/'release-window-validation-20261028/artifacts/release-window'
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
    images=[]
    for frame in range(970,987):
        points=dict(zip(names,xyz[frame+1]));sheet=Image.new('RGB',(1280,560),'#101923');d=ImageDraw.Draw(sheet)
        d.text((16,8),'관절점 재점검 | 초록: 투구측 왼팔 / 주황: 이전 코드가 계산한 오른팔',font=font,fill='white')
        d.text((16,45),'같은 실제 투구 · 분절 원점의 투영 · 실측 부착 마커가 아님 · 제공된 정렬 유지',font=small,fill='#d2dae4')
        for j,camera in enumerate([19,20]):
            im=Image.open(rawroot/f'camera{camera}_{frame}.png').convert('RGB');dr=ImageDraw.Draw(im)
            rig=similarity(xyz[frame+1],align);uv=project(rig,poses[str(camera)],intr[f'optitrack_{camera}'])
            for side,color in [('r','#ffae3b'),('l','#49ff87')]:
                q=[uv[names.index(side+'_'+p)] for p in ['uarm','larm','hand']]
                dr.line([tuple(p) for p in q],fill=color,width=5)
                for p in q:dr.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),outline=color,width=3)
            dr.text((20,15),f'camera{camera} / decoded frame{frame}',font=font,fill='white')
            sheet.paste(im.resize((640,360)),(j*640,80))
        right=elbow_angle(points,'r');left=elbow_angle(points,'l')
        d.text((16,450),f'계산 대상 수정: 오른팔 {right:.2f}° → 왼팔 {left:.2f}° (팔꿈치 내각, 펴짐=180°)',font=font,fill='white')
        d.text((16,487),f'왼팔 굽힘=180°−내각={180-left:.2f}° | 실제 자세를 바꾼 결과가 아닙니다.',font=small,fill='#d2dae4')
        d.text((16,522),'손 분절 원점=추정 손목 / 손끝·공 중심과 구분 | 구속·힘·일 계산은 이 수정으로 바뀌지 않음',font=small,fill='#d2dae4')
        images.append(np.asarray(sheet))
        if frame==977:sheet.save(out/'marker_check_preview.png')
    video=out/'baseball_marker_side_check.mp4';writer=ff.write_frames(str(video),(1280,560),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None);count=0
    for _ in range(30):writer.send(images[0]);count+=1
    for _ in range(2):
        for im in images:
            for __ in range(6):writer.send(im);count+=1
    for _ in range(30):writer.send(images[-1]);count+=1
    writer.close();reader=ff.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);decoded=0
    for raw in reader:assert len(raw)==1280*560*3;decoded+=1
    reader.close();assert decoded==count
    report=dict(frames=decoded,fps=30,duration_s=decoded/30,size=[1280,560],
                scope='Anatomical side diagnostic only, no speed gain or new physical model',
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (out/'video_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report));return report


if __name__=='__main__':run()
