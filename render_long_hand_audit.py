"""Actual pitching scene with rejected calibration and single-frame IK audit."""
import json
from pathlib import Path
import imageio_ffmpeg as ff
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prime17_hand_alignment import Source,sha,project
from long_hand_alignment import ROOT,OUT


def text(d,xy,s,size=20,color='white'):
    d.text(xy,s,font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size),fill=color)


def run():
    source=Source();results=json.loads((ROOT/'long_hand_results.json').read_text(encoding='utf-8'))
    pilot=json.loads((ROOT/'pose_region_pilot_results.json').read_text(encoding='utf-8'))
    lab=ROOT.parent;paths={17:lab/'release-window-validation-20261028/artifacts/release-window/camera17.mp4',
                         20:lab/'speed-journey/artifacts/release-observation/camera20.mp4',
                         21:lab/'speed-journey/artifacts/release-observation/camera21.mp4'}
    images={}
    for c,path in paths.items():
        reader=ff.read_frames(str(path),pix_fmt='rgb24');meta=next(reader);images[c]={}
        for f,raw in enumerate(reader):
            if 940<=f<=979:images[c][f]=Image.frombytes('RGB',meta['size'],raw)
            if f>=979:break
        reader.close()
    sequence=[f for f in range(940,980) for _ in range(4)]+[950]*90
    output=OUT/'long_window_rejected_calibration_audit.mp4'
    writer=ff.write_frames(str(output),(1280,800),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart']);writer.send(None)
    for index,f in enumerate(sequence):
        im=Image.new('RGB',(1280,800),'#101923');d=ImageDraw.Draw(im)
        text(d,(18,10),'야구 투구 · 긴 구간 보정 실패와 관절 자세 재구성 검사',27)
        text(d,(18,48),'청록 기존 / 주황 전역 시간·병진 보정(기각) / 초록 950 단일프레임 길이보존 재구성',19)
        for col,c in enumerate((17,20,21)):
            raw=images[c][f].copy();rd=ImageDraw.Draw(raw)
            for name,color in [('M0','#55d7ff'),('M2','#ff9a51')]:
                m=results['models'][name];pts=np.array([source.origins[s](f+m['offset_frames'])+m['translation_m'] for s in ['l_uarm','l_larm','l_hand']]);uv=project(pts,source.cameras[c])
                rd.line([tuple(p) for p in uv],fill=color,width=3)
                for x,y in uv:rd.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            if f==950 and index>=160:
                pts=np.array([pilot['shoulder_m'],pilot['fitted_elbow_m'],pilot['fitted_wrist_m']]);uv=project(pts,source.cameras[c]);rd.line([tuple(p) for p in uv],fill='#75f2a0',width=3)
            im.paste(raw.resize((426,240)),(col*426,86))
            center=project(source.origins['l_hand'](f+1),source.cameras[c])[0];x,y=map(int,center)
            crop=raw.crop((x-120,y-80,x+120,y+80)).resize((426,284));im.paste(crop,(col*426,326))
            text(d,(col*426+10,94),f'camera{c} · decoded{f}',17)
        b,a=results['models']['M0'],results['models']['M2']
        text(d,(18,624),f'전역 보정: 평가 영역밖 거리 {b["evaluation"]["mean_outside_px"]:.3f}→{a["evaluation"]["mean_outside_px"]:.3f}px / 광학 속력 MAE {b["B_ball_evaluation"]["mean_absolute_speed_discrepancy_kmh"]:.3f}→{a["B_ball_evaluation"]["mean_absolute_speed_discrepancy_kmh"]:.3f}km/h',21)
        fm,hm=pilot['metrics']['fit'],pilot['metrics']['heldout']
        text(d,(18,659),f'950 길이보존 pilot: 적합{fm["before"]["mean_outside_px"]:.3f}→{fm["after"]["mean_outside_px"]:.3f}px / 제외17시야{hm["before"]["mean_outside_px"]:.3f}→{hm["after"]["mean_outside_px"]:.3f}px, 손목은 악화',20)
        text(d,(18,695),'기준 모델 교체 없음 · 선수 자세 수정/구속 향상 아님 · 근육·힘·일 미검증',21,'#ffca8a')
        text(d,(18,735),f'재구성 이동량: 팔꿈치{pilot["joint_shift_mm"]["elbow"]:.1f}mm / 손목{pilot["joint_shift_mm"]["wrist"]:.1f}mm — 선수에게 권할 교정량이 아님',20)
        text(d,(18,773),'Prime BaseballThrow001 · OpenBiomechanics / Driveline · 제공360Hz · 느린 재생/정지',16,'#a9b9c9')
        writer.send(np.asarray(im).tobytes())
        if index==205:im.save(OUT/'audit_preview.png')
    writer.close();reader=ff.read_frames(str(output),pix_fmt='rgb24');metadata=next(reader);count=sum(1 for _ in reader);reader.close();assert count==len(sequence)
    manifest={'video':str(output.resolve()),'sha256':sha(output),'frames':count,'fps':30,'size':list(metadata['size']),'all_frames_decoded':True,
              'sources':{str(p):sha(p) for p in paths.values()},'results_sha256':sha(ROOT/'long_hand_results.json'),
              'pilot_sha256':sha(ROOT/'pose_region_pilot_results.json'),'calibration_adopted':False,'actual_performance_gain':False}
    (OUT/'video_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(str(output))


if __name__=='__main__':run()
