"""Rejected observation reconstruction over the actual baseball pitching video."""
import json
import imageio_ffmpeg as ff
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from free_arm_diagnostics import loaded
from free_arm_regions import ROOT,source_points,project,observations,sha


def run():
    source,model,result=loaded();out=ROOT/'artifacts/free-arm-regions';out.mkdir(parents=True,exist_ok=True)
    lab=ROOT.parent
    paths={17:lab/'release-window-validation-20261028/artifacts/release-window/camera17.mp4',
           21:lab/'speed-journey/artifacts/release-observation/camera21.mp4'}
    images={}
    for c,p in paths.items():
        reader=ff.read_frames(str(p),pix_fmt='rgb24');meta=next(reader);images[c]={}
        for f,raw in enumerate(reader):
            if 940<=f<=979:images[c][f]=Image.frombytes('RGB',meta['size'],raw)
            if f>=979:break
        reader.close()
    sequence=[f for f in range(940,980) for _ in range(4)]+[976]*60
    path=out/'baseball_reconstruction_rejected_overlay.mp4'
    writer=ff.write_frames(str(path),(1280,900),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart']);writer.send(None)
    train,test=observations()
    def text(d,xy,t,size=21,color='white'):
        d.text(xy,t,font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size),fill=color)
    for index,f in enumerate(sequence):
        im=Image.new('RGB',(1280,900),'#101923');d=ImageDraw.Draw(im)
        text(d,(18,10),'야구 투구 · 손/팔 관측 재구성 실험 — 검증 실패, 기준 모델 유지',28)
        text(d,(18,51),'청록: 기존 좌표 / 주황: 재구성 좌표 · 선수 자세 수정이나 구속 증가 장면이 아닙니다',21)
        original=source_points(source,f);changed=model.points(f)
        for col,c in enumerate((17,21)):
            raw=images[c][f].copy();rd=ImageDraw.Draw(raw)
            for pts,color in [(original,'#55d7ff'),(changed,'#ffa45c')]:
                uv=project(pts,source.cameras[c]);rd.line([tuple(p) for p in uv],fill=color,width=4)
                for x,y in uv:rd.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            for row in train+test:
                if row['camera']==c and row['frame']==f:rd.rectangle(row['box'],outline='#a0f0b0',width=2)
            im.paste(raw.resize((640,360)),(col*640,90))
            center=project(original[2],source.cameras[c])[0];x,y=map(int,center)
            im.paste(raw.crop((x-160,y-80,x+160,y+80)).resize((640,320)),(col*640,450))
            text(d,(col*640+12,98),f'camera{c} / decoded {f} / 동일 시각 겹침',18)
        shift=np.linalg.norm(changed-original,axis=1)*1000
        angle=np.degrees(Rotation.from_matrix(model.hand([f])[0]@source.hand(f+1).as_matrix().T).magnitude())
        text(d,(18,780),f'재구성 차이: 어깨 {shift[0]:.1f} / 팔꿈치 {shift[1]:.1f} / 손목 {shift[2]:.1f}mm · 손방향 {angle:.1f}° (교정 권장량 아님)',20)
        text(d,(18,810),'광학 속력 MAE 1.324→0.797km/h · 평가 시야 영역밖 거리 2.645→4.226px 악화 · 기각',21,'#ffcc8a')
        text(d,(18,842),'예측 평균속력 101.546→103.074km/h / 기록 100.906km/h · 실제 향상·힘·일·안전 미검증',20)
        text(d,(18,874),'Prime BaseballThrow001 · OpenBiomechanics / Driveline · 제공360Hz, 느린 재생 · 녹색 영역은 표면 관측',16,'#b0c0d0')
        writer.send(np.asarray(im).tobytes())
        if index==180:im.save(out/'reconstruction_preview.png')
    writer.close();reader=ff.read_frames(str(path),pix_fmt='rgb24');meta=next(reader);count=sum(1 for _ in reader);reader.close()
    if count!=len(sequence):raise ValueError('Video decode count mismatch')
    manifest={'video':str(path.resolve()),'sha256':sha(path),'frames':count,'fps':30,'size':list(meta['size']),'all_frames_decoded':True,
              'source_sha256':{str(p):sha(p) for p in paths.values()},'result_sha256':sha(ROOT/'free_arm_results.json'),
              'scientific_stage_passed':False,'adopted':False,'actual_athlete_improvement':False}
    (out/'video_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest))


if __name__=='__main__':run()
