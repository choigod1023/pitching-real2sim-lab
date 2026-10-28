"""Render actual OBP baseball video with exploratory tracking, not a motion intervention."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'comparison-video/artifacts/video-vendor'))
import imageio_ffmpeg as ff
SOURCE=ROOT.parent/'ball-coordinate-audit/artifacts/video-audit'
REPORT=ROOT.parent/'ball-trajectory-fit/artifacts/trajectory-fit/trajectory_fit.json'
OUT=ROOT/'artifacts/pitch-review';OUT.mkdir(parents=True,exist_ok=True)
report=json.loads(REPORT.read_text(encoding='utf-8'))
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
F=font(27);S=font(21);B=font(32)
det={(r['camera'],r['frame']):r for r in report['detections'] if r['mode']=='component' and r['threshold']==(100 if r['camera']==19 else 70)}
writer=ff.write_frames(str(OUT/'baseball_pitch_tracking.mp4'),(1280,800),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
writer.send(None);count=0;saved=[]
def emit(im):
 global count
 writer.send(np.asarray(im));count+=1
 if count in (100,270,380):
  f=OUT/f'preview_{count}.png';im.save(f);saved.append(str(f))
def canvas(title,subtitle):
 im=Image.new('RGB',(1280,800),'#101820');d=ImageDraw.Draw(im)
 d.text((24,12),title,font=B,fill='white');d.text((24,58),subtitle,font=S,fill='#afc5d5')
 return im
readers=[ff.read_frames(str(SOURCE/f'camera{c}.mp4'),pix_fmt='rgb24') for c in (19,22)]
meta=[next(r) for r in readers]
for i,pair in enumerate(zip(*readers)):
 if i<780:continue
 if i>1040:break
 im=canvas('야구 투구 · 실제 영상과 공 추적','360 fps 촬영 / 30 fps 재생 · 1/12 배속')
 d=ImageDraw.Draw(im)
 for j,(cam,raw) in enumerate(zip((19,22),pair)):
  pic=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(720,1280,3)).resize((640,360))
  im.paste(pic,(640*j,110));d.text((24+640*j,480),f'카메라 {cam} · 프레임 {i}',font=S,fill='white')
  r=det.get((cam,i))
  if r and r['pixel']:
   x,y=r['pixel'];x=x/2+640*j;y=y/2+110
   color='#ff6868' if r.get('visual_shoe_overlap') else '#4be2b1'
   d.ellipse((x-9,y-9,x+9,y+9),outline=color,width=2)
 d.text((24,550),'같은 투구를 두 방향에서 관찰',font=F,fill='white')
 d.text((24,603),'녹색: 공 위치 후보   빨강: 신발과 겹침 — 구속 적합에서 제외',font=S,fill='#afc5d5')
 d.text((24,730),'OpenBiomechanics / Driveline Baseball · 임시 보정·탐색적 추적',font=S,fill='#91a5b5')
 emit(im)
for r in readers:r.close()
for frame in range(980,986):
 im=canvas('공이 손에서 나온 직후 · 확대 관찰','프레임 980·981·985 사용 / 982~984는 신발 겹침으로 제외')
 d=ImageDraw.Draw(im)
 for j,cam in enumerate((19,22)):
  src=Image.open(SOURCE/f'camera{cam}_frame{frame}.png')
  cx,cy=(785-20*(frame-978),519+5.6*(frame-978)) if cam==19 else (715+17*(frame-980),298-.8*(frame-980))
  x0,y0=int(cx-65),int(cy-45)
  pic=src.crop((x0,y0,x0+130,y0+90)).resize((624,432))
  im.paste(pic,(8+j*640,110))
  r=det[(cam,frame)]
  if r['pixel']:
   x,y=r['pixel'];x=(x-x0)*4.8+8+j*640;y=(y-y0)*4.8+110
   color='#ff6868' if r.get('visual_shoe_overlap') else '#4be2b1'
   d.ellipse((x-18,y-18,x+18,y+18),outline=color,width=3)
  d.text((24+640*j,560),f'카메라 {cam} · 프레임 {frame}',font=F,fill='white')
 d.text((24,625),'공 중심 후보입니다. 손가락 관절·정확한 이탈 시각은 아직 미확정입니다.',font=S,fill='#afc5d5')
 d.text((24,730),'OpenBiomechanics / Driveline Baseball · 밝기 중심 추적 및 확대 편집',font=S,fill='#91a5b5')
 for _ in range(15):emit(im)
im=canvas('현재 결과 · 구속 증가를 예측한 영상은 아닙니다','실제 공 추적을 안정화하는 단계입니다.')
d=ImageDraw.Draw(im)
for y,text in [(180,'가시성 3프레임 적합: 101.67~102.67 km/h'),(260,'다른 중심 검출 방식: 99.99~100.32 km/h'),(375,'방법 간 차이가 남아 실제 구속 정답으로 채택하지 않았습니다.'),(470,'다음: 공 중심 → 카메라 보정·동기화 → 독립 구속 검증')]:d.text((35,y),text,font=F,fill='white')
d.text((35,680),'이후에 같은 힘 조건에서 관절 동작 수정과 구속 변화를 비교합니다.',font=S,fill='#afc5d5')
for _ in range(60):emit(im)
writer.close()
manifest={'frames':count,'fps':30,'seconds':count/30,'source_frame_interval':[780,1040],'source_fps_assumed':360,'claim':'actual video + exploratory measured-image tracking; no motion intervention or validated speed gain','source_sha256':{str(SOURCE/f'camera{c}.mp4'):hashlib.sha256((SOURCE/f'camera{c}.mp4').read_bytes()).hexdigest() for c in (19,22)},'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'previews':saved}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest))
