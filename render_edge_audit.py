"""Actual baseball footage and observed/reconstructed target overlays. No intervention."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from edge_calibration import OUT,triangulate,project

def font(size):return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size)
def read(cap,frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read()
    if not ok:raise ValueError(f'Cannot decode source frame {frame}')
    return im

def run():
    source=Path('../paired-video-numeric-search-20261028/artifacts/paired-source')
    names=['edgertronic_dynamic_baseballthrow67mph_3.mp4','edgertronic_dynamic_baseballthrow70mph_3.mp4']
    paths=[source/n for n in names];caps=[cv2.VideoCapture(str(p)) for p in paths]
    a=json.loads(Path('edge_target_annotations_plane_round3.json').read_text());r=json.loads((OUT/'published-stereo-heldout-results.json').read_text());cams=[{k:np.array(v) for k,v in c.items()} for c in r['cameras']]
    targets=[];boxes={1:(600,640,1000,990),4:(490,270,780,580)}
    for pos,i in enumerate([1,4]):
        im=cv2.imread(str(OUT/f'edgertronic_static_cube_{i}_f120.png'))
        for name in a['cameras']['1']['fit']:
            uv=[a['cameras'][j]['fit'][name] for j in ['1','4']];x,_,_=triangulate(uv,cams);q=project([x],cams[pos])[0]
            cv2.circle(im,tuple(uv[pos]),4,(0,255,0),1);cv2.drawMarker(im,tuple(np.rint(q).astype(int)),(0,0,255),cv2.MARKER_CROSS,8,1)
        x,y,xx,yy=boxes[i];crop=im[y:yy,x:xx];scale=min(480/crop.shape[1],270/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
        targets.append(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
    path=OUT/'baseball_calibration_audit.mp4';writer=imageio_ffmpeg.write_frames(str(path),(1280,960),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=7,macro_block_size=1);writer.send(None)
    for n in range(180):
        im=Image.new('RGB',(1280,960),'#111923');d=ImageDraw.Draw(im)
        d.text((20,8),'야구 투구 원본과 거리 보정 검증',font=font(27),fill='white')
        d.text((20,44),'자세 수정·구속 향상 결과가 아닙니다. 실제 구속 대응과 3D 보정은 미검증입니다.',font=font(19),fill='#ffc66d')
        for j,cap in enumerate(caps):
            f=350+80*j+n//4;frame=read(cap,f);tile=Image.fromarray(cv2.cvtColor(cv2.resize(frame,(640,512)),cv2.COLOR_BGR2RGB));im.paste(tile,(j*640,80))
            d.rectangle((j*640,80,j*640+640,112),fill='#19242c');d.text((j*640+12,83),f"파일명 {67+3*j}mph / 카메라 3 / 원본 프레임 {f}",font=font(19),fill='white')
            im.paste(targets[j],(j*640+80,610));d.text((j*640+12,592),f'카메라 {[1,4][j]}: 초록=관찰점, 빨강=3D 재투영',font=font(18),fill='#9de7a8')
        d.rectangle((0,885,1280,960),fill='#111923')
        d.text((20,891),'점의 재투영 차이 0.10~1.08 px여도, 698.5 mm 모서리 복원은 649.7~703.7 mm',font=font(22),fill='#ffc66d')
        d.text((20,925),'미세 구속 차이를 평가하기 전에 거리·표식 대응을 검증해야 합니다. 느린 재생이며 물리 이탈 동기화는 미확정입니다.',font=font(17),fill='white')
        if n==108:im.save(OUT/'baseball_calibration_audit_preview.png')
        writer.send(np.asarray(im))
    writer.close()
    for cap in caps:cap.release()
    cap=cv2.VideoCapture(str(path));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame.shape!=(960,1280,3):raise ValueError('Unexpected decoded dimensions')
        count+=1
    cap.release()
    if count!=180:raise ValueError(f'Only {count} frames decoded')
    record={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decoded_frames':count,'fps':30,'size':[1280,960],
        'source_paths':[str(p.resolve()) for p in paths],'source_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        'meaning':'Raw baseball footage + conditional calibration target audit; no simulated intervention, no demonstrated speed-error reduction; display-only alignment'}
    (OUT/'video-verification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
if __name__=='__main__':run()
