"""Before/after baseline reconstruction, never label calibration as athlete gain."""
import json,subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from scipy.interpolate import CubicSpline
from baseline_calibration import OUT
from coupled_body_search import positions
from full_body_comparison import load
from pilot import vector,regular_time
from render_speed_journey import imageio_ffmpeg
from render_coupled_body_search import BONES
F='C:/Windows/Fonts/malgun.ttf'
def font(n):return ImageFont.truetype(F,n)

def main(before_dir=OUT,after_dir=OUT,output_dir=OUT):
    before_dir,after_dir,output_dir=map(Path,[before_dir,after_dir,output_dir]);output_dir.mkdir(parents=True,exist_ok=True)
    report=json.loads((before_dir/'report.json').read_text());after=json.loads((after_dir/'report.json').read_text());rows={r['mode']:r for r in report['records'] if r['pitch']=='3034_4' and r['hz']==5760};rows['frozen_geometry_mass']=next(r for r in after['records'] if r['pitch']=='3034_4' and r['hz']==5760 and r['mode']=='frozen_geometry_mass');data=[]
    for mode in ['nominal','frozen_geometry_mass']:
        folder=before_dir if mode=='nominal' else after_dir;z=np.load(folder/f'{mode}_states.npz');m=mujoco.MjModel.from_xml_path(str(folder/f'{mode}.xml'));names,p=positions(m,z['qpos']);data.append(p)
    t=z['time'];_,_,tables=load();g=tables['landmarks'].query("session_pitch=='3034_4'").reset_index(drop=True)
    observed={n:CubicSpline(regular_time(g),vector(g,n))(t) for n in names if all(n+'_'+a in g for a in 'xyz')}
    views=[np.array([[1,0,0],[0,0,-1]]),np.array([[0,1,0],[0,0,-1]])];limits=[]
    for view in views:
        p=np.concatenate([x.reshape(-1,3) for x in data])@view.T;limits.append((p.min(0),p.max(0)))
    writer=imageio_ffmpeg.write_frames(str(output_dir/'baseline_calibration_overlay.mp4'),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    b=rows['nominal'];c=rows['frozen_geometry_mass'];frames=np.r_[np.rint(np.linspace(0,len(t)-1,240)).astype(int),np.full(60,len(t)-1)]
    for fi,i in enumerate(frames):
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        d.text((28,18),'기준 투구 재현 보정 | 3034_4 · 동작 변경 없음',font=font(30),fill='white')
        d.text((28,66),f'실측 {b["measured_speed_kmh"]:.2f} km/h | 모델 {b["simulated_speed_kmh"]:.2f} → {c["simulated_speed_kmh"]:.2f} km/h',font=font(27),fill='white')
        d.text((28,110),f'구속 차이 {b["absolute_speed_error_kmh"]:.2f} → {c["absolute_speed_error_kmh"]:.2f} km/h | 아직 검증 미통과',font=font(27),fill=(255,180,100))
        d.text((28,153),'회색: 관찰 / 파랑: 보정 전 / 초록: 첫 투구로 보정 후 다른 투구 재현',font=font(23),fill=(205,215,225))
        for vi,view in enumerate(views):
            left=25+vi*700;top=200;w=675;h=420;mn,mx=limits[vi];center=(mn+mx)/2;scale=min((w-50)/max(mx[0]-mn[0],.1),(h-25)/max(mx[1]-mn[1],.1))
            def proj(p):return tuple((p@view.T-center)*scale+np.array([left+w/2,top+h/2]))
            d.rectangle((left,top,left+w,top+h),outline=(60,70,90),width=2)
            for a,bb in BONES:
                if a in observed and bb in observed:d.line([proj(observed[a][i]),proj(observed[bb][i])],fill=(150,155,165),width=3)
            for p,color in zip(data,[(75,180,255),(80,220,170)]):
                for a,bb in BONES:d.line([proj(p[i,names.index(a)]),proj(p[i,names.index(bb)])],fill=color,width=3)
        idx=names.index('shoulder_jc');e=[np.linalg.norm(x[i,idx]-observed['shoulder_jc'][i])*1000 for x in data]
        d.text((28,638),f'FP 후 {(t[i]-t[0])*1000:.1f} ms | 현재 관찰 어깨와 거리 {e[0]:.1f} → {e[1]:.1f} mm',font=font(24),fill='white')
        d.text((28,685),f'별도 4투구 평균 구속 차이 {report["summary"]["nominal"]["absolute_mae_kmh"]:.2f} → {after["summary"]["frozen_geometry_mass"]["absolute_mae_kmh"]:.2f} km/h (개발 자료)',font=font(24),fill='white')
        d.text((28,732),'보정: 선수 질량 + 첫 투구 신체 치수·어깨 부착 위치 / 목표 구속 사용 안 함',font=font(23),fill=(205,215,225))
        d.text((28,780),'전신 연결 전 팔–손–공 모델에도 큰 차이가 남음 / 원인 추가 분리 필요',font=font(23),fill=(255,180,100))
        d.text((28,828),'이 영상은 모델 보정 비교입니다. 선수 구속 향상이나 새로운 자세 제안이 아닙니다.',font=font(23),fill='white')
        d.text((28,876),f'투구 간 변화 MAE {report["summary"]["nominal"]["change_from_first_pitch_mae_kmh"]:.2f} → {after["summary"]["frozen_geometry_mass"]["change_from_first_pitch_mae_kmh"]:.2f} km/h · 보정안 미채택',font=font(23),fill=(205,215,225))
        writer.send(np.asarray(im))
        if fi==210:im.save(output_dir/'baseline_preview.png')
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(output_dir/'baseline_calibration_overlay.mp4'),'-f','null','-'],check=True)
if __name__=='__main__':main()
