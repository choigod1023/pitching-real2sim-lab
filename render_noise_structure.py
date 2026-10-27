"""Overlay reconstruction improvements without athlete-gain claims."""
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from noise_structure_audit import OUT
from render_coupled_body_search import BONES
from render_speed_journey import imageio_ffmpeg

def main():
    rows=json.loads((OUT/'report.json').read_text())['records']
    rr={(r['noise'],r['structure']):r for r in rows if r['pitch']=='3034_4'}
    b=rr['original','fixed_origin'];n=rr['rounding_bins','fixed_origin'];c=rr['rounding_bins','measured_mobile_origin']
    z=np.load(OUT/'3034_4_overlay.npz');names=list(z['names']);t=z['time'];font=lambda s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s)
    data=[z['original__fixed_origin'],z['rounding_bins__fixed_origin'],z['rounding_bins__measured_mobile_origin']]
    colors=[(100,170,255),(255,190,80),(80,230,150)]
    views=[np.array([[1,0,0],[0,0,-1]]),np.array([[0,1,0],[0,0,-1]])]
    writer=imageio_ffmpeg.write_frames(str(OUT/'noise_structure_overlay.mp4'),(1440,960),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart']);writer.send(None)
    for fi,i in enumerate(np.r_[np.rint(np.linspace(0,len(t)-1,240)).astype(int),np.full(60,len(t)-1)]):
        im=Image.new('RGB',(1440,960),(16,23,35));d=ImageDraw.Draw(im)
        lines=['미분 잡음 vs 어깨 연결 구조 | 3034_4 · 같은 관측 투구 재현',
            f'실측 {b["measured_speed_kmh"]:.2f} km/h | 기존 {b["speed_kmh"]:.2f} → 잡음 처리 {n["speed_kmh"]:.2f} → 구조 수정 {c["speed_kmh"]:.2f}',
            '파랑: 기존 / 노랑: 반올림 잡음 처리 / 초록: 어깨 이동 추가 / 흰색: 관측 팔',
            '3D 재구성 비교입니다. 목표 구속으로 맞추지 않았으며 선수 능력 향상은 아닙니다.']
        for j,s in enumerate(lines):d.text((25,20+43*j),s,font=font(26 if j<2 else 22),fill='white')
        for vi,view in enumerate(views):
            left=25+710*vi;top=205;w=680;h=400
            cloud=np.concatenate([x.reshape(-1,3) for x in data])@view.T;mn,mx=cloud.min(0),cloud.max(0)
            scale=min((w-50)/max(mx[0]-mn[0],.1),(h-40)/max(mx[1]-mn[1],.1));center=(mn+mx)/2
            def project(p):return tuple((p@view.T-center)*scale+[left+w/2,top+h/2])
            d.rectangle((left,top,left+w,top+h),outline=(70,80,95),width=2)
            for x,col in zip(data,colors):
                for a,bb in BONES:d.line([project(x[i,names.index(a)]),project(x[i,names.index(bb)])],fill=col,width=3)
            d.line([project(p) for p in z['observed_arm'][i]],fill='white',width=3)
        ix=names.index('shoulder_jc');shift=np.linalg.norm(data[2][i,ix]-data[0][i,ix])*1000
        lines=[f'FP 후 {(t[i]-t[0])*1000:.1f} ms | 현재 어깨 위치 수정 {shift:.2f} mm / 구조 수정의 상대 관절각 변경 0°',
            f'실측 구속과 차이 {abs(b["signed_speed_error_kmh"]):.2f} → {abs(c["signed_speed_error_kmh"]):.2f} km/h (남은 차이가 큼)',
            f'관측 팔 위치 RMSE {b["arm_rmse_mm"]:.2f} → {c["arm_rmse_mm"]:.2f} mm',
            f'보조 힘 RMS {b["root_force_rms_n"]:.0f} → {c["root_force_rms_n"]:.0f} N / 실제 반력만으로 재현한 결과가 아님',
            f'추가 어깨 이동 입력: 최대 {c["added_origin_peak_force_n"]:.0f} N / 양의 일 {c["added_origin_positive_work_j"]:.2f} J',
            '새 이동 관절은 원인 분리용입니다. 견갑골·근육의 해부학적 검증은 아직 없습니다.',
            '모델 재현 개선 ≠ 선수 구속 향상 / 같은 힘·일 조건의 동작 처방으로 사용하지 않음']
        for j,s in enumerate(lines):d.text((25,625+43*j),s,font=font(22),fill=(255,190,110) if j>=5 else 'white')
        writer.send(np.asarray(im))
        if fi==230:im.save(OUT/'preview.png')
    writer.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(OUT/'noise_structure_overlay.mp4'),'-f','null','-'],check=True)

if __name__=='__main__':main()
