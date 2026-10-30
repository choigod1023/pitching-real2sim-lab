"""Overlay only a strict-budget conditional body/ball simulation candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from scipy.interpolate import CubicSpline

from coupled_payload import model_for
from pilot import regular_time,vector
from trace_error_audit import kinematics
from render_speed_journey import BONES,ARM,imageio_ffmpeg
from coach_hypotheses.gain_feasibility.shield import actual_accounting


def require_candidate(report):
    """A failed optimization or baseline fallback must not create a gain movie."""
    if report.get('baseline_fallback',True) or not report.get('eligible_count',0):
        raise ValueError('No strict-budget candidate to render')
    pair=report['selected']
    for key in ('coarse','fine'):
        r=pair[key]
        if not r['budget_gate'].get('strict_model_budget_pass',False) or not r['pose_screen_pass'] or not r['initial_state_equal'] or r['paired_gain_kmh']<.5:
            raise ValueError('Incomplete strict candidate validation')
    if abs(pair['fine']['paired_gain_kmh']-pair['coarse']['paired_gain_kmh'])>.001:
        raise ValueError('Unresolved simulation candidate')
    return pair['fine']


def project(points,center,scale,origin):
    """Same fixed orthographic axes and scale for original and modified poses."""
    basis=np.array([[.70710678,-.70710678,0.],[.24184476,.24184476,-.93969262]])
    return (np.asarray(points)-center)@basis.T*scale+origin


def fit_view(points,left,top,width,height):
    """Fit the union of all frames; never clip a limb or change overlay scale."""
    projected=project(points,np.zeros(3),1.,np.zeros(2))
    minimum,maximum=projected.min(0),projected.max(0)
    span=np.maximum(maximum-minimum,1e-6)
    scale=float(min((width-90)/span[0],(height-140)/span[1]))
    origin=np.array([left+width/2,top+70+(height-110)/2])-(minimum+maximum)*scale/2
    return np.zeros(3),scale,origin


def export(report_path,state_dir,dataset,out):
    """Render actual RK4 arm states with unchanged observed full-body context."""
    report_path,state_dir,dataset,out=map(Path,(report_path,state_dir,dataset,out))
    report=json.loads(report_path.read_text(encoding='utf-8'));r=require_candidate(report)
    paths=[state_dir/'baseline-sub2.npz',state_dir/'selected-sub2.npz']
    states=[np.load(p) for p in paths]
    b,c=states
    if b['qpos'].shape!=c['qpos'].shape or float(b['dt'])!=float(c['dt']):
        raise ValueError('Overlay requires matched time samples')
    ids=np.rint(np.linspace(0,len(b['qpos'])-1,150)).astype(int)
    dt=float(b['dt']);times=ids*dt
    model=model_for(b['lengths'],dt)
    base_cost=actual_accounting(b['lengths'],b['qpos'],b['qvel'],b['applied_loads'],dt)
    net_change=r['total_net_work_j']-base_cost['total_net_work_j']
    root_norm_change=float(np.linalg.norm(c['applied_loads'][:,:3],axis=1).max()-np.linalg.norm(b['applied_loads'][:,:3],axis=1).max())
    arm=[kinematics(model,s['qpos'][ids],s['qvel'][ids]) for s in states]
    speed=[np.linalg.norm(a[1][:,3],axis=1)*3.6 for a in arm]
    if abs(speed[1][-1]-r['speed_kmh'])>1e-8 or abs(speed[1][-1]-speed[0][-1]-r['paired_gain_kmh'])>1e-8:
        raise ValueError('Displayed release speeds do not match actual rollout')
    names=sorted(set(sum(([a,b] for a,b in BONES+ARM),[])))
    columns=['session_pitch','time','fp_10_time','BR_time']+[n+'_'+axis for n in names for axis in 'xyz']
    frame=pd.read_csv(dataset/'full_sig/landmarks.csv',usecols=columns)
    g=frame[frame.session_pitch.eq('3034_1')].sort_values('time')
    fp,br=float(g.fp_10_time.iloc[0]),float(g.BR_time.iloc[0])
    if abs(times[-1]-(br-fp))>1e-8:
        raise ValueError('Observed body and simulated arm intervals differ')
    observed={n:CubicSpline(regular_time(g),vector(g,n))(fp+times) for n in names}
    meta=pd.read_csv(dataset/'metadata.csv')
    row=meta[meta.session_pitch.eq('3034_1')]
    if len(row)!=1:raise ValueError('Ambiguous observed pitch record')
    recorded=float(row.pitch_speed_mph.iloc[0])*1.609344
    all_body=np.concatenate(list(observed.values()))
    bodycenter=(all_body.max(0)+all_body.min(0))/2
    all_arm=np.concatenate([a[0].reshape(-1,3) for a in arm])
    armcenter=(all_arm.max(0)+all_arm.min(0))/2
    bodyview=fit_view(np.concatenate([all_body,all_arm]),24,145,672,550)
    armview=fit_view(all_arm,744,145,672,550)
    out.mkdir(parents=True,exist_ok=True)
    video=out/'budgeted_pitch_overlay.mp4'
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
    small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
    title=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',29)
    colors=['#52c6ff','#ffa54d']
    writer=imageio_ffmpeg.write_frames(str(video),(1440,1024),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart'])
    writer.send(None)
    sequence=list(range(150))+[149]*60
    for f,i in enumerate(sequence*2):
        image=Image.new('RGB',(1440,1024),'#101b2b');d=ImageDraw.Draw(image)
        d.text((28,18),'3034_1 · 기준/수정 투구 겹침 · 조건부 모델 결과',font=title,fill='white')
        d.text((28,62),'파랑: 원래 입력   주황: 일 예산 아래 재분배한 입력   회색: 같은 투구의 관찰 신체',font=font,fill='#c6d1de')
        d.text((28,102),'실사 아님 · 팔/손/공은 실제 RK4 적분 · 몸통/다리는 관찰 재생',font=small,fill='#aebdd0')
        for left,(center,scale,origin) in [(24,bodyview),(744,armview)]:
            d.rectangle((left,145,left+672,695),outline='#34485f',width=2)
            d.text((left+15,158),'전신 맥락' if left==24 else '팔·손 확대 — 같은 시각·축척',font=font,fill='white')
            if left==24:
                for a,bone in BONES:
                    line=project([observed[a][i],observed[bone][i]],center,scale,origin)
                    d.line([tuple(p) for p in line],fill='#77879a',width=7)
            for side,(points,_,rotation) in enumerate(arm):
                p=project(points[i],center,scale,origin)
                d.line([tuple(x) for x in p],fill=colors[side],width=9 if side==0 else 5)
                for point in p[:3]:
                    d.ellipse((*tuple(point-5),*tuple(point+5)),fill=colors[side])
                corners=np.array([[x,y,z] for x in (0,.16) for y in (-.04,.04) for z in (-.015,.015)])
                hand=points[i,2]+corners@rotation[i].T
                hp=project(hand,center,scale,origin)
                for a in range(8):
                    for bindex in range(a+1,8):
                        if np.count_nonzero(corners[a]!=corners[bindex])==1:
                            d.line([tuple(hp[a]),tuple(hp[bindex])],fill=colors[side],width=2)
                radius=.0364*scale
                d.ellipse((*tuple(p[3]-radius),*tuple(p[3]+radius)),outline=colors[side],width=3)
        d.text((28,716),f'모델 BR: {speed[0][-1]:.3f} → {speed[1][-1]:.3f} km/h   변화 {r["paired_gain_kmh"]:+.3f}',font=title,fill='white')
        d.text((28,756),f'현재 속도 {speed[0][i]:.2f} / {speed[1][i]:.2f} km/h · BR까지 {(times[-1]-times[i])*1000:.1f}ms',font=font,fill='#c6d1de')
        d.text((28,795),f'최대 회전 차이 {r["max_rotation_change_deg"]:.3f}° · 공 중심 차이 {r["max_ball_position_change_mm"]:.3f}mm · 방향 차이 {r["release_direction_change_deg"]:.3f}°',font=font,fill='#c6d1de')
        d.text((28,835),'기존 A의 성분별 일·peak·어깨 순일·자세 검사: 두 해상도 통과',font=font,fill='#a5debe')
        d.text((28,873),f'전체 순일 {net_change:+.3f}J / 어깨 병진힘 벡터 peak {root_norm_change:+.5f}N — 전체 힘·일 불증가 판정은 아님',font=small,fill='#ffcf80')
        d.text((28,911),f'기록 {recorded:.3f} km/h · 기준 모델과 차이 {recorded-speed[0][-1]:.3f} km/h · 실제 효과 95% 구간 미확인',font=small,fill='#ffcf80')
        d.text((28,949),'실제 근력·선수효과·회전·제구: 미검증 / 구속 향상 처방 아님',font=small,fill='#b8c9db')
        d.text((28,986),'OpenBiomechanics / Driveline Baseball · CC BY-NC-SA 4.0 · 일반 손 관성/고정 그립 가정',font=small,fill='#9daec2')
        writer.send(np.asarray(image))
        if f in (90,149):image.save(out/f'preview-{f}.png')
    writer.close()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
    manifest={'frames':len(sequence)*2,'fps':30,'physical_interval_s':float(times[-1]),'full_decode_passed':True,
              'scope':'Strict-budget conditional simulation; real athlete effect, spin and command unverified.',
              'policy_scope':'Legacy A only; total net work and root force vector norm increased, so not a full no-extra-work result.',
              'total_net_work_change_j':net_change,'root_force_norm_change_n':root_norm_change,
              'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
              'state_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'landmarks_sha256':hashlib.sha256((dataset/'full_sig/landmarks.csv').read_bytes()).hexdigest(),
              'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return video


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True);p.add_argument('--state-dir',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(export(a.report,a.state_dir,a.dataset,a.out))
