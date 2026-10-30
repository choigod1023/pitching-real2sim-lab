"""Render only a verified recipient candidate in its original pitch coordinates."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from scipy.interpolate import CubicSpline

from coupled_payload import model_for
from pilot import regular_time,vector
from trace_error_audit import kinematics
from render_speed_journey import BONES,ARM,imageio_ffmpeg
from coach_hypotheses.budgeted_input_search.render_candidate import project,fit_view
from .study import transfer_pass,independent_check


def select_pair(report,method):
    """Reject source-only, infeasible, zero or unresolved outcome movies."""
    if report['pitch']=='3034_1':raise ValueError('A recipient video is required')
    pair=report['refit']['selected'] if method=='refit' else report['transfers'][method]
    if not transfer_pass(pair,report['baseline'])['pass']:
        raise ValueError('No verified positive recipient candidate')
    return pair


def export(report_path,method,dataset,out):
    """Actual RK4 body overlay; observed legs/trunk remain explicitly replayed."""
    report_path,dataset,out=map(Path,(report_path,dataset,out))
    report=json.loads(report_path.read_text(encoding='utf8'));pair=select_pair(report,method);r=pair['fine']
    paths=[report_path.parent/'baseline-sub2.npz',report_path.parent/f'{method}-sub2.npz']
    states=[np.load(p) for p in paths];b,c=states
    if b['qpos'].shape!=c['qpos'].shape or float(b['dt'])!=float(c['dt']):raise ValueError('Mismatched overlay states')
    independent_check(dict(c),r)
    ids=np.rint(np.linspace(0,len(b['qpos'])-1,120)).astype(int);times=ids*float(b['dt'])
    model=model_for(b['lengths'],float(b['dt']))
    arms=[kinematics(model,s['qpos'][ids],s['qvel'][ids]) for s in states]
    speed=[np.linalg.norm(a[1][:,3],axis=1)*3.6 for a in arms]
    np.testing.assert_allclose(speed[1][-1]-speed[0][-1],r['paired_gain_kmh'],atol=1e-8,rtol=0)
    pitch=report['pitch'];names=sorted(set(sum(([a,b] for a,b in BONES+ARM),[])))
    columns=['session_pitch','time','fp_10_time','BR_time']+[n+'_'+axis for n in names for axis in 'xyz']
    frame=pd.read_csv(dataset/'full_sig/landmarks.csv',usecols=columns)
    g=frame[frame.session_pitch.eq(pitch)].sort_values('time')
    fp,br=float(g.fp_10_time.iloc[0]),float(g.BR_time.iloc[0])
    np.testing.assert_allclose(times[-1],br-fp,atol=1e-8,rtol=0)
    observed={n:CubicSpline(regular_time(g),vector(g,n))(fp+times) for n in names}
    meta=pd.read_csv(dataset/'metadata.csv');rows=meta[meta.session_pitch.eq(pitch)]
    if len(rows)!=1:raise ValueError('Nonunique original radar record')
    recorded=float(rows.pitch_speed_mph.iloc[0])*1.609344
    all_body=np.concatenate(list(observed.values()));all_arm=np.concatenate([a[0].reshape(-1,3) for a in arms])
    views=[fit_view(np.concatenate([all_body,all_arm]),24,145,672,550),fit_view(all_arm,744,145,672,550)]
    out.mkdir(parents=True,exist_ok=True);video=out/f'{pitch}-{method}-overlay.mp4'
    fonts=[ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n) for n in (29,22,17)]
    title,font,small=fonts;colors=['#52c6ff','#ffa54d']
    writer=imageio_ffmpeg.write_frames(str(video),(1440,1024),fps=30,codec='libx264',pix_fmt_out='yuv420p',output_params=['-crf','18','-movflags','+faststart'])
    writer.send(None);sequence=(list(range(120))+[119]*30)*2
    label='이 투구에서 다시 탐색' if method=='refit' else '원래 수정안의 고정 전이'
    for f,i in enumerate(sequence):
        image=Image.new('RGB',(1440,1024),'#101b2b');d=ImageDraw.Draw(image)
        d.text((28,18),f'{pitch} · {label} · 조건부 팔/손/공 적분',font=title,fill='white')
        d.text((28,62),'파랑: 이 투구의 원래 입력   주황: 수정 입력   회색: 관측 몸통·다리 재생',font=font,fill='#c6d1de')
        d.text((28,102),'실사 아님 · 같은 시각·좌표·축척 · 전신 가상 개입이나 실제 선수 효과는 미검증',font=small,fill='#aebdd0')
        for left,(center,scale,origin) in zip((24,744),views):
            d.rectangle((left,145,left+672,695),outline='#34485f',width=2)
            d.text((left+15,158),'전신 맥락' if left==24 else '팔·손 확대',font=font,fill='white')
            if left==24:
                for a,z in BONES:
                    p=project([observed[a][i],observed[z][i]],center,scale,origin)
                    d.line([tuple(x) for x in p],fill='#77879a',width=7)
            for side,(points,_,rotation) in enumerate(arms):
                p=project(points[i],center,scale,origin)
                d.line([tuple(x) for x in p],fill=colors[side],width=9 if side==0 else 5)
                for point in p[:3]:d.ellipse((*tuple(point-5),*tuple(point+5)),fill=colors[side])
                corners=np.array([[x,y,z] for x in (0,.16) for y in (-.04,.04) for z in (-.015,.015)])
                hand=project(points[i,2]+corners@rotation[i].T,center,scale,origin)
                for a in range(8):
                    for z in range(a+1,8):
                        if np.count_nonzero(corners[a]!=corners[z])==1:d.line([tuple(hand[a]),tuple(hand[z])],fill=colors[side],width=2)
                radius=.0364*scale;d.ellipse((*tuple(p[3]-radius),*tuple(p[3]+radius)),outline=colors[side],width=3)
        d.text((28,716),f'모델 BR: {speed[0][-1]:.3f} → {speed[1][-1]:.3f} km/h   변화 {r["paired_gain_kmh"]:+.3f}',font=title,fill='white')
        d.text((28,757),f'현재 {speed[0][i]:.2f} / {speed[1][i]:.2f} km/h · BR까지 {(times[-1]-times[i])*1000:.1f}ms',font=font,fill='#c6d1de')
        d.text((28,797),f'최대 회전 차이 {r["max_rotation_change_deg"]:.3f}° · 공 중심 {r["max_ball_position_change_mm"]:.3f}mm · 방향 {r["release_direction_change_deg"]:.3f}°',font=font,fill='#c6d1de')
        d.text((28,837),'A의 성분별 일·입력 peak·어깨 순일·자세: 두 해상도 통과',font=font,fill='#a5debe')
        e=pair['extra']
        d.text((28,876),f'전체 순일 {e["net_work_change_j"]:+.3f}J · root 힘 벡터 peak {e["root_force_norm_peak_change_n"]:+.5f}N · B 통과 {e["B_gate"]["policy_budget_pass"]}',font=small,fill='#ffcf80')
        d.text((28,914),f'원래 기록 {recorded:.3f} km/h · 기준 모델 차이 {recorded-speed[0][-1]:+.3f} km/h · 수정 투구의 실측 정답 없음',font=small,fill='#ffcf80')
        d.text((28,951),'전체 힘·일 불증가나 훈련 처방 아님 · 효과 95% 구간/근육 능력/구위/제구 미검증',font=small,fill='#b8c9db')
        d.text((28,986),'OpenBiomechanics / Driveline Baseball · CC BY-NC-SA 4.0 · 일반 관성/15cm 고정 그립',font=small,fill='#9daec2')
        writer.send(np.asarray(image))
        if f in (65,119):image.save(out/f'{pitch}-{method}-preview-{f}.png')
    writer.close()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
    manifest={'pitch':pitch,'method':method,'frames':len(sequence),'fps':30,'full_decode_passed':True,
              'gain_kmh':r['paired_gain_kmh'],'observed_body_is_replay':True,'real_athlete_effect_verified':False,
              'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
              'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
              'states_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/f'{pitch}-{method}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    return video


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--method',choices=('normalized','absolute','refit'),required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(export(a.report,a.method,a.dataset,a.out))
