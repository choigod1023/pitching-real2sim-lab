import hashlib,itertools,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import mujoco
import numpy as np
from compliant_timing import rollout,budget
from coupled_payload import model_for

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz'
OUT=ROOT/'artifacts/fine-timing'


def geometry(q,bq,lengths,dt):
    m=model_for(lengths,dt,ball_mass=0.);dq=np.zeros(11);d=mujoco.MjData(m);e=mujoco.MjData(m)
    rotations=[];wrists=[]
    for qa,qb in zip(q,bq):
        mujoco.mj_differentiatePos(m,dq,1.,qb,qa)
        rotations.append(max(np.linalg.norm(dq[3:6]),abs(dq[6]),abs(dq[7]),np.linalg.norm(dq[8:11])))
        d.qpos[:]=qa;e.qpos[:]=qb;mujoco.mj_forward(m,d);mujoco.mj_forward(m,e)
        wrists.append(np.linalg.norm(d.site_xpos[m.site('wrist').id]-e.site_xpos[m.site('wrist').id]))
    return float(np.rad2deg(max(rotations))),float(max(wrists)*1000)


def evaluate(job):
    shifts,sub=job;src=np.load(SOURCE);r,q,v,b,iq,iv=rollout(src,shifts,sub)
    base=json.loads((OUT/f'baseline{sub}.json').read_text());bq=np.load(OUT/f'baseline{sub}.npz')['qpos']
    angle,position=geometry(q,bq,src['lengths'],float(src['dt']))
    r.update(gain_kmh=r['speed_kmh']-base['speed_kmh'],violations=budget(base,r),max_rotation_deg=angle,max_wrist_delta_mm=position,small_motion=angle<=3 and position<=10)
    return r


def save_state(name,shifts,sub):
    src=np.load(SOURCE);r,q,v,b,iq,iv=rollout(src,shifts,sub)
    np.savez_compressed(OUT/f'{name}.npz',qpos=q,qvel=v,ball=b,initial_q=iq,initial_v=iv,dt=src['dt'],lengths=src['lengths'])
    return r


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    for sub in (1,2,4):
        r=save_state(f'baseline{sub}',(0,0,0),sub)
        (OUT/f'baseline{sub}.json').write_text(json.dumps(r),encoding='utf8')
    grid=[(0,0,0)]+list(itertools.product([-.7,-.6,-.5,-.4,-.3],[.3,.4,.5,.6,.7],[-.1,-.05,0,.05,.1]))+[(-.5*s,.5*s,0) for s in [.1,.25,.5,.75]]
    jobs=[(s,sub) for sub in (1,2) for s in grid]
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[]
        for i,r in enumerate(pool.map(evaluate,jobs)):
            rows.append(r)
            if (i+1)%30==0:print('completed',i+1,'/',len(jobs),flush=True)
    n=len(grid);coarse=rows[:n];fine=rows[n:];baseline=fine[0];original=next(r for r in fine if r['shifts_ms']==[-.5,.5,0]);accepted=[]
    for a,b in zip(coarse,fine):
        b['resolution_bound_kmh']=abs(a['speed_kmh']-b['speed_kmh'])+abs(coarse[0]['speed_kmh']-baseline['speed_kmh'])
        b['accepted']=not a['violations'] and not b['violations'] and a['small_motion'] and b['small_motion'] and b['gain_kmh']>b['resolution_bound_kmh']
        if b['accepted']:accepted.append(b)
    diagnostic=[r for r in fine[1:] if r['small_motion'] and r['gain_kmh']>r['resolution_bound_kmh']]
    def score(r):
        return max(r['violations'].get('work_positive_j',0)/original['violations']['work_positive_j'],r['violations'].get('coupling_peak_force_n',0)/original['violations']['coupling_peak_force_n'])
    selected=max(accepted,key=lambda r:r['gain_kmh']) if accepted else min(diagnostic,key=score)
    save_state('original',[-.5,.5,0],4);sr=save_state('selected',selected['shifts_ms'],4)
    b4=json.loads((OUT/'baseline4.json').read_text());sr.update(gain_kmh=sr['speed_kmh']-b4['speed_kmh'],violations=budget(b4,sr))
    sr['max_rotation_deg'],sr['max_wrist_delta_mm']=geometry(np.load(OUT/'selected.npz')['qpos'],np.load(OUT/'baseline4.npz')['qpos'],np.load(SOURCE)['lengths'],float(np.load(SOURCE)['dt']))
    result=dict(condition_count=n,accepted_count=len(accepted),selected=selected,selected_sub4=sr,baseline_sub4=b4,original=original,rows=rows,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'FINE_TIMING_PROTOCOL.md').read_bytes()).hexdigest())
    (ROOT/'fine_timing_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':run()
