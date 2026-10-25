"""Auxiliary basketball observation ablation; offline kinematics, not pitching dynamics."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter


def frame(wrist,thumb,pinky):
    x=(thumb+pinky)/2-wrist
    if np.min(np.linalg.norm(x,axis=1))<1e-6:raise ValueError('Degenerate hand axis')
    x=x/np.linalg.norm(x,axis=1)[:,None]
    z=np.cross(x,pinky-thumb)
    if np.min(np.linalg.norm(z,axis=1))<1e-6:raise ValueError('Degenerate hand plane')
    z=z/np.linalg.norm(z,axis=1)[:,None];y=np.cross(z,x)
    return np.stack([x,y,z],axis=-1)


def rigid_point(wrist,rotation,local):
    return np.asarray(wrist)+np.asarray(rotation)@np.asarray(local)


def ballistic(past,present,dt,horizon):
    g=np.array([0.,0.,-9.81])
    # Exact present velocity for two noiseless positions under constant gravity.
    velocity=(present-past)/dt+.5*g*dt
    return present+velocity*horizon+.5*g*horizon**2


def load_case(path):
    d=json.loads(Path(path).read_text());tracking=d['tracking'];rate=d['sampling_rate']
    if rate!=60:raise ValueError('Expected prespecified 60Hz session')
    indices=np.array([f['frame'] for f in tracking]);t=indices/rate
    if not np.all(np.diff(indices)==1):raise ValueError('Nonconsecutive frames')
    # JSON timestamps appear rounded to ms; confirm against documented rate, use exact frame clock.
    if np.max(np.abs(np.array([f['time'] for f in tracking])/1000-t))>.0011:raise ValueError('Timestamp/rate mismatch')
    def points(side,name):
        return np.array([f['data']['player'][side+'_'+name] for f in tracking],dtype=float)*.3048
    candidates=[]
    for side in ('LEFT','RIGHT'):
        wrist=points(side,'WRIST')
        if not np.isfinite(wrist).all():raise ValueError('Missing wrist observations')
        vz=savgol_filter(wrist[:,2],7,2,deriv=1,delta=1/rate)
        # Full hand-only interval, excluding 3 edge frames; not claimed true release.
        anchor=int(np.argmax(vz[3:-3]))+3
        candidates.append((float(vz[anchor]),side,anchor,wrist))
    _,side,anchor,wrist=max(candidates,key=lambda x:x[0])
    thumb,pinky=points(side,'THUMB'),points(side,'PINKY')
    ball=np.array([f['data']['ball'] if f['data']['ball'] is not None else [np.nan]*3 for f in tracking],dtype=float)*.3048
    sl=slice(anchor-1,anchor+4)
    if not all(np.isfinite(a[sl]).all() for a in (ball,thumb,pinky)):raise ValueError('Missing evaluation-window points')
    # Only frames used by this evaluation need a nondegenerate hand frame.
    rotations=frame(wrist[sl],thumb[sl],pinky[sl])
    r=rotations[1].T@(ball[anchor]-wrist[anchor])
    return {'participant':d['participant_id'],'trial':d['trial_id'],'anchor_frame':int(indices[anchor]),
            'side':side,'wrist':wrist[sl],'rotation':rotations,'ball':ball[sl],'local':r,'rate':rate}


def run(root=Path('artifacts/spl')):
    selection=json.loads((root/'selection.json').read_text());cases=[];failures=[]
    for path in selection['paths']:
        try:cases.append(load_case(root/path))
        except (ValueError,KeyError,TypeError) as e:failures.append({'path':path,'reason':str(e)})
    if len(set(c['participant'] for c in cases))<2:raise ValueError('Need more than one participant')
    rows=[];personalized_rows=[]
    for case in cases:
        train=[c for c in cases if c['participant']!=case['participant']]
        r_train=np.median([c['local'] for c in train],axis=0)
        calibration=[c for c in cases if c['participant']==case['participant'] and c['trial'] in ('T0001','T0002')]
        w,R,b=case['wrist'],case['rotation'],case['ball'];rate=case['rate']
        for h in (1,2,3):
            target=b[1+h]
            predictions={'hand_only_fixed_grip':rigid_point(w[1+h],R[1+h],r_train),
                         'hand_plus_one_ball_position':rigid_point(w[1+h],R[1+h],case['local']),
                         'two_ball_positions_gravity':ballistic(b[0],b[1],1/rate,h/rate)}
            row={k:case[k] for k in ('participant','trial','anchor_frame','side')}
            row.update(horizon_frames=h,horizon_ms=1000*h/rate,
                       local_grip_change_mm=float(np.linalg.norm(R[1+h].T@(target-w[1+h])-case['local'])*1000))
            row['errors_mm']={name:float(np.linalg.norm(pred-target)*1000) for name,pred in predictions.items()}
            rows.append(row)
            if case['trial'] not in ('T0001','T0002') and len(calibration)==2:
                personal=np.median([c['local'] for c in calibration],axis=0)
                personalized_rows.append({**row,'personalized_fixed_grip_error_mm':float(np.linalg.norm(rigid_point(w[1+h],R[1+h],personal)-target)*1000)})
    summaries=[]
    for h in (1,2,3):
        subset=[r for r in rows if r['horizon_frames']==h]
        summaries.append({'horizon_frames':h,'trials':len(subset),
                          'mean_error_mm':{name:float(np.mean([r['errors_mm'][name] for r in subset])) for name in subset[0]['errors_mm']},
                          'median_error_mm':{name:float(np.median([r['errors_mm'][name] for r in subset])) for name in subset[0]['errors_mm']},
                          'one_ball_better_trials':sum(r['errors_mm']['hand_plus_one_ball_position']<r['errors_mm']['hand_only_fixed_grip'] for r in subset)})
    personal_summaries=[]
    for h in (1,2,3):
        subset=[r for r in personalized_rows if r['horizon_frames']==h]
        if subset:
            personal_summaries.append({'horizon_frames':h,'evaluation_trials':len(subset),
                'personalized_fixed_mean_mm':float(np.mean([r['personalized_fixed_grip_error_mm'] for r in subset])),
                'one_ball_mean_mm':float(np.mean([r['errors_mm']['hand_plus_one_ball_position'] for r in subset])),
                'one_ball_better_trials':sum(r['errors_mm']['hand_plus_one_ball_position']<r['personalized_fixed_grip_error_mm'] for r in subset)})
    report={'selected':len(selection['paths']),'successful':len(cases),'failures':failures,'summaries':summaries,'rows':rows,
            'personalized_summaries':personal_summaries,'personalized_rows':personalized_rows,
            'personalized_scope':'Follow-up control after initial ablation: first two chronological trials calibrate each participant; remaining selected trials evaluate. No per-person grip fitting on evaluated trial. Exploratory, not pre-registered.',
            'selection':selection,'scope':'Basketball auxiliary OFFLINE hand-conditioned reconstruction. Hand at target frame available to both hand methods. No actual release labels, independent radar speed, muscle constraints or causal intervention.',
            'anchor':'Side with larger maximum upward wrist speed; SG 7-frame quadratic derivative, 3 edge frames excluded. Hand-only surrogate, not ball release.',
            'training':'Fixed local grip median from other participants only; evaluated participant excluded. One-ball model uses only anchor ball position, not target ball. Ballistic reference uses anchor and preceding ball positions, no hand.',
            'units':'Source feet x0.3048=m; frame index/60=s. JSON timestamp ms interpretation checked within1.1ms.',
            'limits':'Same tracking system supplies inputs and withheld future ball target; correlated errors and no independent measurement. 30 selected trials, only five participants, no population significance claim.'}
    out=Path('artifacts/direct-observation');out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('selected','successful','failures','summaries')},indent=2))
    return report


if __name__=='__main__':run()
