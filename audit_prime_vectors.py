"""Audit saved held-out predictions as vectors, not only speed magnitudes."""
import json,hashlib
from pathlib import Path
import numpy as np


def compare_vectors(predicted,observed,dt):
    v=np.diff(predicted,axis=0)/dt;truth=np.diff(observed,axis=0)/dt
    nv=np.linalg.norm(v,axis=1);nt=np.linalg.norm(truth,axis=1)
    if np.any(nv<=0) or np.any(nt<=0):raise ValueError('Nonzero velocities required')
    angles=np.degrees(np.arccos(np.clip(np.sum(v*truth,axis=1)/(nv*nt),-1,1)))
    return dict(speed_mae_kmh=float(np.mean(abs(nv-nt))*3.6),
        vector_error_kmh=float(np.linalg.norm(v-truth,axis=1).mean()*3.6),
        mean_direction_error_deg=float(angles.mean()),direction_errors_deg=angles.tolist())


def run():
    sources=[Path('../heldout-speed-comparison/artifacts/physical-comparison/comparison.json'),
             Path('../hand-ball-link/artifacts/hand-ball/observation.json')]
    saved,obs=[json.loads(p.read_text()) for p in sources]
    rows=[]
    for item in saved['conditions']:
        matched=[r for r in obs['conditions'] if r['threshold']==item['threshold'] and r['c3d_frame_offset']==item['offset']]
        if len(matched)!=1:raise ValueError('Nonunique observation identity')
        truth=np.array([p['ball_rig_m'] for p in matched[0]['positions']])[4:]
        metrics={name:compare_vectors(np.array(m['predicted_positions_m']),truth,1/obs['rate_hz']) for name,m in item['metrics'].items()}
        rows.append(dict(threshold=item['threshold'],offset=item['offset'],metrics=metrics))
    result=dict(rows=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='Previously inspected single Prime pitch, training973-976 evaluation977-979; conditional video observations, not independent athlete intervention validation',
        models=dict(A='fixed hand-relative ball',B='linear relative hand-ball motion fitted on training ball observations',C='ball-only linear launch'),
        use='B consumes observed ball training positions; not a learned joint intervention response')
    out=Path('artifacts/release-vector');out.mkdir(parents=True,exist_ok=True)
    (out/'prime-vectors.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(next(r for r in rows if r['threshold']==80 and r['offset']==1),indent=2))


if __name__=='__main__':run()
