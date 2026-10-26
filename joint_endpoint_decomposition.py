"""Endpoint pose/rate diagnostic; mixed states are not simulated interventions."""
import json
from pathlib import Path
import numpy as np
from coupled_payload import model_for
from speed_gap_numeric import state_components

ROOT=Path(__file__).resolve().parent

def run():
    folder=ROOT/'artifacts/joint-intervention'
    with np.load(folder/'state-baseline-sub2.npz') as b:
        q0=b['qpos'][-1].copy();v0=b['qvel'][-1].copy()
        m=model_for(b['lengths'],float(b['dt']))
    def speed(q,v):return state_components(m,q,v)['ball_speed_kmh']
    s00=speed(q0,v0); rows=[]
    for path in sorted(folder.glob('state-*-sub2.npz')):
        with np.load(path) as x:
            q1=x['qpos'][-1];v1=x['qvel'][-1]
            s10=speed(q1,v0);s01=speed(q0,v1);s11=speed(q1,v1)
            rows.append(dict(case=path.name[len('state-'):-len('-sub2.npz')],baseline_kmh=s00,
                full_changed_kmh=s11,total_change_kmh=s11-s00,
                pose_only_with_baseline_generalized_velocity_change_kmh=s10-s00,
                velocity_only_with_baseline_pose_change_kmh=s01-s00,
                pose_then_velocity_increment_kmh=s11-s10,
                interaction_kmh=s11-s10-s01+s00))
    out=ROOT/'artifacts/joint-endpoint';out.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Algebraic Jacobian evaluations on mixed generalized states. Not independent causal contributions, realizable trajectories, or actual intervention effects. Pose/rate decomposition depends on coordinate choice and ordering.',rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    print(json.dumps([r for r in run()['rows'] if r['case'] in ['elbow_p1','elbow_p2','wrist_y_p2','wrist_z_p2']],indent=2))
