"""Report paired observations and gate coaching claims; no causal prescription."""
import json
import numpy as np
from anatomical_shoulder_fit import OUT

def local_shoulder(z):
    clav,c7,stern=np.moveaxis(z['torso'],1,0)
    up=clav-stern;up/=np.linalg.norm(up,axis=1)[:,None]
    anterior=clav-c7;anterior-=np.sum(anterior*up,axis=1)[:,None]*up;anterior/=np.linalg.norm(anterior,axis=1)[:,None]
    lateral=np.cross(anterior,up)
    axes=np.stack([anterior,lateral,up],axis=2)
    return np.einsum('nji,nj->ni',axes,z['arm'][:,0]-clav)

def main():
    meta=json.loads((OUT/'input_manifest.json').read_text());fast=max(meta,key=lambda p:meta[p]['measured_speed_kmh']);slow=min(meta,key=lambda p:meta[p]['measured_speed_kmh'])
    ik=json.loads((OUT/'ik_report.json').read_text())['rows'];idx={r['pitch']:r for r in ik};screen=json.loads((OUT/'muscle_screen.json').read_text())['reports']
    a=np.load(OUT/f'{fast}_input.npz');b=np.load(OUT/f'{slow}_input.npz');delta=local_shoulder(a)[-1]-local_shoulder(b)[-1]
    result=dict(fast_pitch=fast,slow_pitch=slow,measured_speed_difference_kmh=meta[fast]['measured_speed_kmh']-meta[slow]['measured_speed_kmh'],
        source_model_wrist_proxy_difference_kmh=idx[fast]['wrist_speed_at_br_kmh']-idx[slow]['wrist_speed_at_br_kmh'],
        wrist_proxy_is_not_ball_prediction=True,observed_br_local_shoulder_difference_mm=delta.tolist(),
        local_axes=['CLAV-C7 anterior proxy','cross(anterior,superior) lateral proxy','CLAV-STRN superior proxy'],
        accepted_improvement_candidates=0,counterfactual_candidates_evaluated=0,
        reason='No valid baseline muscle-driven ball simulation; prerequisite failed, not proof that no feasible improvement exists.',
        model_frames=sum(r['frames'] for r in screen),frames_active_envelope_residual_over_1nm=sum(r['envelopes']['1.4']['frames_residual_over_1nm'] for r in screen),
        ability_improvement_demonstrated=False,source_model_ik_only=True)
    result['observed_br_local_shoulder_difference_mm']=(delta*1000).tolist()
    (OUT/'training_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
