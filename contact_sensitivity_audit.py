"""Decompose the frozen 3x3 engineering grid; not population error attribution."""
import hashlib
import json
from pathlib import Path
import numpy as np


def decompose(rows):
    ks = sorted({r['stiffness'] for r in rows})
    ts = sorted({r['release_ms'] for r in rows})
    if len(ks) < 2 or len(ts) < 2:
        raise ValueError('Need at least two levels per factor')
    pairs = [(r['stiffness'], r['release_ms']) for r in rows]
    if len(set(pairs)) != len(pairs) or len(rows) != len(ks)*len(ts):
        raise ValueError('Complete unique factorial grid required')
    values = {(r['stiffness'], r['release_ms']): r['baseline']['speed_kmh'] for r in rows}
    a = np.array([[values[k,t] for t in ts] for k in ks])
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite output')
    mu = a.mean(); k_effect = a.mean(axis=1)-mu; t_effect = a.mean(axis=0)-mu
    interaction = a-mu-k_effect[:,None]-t_effect[None,:]
    ss = np.array([len(ts)*sum(k_effect**2), len(ks)*sum(t_effect**2), np.sum(interaction**2)])
    return dict(stiffness_levels=ks, release_ms_levels=ts, speed_kmh=a.tolist(),
                squared_deviation_components=ss.tolist(),
                grid_variation_percent=(100*ss/ss.sum()).tolist() if ss.sum()>0 else [0.,0.,0.],
                stiffness_range_at_fixed_release_kmh=np.ptp(a,axis=0).tolist(),
                release_range_at_fixed_stiffness_kmh=np.ptp(a,axis=1).tolist(),
                range_kmh=[float(a.min()),float(a.max())])


def run():
    root=Path(__file__).resolve().parent; source=root/'fine_uncertainty_results.json'
    data=json.loads(source.read_text()); rows=data['sensitivity']; result=decompose(rows)
    nominal=next(r['baseline'] for r in rows if r['stiffness']==10000 and r['release_ms']==2)
    result.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        recorded_speed_kmh=128.4256512, nominal_speed_kmh=nominal['speed_kmh'],
        nominal_record_gap_kmh=abs(nominal['speed_kmh']-128.4256512),
        extension_range_mm=[min(r['baseline']['max_grip_extension_mm'] for r in rows),max(r['baseline']['max_grip_extension_mm'] for r in rows)],
        loss_range_j=[min(r['baseline']['coupling_loss_j'] for r in rows),max(r['baseline']['coupling_loss_j'] for r in rows)],
        model_changed=False, heldout_before_mae_kmh=None, heldout_after_mae_kmh=None,
        independently_validated_error_reduction_kmh=None,
        scope='Reanalysis of previous frozen runs, not new measured trials. Equal-weight engineering grid, not causal/population variance or confidence bounds.')
    (root/'contact_sensitivity_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
