"""Paired model comparison; all scenario ranges remain engineering assumptions."""
import hashlib,json
from pathlib import Path
import numpy as np
from friction_contact import DIRECTIONS


def run():
    root=Path(__file__).resolve().parent
    new=json.loads((root/'friction_contact_results.json').read_text())
    old=json.loads((root/'fine_uncertainty_results.json').read_text())['sensitivity']
    rows=new['rows'];matched=[r for r in rows if r['mu']==.6]
    def span(values):return [float(min(values)),float(max(values))]
    original=span([r['baseline']['speed_kmh'] for r in old])
    nominal=next(r for r in rows if (r['stiffness'],r['release_ms'],r['mu'])==(10000.,2.,.6))
    original_nominal=next(r['baseline'] for r in old if (r['stiffness'],r['release_ms'])==(10000.,2.))
    record=128.4256512
    result=dict(old_range_kmh=original,new_matched_mu06_range_kmh=span([r['speed_kmh'] for r in matched]),
        new_all_mu_range_kmh=span([r['speed_kmh'] for r in rows]),
        original_nominal_speed_kmh=original_nominal['speed_kmh'],new_nominal_speed_kmh=nominal['speed_kmh'],
        original_nominal_record_gap_kmh=abs(original_nominal['speed_kmh']-record),new_nominal_record_gap_kmh=abs(nominal['speed_kmh']-record),
        record_speed_kmh=record,nominal=nominal,
        new_spin_range_rpm=span([r['spin_rpm'] for r in rows]),new_contact_fraction_range=span([r['contact_fraction'] for r in rows]),
        new_penetration_range_mm=span([r['max_penetration_mm'] for r in rows]),
        new_energy_residual_range_j=span([r['energy_balance_residual_j'] for r in rows]),
        convergence=[dict(stiffness=r['stiffness'],release_ms=r['release_ms'],mu=r['mu'],speed_difference_kmh=r['speed_kmh']-r['coarse_speed_kmh'],energy_residual_j=r['energy_balance_residual_j']) for r in new['checks']],
        no_contact=new['no_contact'],
        all_active_normal_stiffness_eigenvalues_per_total_k=np.linalg.eigvalsh(DIRECTIONS.T@DIRECTIONS/3).tolist(),
        new_results_sha256=hashlib.sha256((root/'friction_contact_results.json').read_bytes()).hexdigest(),
        old_results_sha256=hashlib.sha256((root/'fine_uncertainty_results.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((root.parent/'trace-error-audit/artifacts/trace-error/3034_4-11520-pad1-foh-sub1.npz').read_bytes()).hexdigest(),
        interpretation='Same total stiffness label does not imply same directional compliance. Normal eigenvalues assume all pads active, omit friction and are not measured stiffness. No athlete improvement or independent validation.')
    (root/'friction_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
