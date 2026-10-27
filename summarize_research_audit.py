"""Machine-readable evidence status; engineering tolerances are not clinical limits."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/research-evidence-audit'

def path_consistency_gate(reports,tolerance=1e-4):
    # Require ALL prespecified finite-difference sizes to pass, never select
    # a favorable step. The result gates exploration, not injury or ability.
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive finite tolerance required')
    failed=[]
    for r in reports:
        values=[x['max_error_m_s'] for x in r.get('epsilon_sweep',[])]
        if not values or not np.isfinite(values).all() or any(x>tolerance for x in values):
            failed.append({'pitch':r['pitch'],'sample':r['sample'],'max_error_m_s':float(max(values)) if values and np.isfinite(values).all() else None})
    return {'passed':bool(reports) and not failed,'missing_evidence':not bool(reports),'engineering_tolerance_m_s':tolerance,'failed_states':failed,'scope':'Derivative consistency for synthetic tangent velocities; not athlete measurement precision.'}

def main():
    data=json.loads((OUT/'data_checks.json').read_text());formulas=json.loads((OUT/'formula_checks.json').read_text());constraints=json.loads((OUT/'constraint_checks.json').read_text())['rows']
    summary={'data':data,'formula_checks':formulas,'wrapping_control':json.loads((OUT/'wrapping_control.json').read_text()),'path_derivative_gate':path_consistency_gate(formulas['reports']),'constraint_mean':{mode:float(np.mean([x['rms']['residual'] for x in constraints if x['mode']==mode])) for mode in ['original','consistent']},'optimization_eligible':False,'athlete_performance_improvement_demonstrated':False,'reasons':['Wrapped path length/velocity disagreement','Large internal force-allocation residual','Unvalidated continuous muscle and hand-ball release dynamics','No athlete intervention validation']}
    (ROOT/'research_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print({'path_gate':summary['path_derivative_gate'],'constraint_mean':summary['constraint_mean']})

if __name__=='__main__':main()
