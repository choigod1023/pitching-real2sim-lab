"""Independently replay selected cases and inspect all budget/initial-state checks."""
import json
from pathlib import Path
import numpy as np
from audit_multi_joint_timing import replay, independent_budget, rotations

def main():
    folder=Path('artifacts/observed-candidate')
    report=json.loads((folder/'report.json').read_text())
    fine={r['case']:r for r in report['runs'] if r.get('substeps')==2}
    coarse={r['case']:r for r in report['runs'] if r.get('substeps')==1}
    base=fine['slow_baseline']; initial=np.load(folder/'state-slow_baseline-sub2.npz')
    checks=[]; tracking=[]
    for name,r in fine.items():
        state=np.load(folder/f'state-{name}-sub2.npz')
        delta=rotations(state['candidate_q'],state['qpos'])
        tracking.append({'case':name,'max_reference_tracking_rotation_deg':delta,
                         'shoulder_translation_rmse_m':float(np.sqrt(np.mean(np.sum((state['candidate_q'][:,:3]-state['qpos'][:,:3])**2,axis=1))))})
        if name!='fast_observed':
            assert np.array_equal(state['qpos'][0],initial['qpos'][0])
            assert np.array_equal(state['qvel'][0],initial['qvel'][0])
            assert independent_budget(base,r)==r['budget_gate']['strict_model_budget_pass']
            assert independent_budget(coarse['slow_baseline'],coarse[name])==coarse[name]['budget_gate']['strict_model_budget_pass']
    for name in ['slow_baseline','elbow_fraction_+0.25','elbow_fraction_+1']:
        state=np.load(folder/f'state-{name}-sub2.npz'); r=fine[name]
        q,v,pos,neg,speed=replay(state['candidate_q'],state['controls'],state['lengths'],float(state['dt']))
        errors={'qpos':float(np.max(np.abs(q-state['qpos']))),'qvel':float(np.max(np.abs(v-state['qvel']))),
                'positive_work_j':float(np.max(np.abs(pos-r['positive_work_per_dof_j']))),
                'negative_work_j':float(np.max(np.abs(neg-r['negative_work_magnitude_per_dof_j']))),
                'speed_kmh':abs(speed-r['speed_kmh'])}
        assert max(errors.values())<1e-8
        checks.append({'case':name,'replay_max_errors':errors})
    result={'selected_fine_replays':checks,'all_initial_state_and_budget_checks_passed':True,
        'trajectory_tracking':tracking,'max_coarse_fine_speed_difference_kmh':max(abs(fine[k]['speed_kmh']-coarse[k]['speed_kmh']) for k in fine),
        'observed_change_kmh':report['measured_fast_minus_slow_kmh'],
        'model_independent_fast_minus_slow_kmh':fine['fast_observed']['speed_kmh']-base['speed_kmh']}
    (folder/'independent-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
