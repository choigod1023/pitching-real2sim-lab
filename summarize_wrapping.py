"""Keep successful local checks and failed full-domain alternatives together."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/wrapping-consistency'

def read(name):return json.loads((OUT/name).read_text())
def selected(r):return next(x for x in r['models'] if x['degree']==r['selected_degree'])

def main():
    isolation=read('isolation.json');global_fit=read('length_based_checks.json');wide=read('local_0p02_length_based_checks.json');local=read('local_0p002_length_based_checks.json');history=read('history_checks.json')
    report={'native_force_mapping_verified':True,'global_geometry_accepted':False,'local_derivative_pass_only':True,'full_pitch_optimization_eligible':False,'candidate_search_executed':False,
        'isolation':[{'condition':r['model'],'length_m':r['cold_length_m'],'speed_m_s':r['cold_speed_m_s'],'max_derivative_error_m_s':max(x['speed_error_m_s'] for x in r['sweep'])} for r in isolation['reports']],
        'source_angle_step_deg':history['slices'][0]['angle_step_deg'],'source_maximum_length_jump_mm':history['slices'][0]['maximum_neighbor_length_change_mm'],'same_pose_history_range_mm':history['same_pose_history_range_mm'],
        'geodesic_trial':read('geodesic_status.json'),'global_length_fit':selected(global_fit),'local_0p02_length_fit':selected(wide),'local_0p002_length_fit':selected(local),
        'local_domain_low_rad':local['domain_low_rad'],'local_domain_high_rad':local['domain_high_rad'],
        'local_before_max_derivative_error_m_s':max(v for r in local['before'] for sw in r['sweeps'] for v in sw['error_m_s']),
        'local_after_max_derivative_error_m_s':max(v for r in local['after'] for sw in r['sweeps'] for v in sw['error_m_s']),
        'force_mapping':read('force_mapping_comparison.json'),'local_single_state_force':read('local_force_comparison.json')}
    (ROOT/'wrapping_fix_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
