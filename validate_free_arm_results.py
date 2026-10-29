"""Validate executed artifacts; an engineering rejection is a valid finding."""
import json
from pathlib import Path
import numpy as np
from free_arm_regions import ROOT,sha


def run():
    r=json.loads((ROOT/'free_arm_results.json').read_text())
    d=json.loads((ROOT/'free_arm_diagnostics.json').read_text())
    surface=json.loads((ROOT/'surface_point_map_results.json').read_text())
    video=json.loads((ROOT/'artifacts/free-arm-regions/video_manifest.json').read_text())
    assert r['code_sha256']==sha(ROOT/'free_arm_regions.py')
    assert d['input_results_sha256']==sha(ROOT/'free_arm_results.json')
    assert surface['code_sha256']==sha(ROOT/'surface_point_map.py')
    for path,h in r['freeze']['sha256'].items():assert sha(path)==h,path
    assert len(r['training_rows'])==12 and len(r['evaluation_rows'])==40
    assert r['solver']['success'] and r['solver']['max_constraint_error_mm']<1e-6
    assert r['length_error_max_mm']<1e-6
    assert r['engineering_candidate_accepted']==all(r['engineering_checks'].values())
    assert not r['engineering_candidate_accepted'] and not surface['diagnostic_accepted']
    assert not r['scientific_stage_passed'] and r['actual_athlete_gain_kmh'] is None
    assert r['force_work_change'] is None
    assert not surface['body_motion_changed'] and not surface['hand_rotation_changed']
    assert video['result_sha256']==sha(ROOT/'free_arm_results.json')
    assert video['sha256']==sha(video['video']) and video['frames']==220 and video['all_frames_decoded']
    a,b=[r['models'][k]['B_ball_evaluation'] for k in ['source','region_reconstruction']]
    np.testing.assert_allclose(a['mean_absolute_speed_discrepancy_kmh'],1.3241394341168302,atol=1e-8)
    # Two adjacent optical intervals do not establish a population uncertainty.
    observed=np.array(b['candidate_interval_speed_kmh']);pred=np.array(b['predicted_interval_speed_kmh'])
    two_interval_mae_lower_bound=max(0.,(abs(np.diff(observed)[0])-abs(np.diff(pred)[0]))/2.)
    result={'artifact_integrity_passed':True,'engineering_reconstruction_accepted':False,
            'surface_mapping_accepted':False,'scientific_stage_passed':False,'actual_athlete_gain_verified':False,
            'ball_evaluation_points':3,'ball_speed_intervals':2,
            'two_interval_algebraic_mae_lower_bound_kmh':float(two_interval_mae_lower_bound),
            'primary_mae_above_this_bound_kmh':float(b['mean_absolute_speed_discrepancy_kmh']-two_interval_mae_lower_bound),
            'population_confidence_interval_available':False,'code_tests_establish_science_pass':False,
            'rejection_reasons':[key for key,value in r['engineering_checks'].items() if not value],
            'sources':{str(ROOT/name):sha(ROOT/name) for name in ['free_arm_results.json','free_arm_diagnostics.json','surface_point_map_results.json']}}
    (ROOT/'free_arm_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result));return result


if __name__=='__main__':run()
