"""Recompute scientific engineering verdict from executed raw/source artifacts."""
import json
from pathlib import Path
import numpy as np
from prime_board_audit import ROOT,OUT,cameras
from fetch_prime_calibration import sha


def run():
    board=json.loads((ROOT/'prime_board_results.json').read_text())
    cube=json.loads((ROOT/'prime_cube_results.json').read_text())
    corrected=json.loads((ROOT/'prime_cube_correction_results.json').read_text())
    background=json.loads((ROOT/'prime_background_results.json').read_text())
    downloads=json.loads((OUT/'download_manifest.json').read_text())
    for r in downloads:
        assert r['hash_matches_upstream'] and sha(r['local_path'])==r['expected_sha256']
    for p,h in cube['freeze']['sha256'].items():assert sha(p)==h,p
    assert board['observations_sha256']==sha(OUT/'prime_board_detections.json')
    assert len(board['rows'])==27 and board['evaluated_pairs']==23 and board['excluded_pairs']==4
    assert len(cube['camera21_correspondence_alternatives'])==8
    assert not board['camera_parameters_modified'] and not cube['camera_parameters_changed']
    assert corrected['code_sha256']==sha(ROOT/'prime_cube_correction.py')
    assert corrected['calibration_accepted']==all(corrected['engineering_checks'].values())
    assert not corrected['calibration_accepted'] and not corrected['applied_to_pitching']
    assert all(v['after_count']==v['before_count'] for v in corrected['board_summary'].values())
    assert len(background['rows'])==6 and not background['applied_to_pitching']
    for k,r,t in cameras().values():
        assert np.isfinite(k).all() and np.isfinite(t).all()
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-8);assert abs(np.linalg.det(r)-1)<1e-8
    result={'artifact_integrity_passed':True,'calibration_accepted':False,'scientific_stage_passed':False,
            'independent_cube_target_checked':True,'actual_joint_accuracy_established':False,'actual_athlete_gain_verified':False,
            'physical_capture_to_export_clock_verified':False,'population_uncertainty_interval_available':False,
            'reason':'Cube-only extrinsic correction improves cube midpoints but worsens all excluded checkerboard pairs. Do not apply to pitching.',
            'source_original_B_speed_mae_kmh':1.3241394341168302,'new_speed_mae_kmh':None,
            'source_sha256':{str(ROOT/n):sha(ROOT/n) for n in ['prime_board_results.json','prime_cube_results.json','prime_cube_correction_results.json','prime_background_results.json']}}
    (ROOT/'prime_camera_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result));return result


if __name__=='__main__':run()
