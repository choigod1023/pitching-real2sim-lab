import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent/'artifacts/anatomical-shoulder'
OUT=BASE.parent/'shoulder-marker-stability'

def muscle_metrics(row):
    success=[r['hill_equilibrium_interval'] for r in row['rows'] if r['hill_equilibrium_interval']['status']=='success']
    return dict(hill_residual_rms_nm=row['hill_residual_rms_nm'],
        max_endpoint_tendon_force_n=max(r['endpoint_force_max_n'] for r in success),
        max_normalized_fiber_length=max(r['normalized_fiber_length_max'] for r in success))

def main():
    legacy={r['pitch']:r for r in json.loads((OUT/'legacy_buggy_screen.json').read_text())['reports']}
    fixed={r['pitch']:r for r in json.loads((BASE/'muscle_screen.json').read_text())['reports']}
    current={r['pitch']:r for r in json.loads((OUT/'muscle_screen.json').read_text())['reports']}
    before={r['pitch']:r for r in json.loads((BASE/'ik_report.json').read_text())['rows']}
    after=json.loads((OUT/'ik_report.json').read_text())['rows'];rows=[]
    for r in after:
        pitch=r['pitch'];z=np.load(OUT/f'{pitch}_ik.npz');old=np.load(BASE/f'{pitch}_ik.npz')
        j=list(z['coordinate_names']).index('pro_sup');rot=z['orientation_targets']
        relative=np.einsum('nji,njk->nik',rot[:,1],rot[:,2])
        variation=Rotation.from_matrix(np.einsum('ij,njk->nik',relative[0].T,relative)).magnitude()
        row=dict(pitch=pitch,before_arm_rmse_mm=before[pitch]['arm_rmse_mm'],after_arm_rmse_mm=r['arm_rmse_mm'],
            before_orientation_rmse_deg=r['before_orientation_rmse_deg'],after_orientation_rmse_deg=r['orientation_rmse_deg'],
            before_pronation_max_step_deg=float(np.rad2deg(np.max(abs(np.diff(old['q'][:,j]))))),
            after_pronation_max_step_deg=float(np.rad2deg(np.max(abs(np.diff(z['q'][:,j]))))),
            observed_hand_relative_forearm_change_max_deg=float(np.rad2deg(variation.max())),
            legacy_buggy=muscle_metrics(legacy[pitch]),bugfix_only=muscle_metrics(fixed[pitch]),direction_markers=muscle_metrics(current[pitch]))
        rows.append(row)
    result=dict(rows=rows,mean_before_arm_rmse_mm=float(np.mean([r['before_arm_rmse_mm'] for r in rows])),mean_after_arm_rmse_mm=float(np.mean([r['after_arm_rmse_mm'] for r in rows])),
        mean_corrected_before_hill_residual_nm=float(np.mean([r['bugfix_only']['hill_residual_rms_nm'] for r in rows])),mean_after_hill_residual_nm=float(np.mean([r['direction_markers']['hill_residual_rms_nm'] for r in rows])),
        source_model_unchanged=hashlib.sha256((BASE/'scaled_source.osim').read_bytes()).digest()==hashlib.sha256((OUT/'scaled_source.osim').read_bytes()).digest(),
        prerequisite_passed=False,candidate_search_executed=False,accepted_candidates=0,
        reason='Orientation residual, source joint bounds and muscle force-balance residual remain. Source wrist is welded despite varying observed relative hand frames. No ball-contact or full muscle-driven baseline has passed.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
