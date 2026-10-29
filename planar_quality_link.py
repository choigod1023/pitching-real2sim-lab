import json
from pathlib import Path
import numpy as np
from dense_camera_link import ROOT,pair_cameras,static_checks
from recover_board_order import fit_correspondence
from dynamic_camera_recovery import board_metrics,board_screen

def run():
    src=json.loads(Path('edge_intrinsics_frozen.json').read_text());base=json.loads((ROOT/'dense-link-results.json').read_text())['methods'];quality=json.loads((ROOT/'per-view-planarity-diagnostic.json').read_text());lookup={(r['camera'],r['frame'],r['method']):r['homography_rms_px'] for r in quality};methods=['native_sb','source720_subpix5'];common=sorted(set.intersection(*(set(base[m]['shared']) for m in methods)));kept=[f for f in common if all(lookup[i,f,m]<=2 for i in [1,3] for m in methods)];excluded={f:{f'{i}_{m}':lookup[i,f,m] for i in [1,3] for m in methods} for f in common if f not in kept};train=[f for f in kept if f%5];held=[f for f in kept if not f%5];out={'common_before_quality':common,'kept':kept,'excluded':excluded,'train':train,'heldout':held,'methods':{},'scientific_stage_passed':False}
    for method in methods:
        if len(kept)<8 or len(train)<5 or len(held)<2:out['methods'][method]={'status':'insufficient_observations'};continue
        data={i:{f:np.array(json.loads((ROOT/'coverage'/f'cam{i}_frame{f}.json').read_text())['methods'][method]['points']) for f in kept} for i in [1,3]};r,t,rms,history,seed,mapped,costs=fit_correspondence(data,train,src);cc=pair_cameras(r,t,src);board=board_metrics(held,data[1],mapped,cc);cube,plane=static_checks(cc,1);alt,altplane=static_checks(cc,3)
        out['methods'][method]={'status':'calculated','train_rms_px':rms,'R':r.tolist(),'t_mm':t.tolist(),'iterations':history,'seed':seed,'association_costs':costs,'heldout_board':board,'primary_cube':cube,'phase3_diagnostic_cube':alt,'plane':plane,'engineering_pass':bool(board_screen(board) and cube['max_abs_error_percent']<=1 and max(cube['projection_rms_px_by_corner'].values())<=2 and cube['min_depth_mm']>0 and plane['conditional_metric_pass']),'scientific_stage_passed':False}
    (ROOT/'planar-quality-link-results.json').write_text(json.dumps(out,indent=2));print(json.dumps({'kept':len(kept),'excluded':len(excluded),'train':len(train),'heldout':len(held),'methods':{m:{'train_rms_px':v.get('train_rms_px'),'heldout_projection_mean_px':float(np.mean([x['reprojection_rms_px'] for x in v.get('heldout_board',[])])) if v.get('heldout_board') else None,'phase3_cube_mae_mm':v.get('phase3_diagnostic_cube',{}).get('length_mae_mm'),'engineering_pass':v.get('engineering_pass')} for m,v in out['methods'].items()}}));return out
if __name__=='__main__':run()
