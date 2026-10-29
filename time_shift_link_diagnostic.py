import json
from pathlib import Path
import numpy as np
from dense_camera_link import ROOT,pair_cameras,static_checks
from recover_board_order import fit_correspondence
from dynamic_camera_recovery import board_metrics

def run():
    src=json.loads(Path('edge_intrinsics_frozen.json').read_text());out={}
    rows={i:{f:json.loads((ROOT/'coverage'/f'cam{i}_frame{f}.json').read_text()) for f in range(540,613)} for i in [1,3]}
    for method in ['native_sb','source720_subpix5']:
        points={i:{f:np.array(v['methods'][method]['points']) for f,v in r.items() if v['methods'][method]['points'] is not None} for i,r in rows.items()};ids=sorted(f for f in points[1] if all(f+s in points[3] for s in [-1,0,1]));train=[f for f in ids if f%5];held=[f for f in ids if not f%5];out[method]={'same_camera1_frames':ids,'train':train,'heldout':held,'shifts':{}}
        for shift in [-1,0,1]:
            if len(ids)<8 or len(train)<5 or len(held)<2:out[method]['shifts'][shift]={'status':'insufficient_observations'};continue
            data={1:{f:points[1][f] for f in ids},3:{f:points[3][f+shift] for f in ids}};r,t,rms,history,seed,mapped,costs=fit_correspondence(data,train,src);cc=pair_cameras(r,t,src);metrics=board_metrics(held,data[1],mapped,cc);cube,plane=static_checks(cc,1);altcube,altplane=static_checks(cc,3)
            out[method]['shifts'][shift]={'status':'calculated','camera3_source_frames':{f:f+shift for f in ids},'train_rms_px':rms,'R':r.tolist(),'t_mm':t.tolist(),'iterations':history,'seed':seed,'association_costs':costs,'heldout_board':metrics,'primary_phase1_cube':cube,'phase3_diagnostic_cube':altcube,'plane':plane,'scientific_stage_passed':False}
    result={'methods':out,'adopted_shift':None,'scientific_stage_passed':False,'speed_error_reduction_kmh':None};(ROOT/'time-shift-link-results.json').write_text(json.dumps(result,indent=2));print(json.dumps({m:{'same_frames':len(v['same_camera1_frames']),'shifts':{s:{'train_rms_px':x.get('train_rms_px'),'heldout_reprojection_mean_px':float(np.mean([q['reprojection_rms_px'] for q in x.get('heldout_board',[])])) if x.get('heldout_board') else None,'phase3_cube_mae_mm':x.get('phase3_diagnostic_cube',{}).get('length_mae_mm')} for s,x in v['shifts'].items()}} for m,v in out.items()}));return result
if __name__=='__main__':run()
