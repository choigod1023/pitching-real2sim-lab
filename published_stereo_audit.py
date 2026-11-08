"""Evaluate the available multiview checkerboard stereo fit on other targets."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
from edge_calibration import OUT,INTRINSICS,camera_parameters,triangulate,lengths,serialise,metric_screen

def run():
    s=json.loads(INTRINSICS.read_text());path=Path('edge_target_annotations_plane_round3.json');a=json.loads(path.read_text())
    sp=s['stereo_pairs']['edgertronic_1__edgertronic_4']
    cameras=[]
    for i in [1,4]:
        k,d=camera_parameters(s,i)
        cameras.append({'R':np.eye(3) if i==1 else np.array(sp['rotation_matrix']),
                        't':np.zeros(3) if i==1 else np.array(sp['translation_mm']),'K':k,'dist':d})
    points=[];ee=[];depth=[]
    for name in ['O','L','S']:
        x,e,z=triangulate([a['cameras'][i]['plane'][name] for i in ['1','4']],cameras);points.append(x);ee.extend(e);depth.extend(z)
    rms=float(np.sqrt(np.mean(np.sum(np.array(ee)**2,1))));ll=lengths(points)
    cube={};errs={}
    for name in a['cameras']['1']['fit']:
        x,e,z=triangulate([a['cameras'][i]['fit'][name] for i in ['1','4']],cameras);cube[name]=x;errs[name]=float(np.sqrt(np.mean(np.sum(e**2,1))))
    edges=[]
    for p,q in itertools.combinations(cube,2):
        if np.count_nonzero(np.array(a['cube_coordinates'][p])-a['cube_coordinates'][q])==1:
            size=float(np.linalg.norm(cube[p]-cube[q]));edges.append({'edge':[p,q],'mm':size,'error_percent':100*(size/a['cube_side_mm']-1)})
    # Each +/-1px observation perturbation; no statistical probability is assigned.
    sensitivity=[]
    for view,name,axis,delta in itertools.product(range(2),range(3),range(2),[-1,1]):
        uv=[list(a['cameras'][i]['plane'][['O','L','S'][name]]) for i in ['1','4']];uv[view][axis]+=delta
        x,_,_=triangulate(uv,cameras);pp=np.array(points);pp[name]=x
        sensitivity.append({'view':[1,4][view],'point':['O','L','S'][name],'axis':axis,'delta_px':delta,'lengths_mm':lengths(pp)})
    result={'source_stereo_pair':'edgertronic_1__edgertronic_4','source_hash':hashlib.sha256(INTRINSICS.read_bytes()).hexdigest(),
        'annotation_hash':hashlib.sha256(path.read_bytes()).hexdigest(),'cameras':[serialise(c) for c in cameras],
        'cube_edges':edges,'cube_reprojection':errs,'plane_lengths_mm':ll,'plane_reprojection_rms_px':rms,'plane_min_depth_mm':min(depth),
        'plane_metric_status':'UNKNOWN: terminal centre identity unconfirmed; conditional checks fail',
        'conditional_metric_pass':metric_screen(ll,a['plane_distances_mm'],rms,min(depth)),
        'engineering_pass':bool(metric_screen(ll,a['plane_distances_mm'],rms,min(depth)) and all(abs(e['error_percent'])<=1 for e in edges)),
        'plane_coordinate_sensitivity':sensitivity,'scientific_data_gate_pass':False,'speed_error_reduction_kmh':None}
    (OUT/'published-stereo-heldout-results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'plane_lengths_mm':ll,'plane_rms_px':rms,'cube_range_mm':[min(e['mm'] for e in edges),max(e['mm'] for e in edges)],'engineering_pass':result['engineering_pass']}))
    return result
if __name__=='__main__':run()
