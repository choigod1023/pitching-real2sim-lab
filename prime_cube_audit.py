"""Independent cube target checks; do not fit cameras, lengths or athlete speed."""
import itertools
import json
import numpy as np
from scipy.spatial.transform import Rotation
from prime_board_audit import ROOT,OUT,cameras,projection,triangulate
from fetch_prime_calibration import sha

VERTICES=[a+z for z in ['1','0'] for a in 'ABCD']
MIDS=[a+'m' for a in 'ABCD']
EDGES=[(a+z,b+z) for z in ['1','0'] for a,b in zip('ABCD','BCDA')]+[(a+'0',a+'1') for a in 'ABCD']


def observations():
    a=json.loads((ROOT/'prime_cube_associations.json').read_text());o=json.loads((ROOT/'prime_cube_observations.json').read_text())
    return a,{int(c):{name:np.array(o[c]['markers'][index]['pixel']) for name,index in markers.items()} for c,markers in a['cameras'].items()}


def metric(points,residual,labels):
    p=dict(zip(labels,points));outer=[];halves=[];midpoint=[]
    for left,right in EDGES:outer.append(float(np.linalg.norm(p[left]-p[right])*1000))
    for a in 'ABCD':
        for z in ['0','1']:halves.append(float(np.linalg.norm(p[a+'m']-p[a+z])*1000))
        midpoint.append(float(np.linalg.norm(p[a+'m']-(p[a+'0']+p[a+'1'])/2)*1000))
    return {'mean_reprojection_px':float(np.mean(residual)),'max_reprojection_px':float(np.max(residual)),
            'outer_edges_mm':dict(zip([f'{a}_{b}' for a,b in EDGES],outer)),
            'outer_edge_mae_mm':float(np.mean(np.abs(np.array(outer)-698.5))),
            'outer_edge_max_abs_error_mm':float(np.max(np.abs(np.array(outer)-698.5))),
            'vertical_half_edges_mm':halves,'half_edge_mae_mm':float(np.mean(np.abs(np.array(halves)-349.25))),
            'vertical_midpoint_displacement_mm':midpoint,'points_m':p}


def plain(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,dict):return {k:plain(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [plain(v) for v in obj]
    return obj


def run():
    freeze=json.loads((ROOT/'prime_cube_annotation_freeze.json').read_text())
    for p,h in freeze['sha256'].items():assert sha(p)==h,p
    a,obs=observations();cam=cameras();labels=VERTICES+MIDS
    primary,residual=triangulate([obs[17][k] for k in labels],[obs[20][k] for k in labels],cam[17],cam[20])
    p=dict(zip(labels,primary));hypotheses=[]
    for winding in [1,-1]:
        for shift in range(4):
            mapping={letter:'ABCD'[(shift+winding*i)%4] for i,letter in enumerate('ABCD')}
            mapped={key:obs[21][mapping[key[0]]+key[1]] for key in labels}
            residual21=np.linalg.norm(projection(np.array([p[k] for k in ['A1','B1','C1','D1']]),cam[21])-np.array([mapped[k] for k in ['A1','B1','C1','D1']]),axis=1)
            hypotheses.append({'winding':winding,'shift':shift,'map':mapping,'top_outer_mean_reprojection_px':float(residual21.mean())})
    selected=min(hypotheses,key=lambda h:h['top_outer_mean_reprojection_px'])
    mapping=selected['map'];obs[21]={key:obs[21][mapping[key[0]]+key[1]] for key in labels}
    result={'reference':'Prime OptiTrack static cube, separate from Edger cameras','freeze':freeze,'primary_pair17_20':metric(primary,residual,labels),
            'camera21_correspondence_alternatives':hypotheses,'camera21_selected_by_top_pixels_only':selected,
            'camera21_midpoints_not_used_for_association_mean_reprojection_px':float(np.mean(np.linalg.norm(projection(np.array([p[k] for k in MIDS]),cam[21])-np.array([obs[21][k] for k in MIDS]),axis=1))),
            'pairs':{},'camera_parameters_changed':False,'athlete_changed':False,'scientific_stage_passed':False,'actual_speed_gain_kmh':None}
    for left,right in itertools.combinations(cam,2):
        try:
            pts,res=triangulate([obs[left][k] for k in labels],[obs[right][k] for k in labels],cam[left],cam[right])
            result['pairs'][f'{left}_{right}']=metric(pts,res,labels)
        except ValueError as e:result['pairs'][f'{left}_{right}']={'error':str(e)}
    # Deterministic coordinate stress, not a random probability/confidence interval.
    stresses=[]
    for camera in [17,20]:
        for axis in [0,1]:
            for sign in [-1.,1.]:
                left=np.array([obs[17][k] for k in labels]);right=np.array([obs[20][k] for k in labels])
                (left if camera==17 else right)[:,axis]+=sign
                points,res=triangulate(left,right,cam[17],cam[20]);m=metric(points,res,labels)
                stresses.append({'camera_shifted':camera,'axis':axis,'pixels':sign,'outer_mae_mm':m['outer_edge_mae_mm'],'max_edge_error_mm':m['outer_edge_max_abs_error_mm']})
    result['coherent_one_pixel_coordinate_sensitivity']=stresses
    (ROOT/'prime_cube_results.json').write_text(json.dumps(plain(result),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'primary':{k:v for k,v in result['primary_pair17_20'].items() if k not in ['points_m','outer_edges_mm','vertical_half_edges_mm','vertical_midpoint_displacement_mm']},'camera21':selected,'camera21_midpoint_check_px':result['camera21_midpoints_not_used_for_association_mean_reprojection_px'],'pairs':{n:{k:v for k,v in m.items() if k in ['outer_edge_mae_mm','outer_edge_max_abs_error_mm','half_edge_mae_mm','mean_reprojection_px']} for n,m in result['pairs'].items()}}))
    return result


if __name__=='__main__':run()
