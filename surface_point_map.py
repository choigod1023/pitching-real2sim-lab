"""Local surface-observation map, explicitly leaving skeleton and B unchanged."""
import json
import ezc3d
import numpy as np
from scipy.spatial.transform import Rotation,RotationSpline
from scipy.optimize import least_squares
from free_arm_regions import ROOT,Source,observations,sha,box_residual,project,ball_metric
from hand_speed_derivative import numerical_rotations


def run():
    s=Source();train,test=observations();rotation={'l_hand':s.hand}
    c=ezc3d.c3d(s.observation['source_sha256']['body']['path']);labels=c['parameters']['ROTATION']['LABELS']['value']
    q=np.array(json.loads(s.alignment_path.read_text(encoding='utf-8'))['rotation'])
    raw=c['data']['rotations'][:3,:3,labels.index('l_larm_4X4'),:]
    proper,_=numerical_rotations(np.einsum('ij,jkn->nik',q,raw))
    rotation['l_larm']=RotationSpline(np.arange(len(proper)),Rotation.from_matrix(proper))
    def errors(rows,offset,margin=0):
        values=[]
        for row in rows:
            seg=row['segment'];i={'l_larm':0,'l_hand':1}[seg]
            origin=s.origins[seg](row['frame']+1)
            point=origin+rotation[seg](row['frame']+1).apply(offset[i])
            values.append(box_residual(project(point,s.cameras[row['camera']])[0],row['box'],margin))
        return np.array(values)
    def residual(x):
        return np.r_[errors(train,x.reshape(2,3)).ravel()/np.sqrt(len(train)),np.sqrt(.001)*x/.01]
    fit=least_squares(residual,np.zeros(6),bounds=(-.03,.03),max_nfev=1000,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    offset=fit.x.reshape(2,3)
    def summary(rows,x,margin=0):
        d=np.linalg.norm(errors(rows,x,margin),axis=1)
        return {'count':len(d),'mean_outside_px':float(d.mean()),'max_outside_px':float(d.max()),
                'per_camera':{str(c):float(d[[r['camera']==c for r in rows]].mean()) for c in sorted({r['camera'] for r in rows})}}
    models={name:{'training':summary(train,x),'evaluation':summary(test,x),
                  'boundary_sensitivity':{str(m):summary(test,x,m) for m in [-3,0,3]}} for name,x in [('joint_origin',np.zeros((2,3))),('local_surface_map',offset)]}
    a,b=models['joint_origin'],models['local_surface_map']
    checks={'solver_success':bool(fit.success),'offset_not_at_bound':bool(np.max(np.abs(offset))<.03-1e-5),
            'evaluation_mean_improved':b['evaluation']['mean_outside_px']<a['evaluation']['mean_outside_px'],
            'each_camera_nonworsening':all(b['evaluation']['per_camera'][k]<=v+1e-9 for k,v in a['evaluation']['per_camera'].items()),
            'boundary_nonworsening':all(b['boundary_sensitivity'][str(m)]['mean_outside_px']<=a['boundary_sensitivity'][str(m)]['mean_outside_px']+1e-9 for m in [-3,0,3])}
    recomputed=ball_metric(s,None)
    primary=json.loads((ROOT/'free_arm_results.json').read_text())['models']['source']['B_ball_evaluation']
    for key in ['mean_absolute_speed_discrepancy_kmh','mean_velocity_vector_discrepancy_kmh','position_rmse_mm']:
        np.testing.assert_allclose(recomputed[key],primary[key],atol=1e-10,rtol=0)
    result={'protocol_sha256':sha(ROOT/'SURFACE_MAP_PROTOCOL.md'),'code_sha256':sha(__file__),'models':models,'checks':checks,
            'diagnostic_accepted':all(checks.values()),'local_offsets_mm':(offset*1000).tolist(),'offset_order':['l_larm','l_hand'],
            'source_B_ball_evaluation_unchanged':recomputed,'body_motion_changed':False,'hand_rotation_changed':False,
            'identified_anatomical_skin_offset':False,'scientific_stage_passed':False,'actual_athlete_gain_kmh':None}
    (ROOT/'surface_point_map_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'offsets_mm':result['local_offsets_mm'],'models':models,'checks':checks}))
    return result


if __name__=='__main__':run()
