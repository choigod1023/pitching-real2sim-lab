"""Cube-only extrinsic correction tested against excluded midpoint/board data."""
import itertools
import json
import cv2
import numpy as np
from prime_board_audit import ROOT,OUT,cameras,triangulate,projection,lengths
from prime_cube_audit import observations,VERTICES,MIDS,metric,plain
from fetch_prime_calibration import sha


def rigid_registration(local,world):
    local,world=np.asarray(local,float),np.asarray(world,float)
    if local.ndim!=2 or local.shape!=world.shape or local.shape[1]!=3 or len(local)<3 or not np.isfinite([local,world]).all():raise ValueError('Invalid rigid correspondence')
    u,_,vt=np.linalg.svd((local-local.mean(axis=0)).T@(world-world.mean(axis=0)))
    r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    return r,world.mean(axis=0)-r@local.mean(axis=0)


def cube_chart():
    xy={'A':[0,0],'B':[1,0],'C':[1,1],'D':[0,1]}
    return {a+z:np.r_[v,{'0':0,'m':.5,'1':1}[z]]*.6985 for a,v in xy.items() for z in ['0','m','1']}


def run():
    _,obs=observations();old=cameras();base=json.loads((ROOT/'prime_cube_results.json').read_text());mapping=base['camera21_selected_by_top_pixels_only']['map']
    obs[21]={key:obs[21][mapping[key[0]]+key[1]] for key in VERTICES+MIDS};chart=cube_chart();new={};fits={}
    for c in old:
        xyz=np.array([chart[k] for k in VERTICES]);uv=np.array([obs[c][k] for k in VERTICES]);k=old[c][0]
        ok,rvec,tvec=cv2.solvePnP(xyz,uv,k,None,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:raise RuntimeError('Cube extrinsic fit failed')
        r=cv2.Rodrigues(rvec)[0];t=tvec.ravel();new[c]=(k,r,t)
        fits[str(c)]={'outer_training_mean_reprojection_px':float(np.linalg.norm(projection(xyz,new[c])-uv,axis=1).mean()),
                      'held_midpoint_mean_reprojection_px':float(np.linalg.norm(projection(np.array([chart[k] for k in MIDS]),new[c])-np.array([obs[c][k] for k in MIDS]),axis=1).mean()),
                      'rotation':r.tolist(),'translation_m':t.tolist()}
    original_points=base['primary_pair17_20']['points_m'];r0,t0=rigid_registration([chart[k] for k in VERTICES],[original_points[k] for k in VERTICES])
    before_mid={str(c):float(np.linalg.norm(projection(np.array([chart[k] for k in MIDS])@r0.T+t0,old[c])-np.array([obs[c][k] for k in MIDS]),axis=1).mean()) for c in old}
    pairs={}
    for a,b in itertools.combinations(old,2):
        points,res=triangulate([obs[a][k] for k in VERTICES+MIDS],[obs[b][k] for k in VERTICES+MIDS],new[a],new[b])
        pairs[f'{a}_{b}']=metric(points,res,VERTICES+MIDS)
    board_obs=json.loads((OUT/'prime_board_detections.json').read_text());board_base=json.loads((ROOT/'prime_board_results.json').read_text());boards=[]
    for row in board_base['rows']:
        if row['status']!='evaluated':continue
        a,b=row['cameras'];f=row['frame'];left=board_obs[f'{a}_{f}']['corners'];right=np.array(board_obs[f'{b}_{f}']['corners'])
        if row['reverse180']:right=right[::-1]
        try:
            points,res=triangulate(left,right,new[a],new[b]);boards.append({'cameras':[a,b],'frame':f,'status':'evaluated','mean_reprojection_px':float(res.mean()),'lengths':lengths(points)})
        except ValueError as e:boards.append({'cameras':[a,b],'frame':f,'status':'excluded','reason':str(e)})
    summary={}
    for a,b in itertools.combinations(old,2):
        before=[r for r in board_base['rows'] if r['status']=='evaluated' and r['cameras']==[a,b]]
        after=[r for r in boards if r['status']=='evaluated' and r['cameras']==[a,b]]
        summary[f'{a}_{b}']={'before_count':len(before),'after_count':len(after),'before_edge_mae_mm':float(np.mean([r['lengths']['mae_from100mm'] for r in before])),
                           'after_edge_mae_mm':float(np.mean([r['lengths']['mae_from100mm'] for r in after])) if after else None,
                           'before_reprojection_px':float(np.mean([r['mean_reprojection_px'] for r in before])),
                           'after_reprojection_px':float(np.mean([r['mean_reprojection_px'] for r in after])) if after else None}
    checks={'all_board_rows_retained':len(boards)==sum(r['status']=='evaluated' for r in boards),
            'every_held_midpoint_camera_nonworsening':all(fits[str(c)]['held_midpoint_mean_reprojection_px']<=before_mid[str(c)]+1e-9 for c in old),
            'every_cube_half_edge_pair_nonworsening':all(v['half_edge_mae_mm']<=base['pairs'][k]['half_edge_mae_mm']+1e-9 for k,v in pairs.items()),
            'every_board_pair_reprojection_nonworsening':all(v['after_reprojection_px'] is not None and v['after_reprojection_px']<=v['before_reprojection_px']+1e-9 for v in summary.values()),
            'every_board_pair_edge_nonworsening':all(v['after_edge_mae_mm'] is not None and v['after_edge_mae_mm']<=v['before_edge_mae_mm']+1e-9 for v in summary.values())}
    result={'protocol_sha256':sha(ROOT/'CUBE_CAMERA_CORRECTION_PROTOCOL.md'),'code_sha256':sha(__file__),'cube_fits':fits,'old_rig_registered_midpoint_reprojection_px':before_mid,
            'new_pairs':pairs,'board_rows':boards,'board_summary':summary,'engineering_checks':checks,'calibration_accepted':all(checks.values()),
            'applied_to_pitching':False,'old_rig_to_cube_registration_scale':1.,'coordinate_frame':'cube chart, not original camera19',
            'scientific_stage_passed':False,'actual_athlete_gain_kmh':None}
    (ROOT/'prime_cube_correction_results.json').write_text(json.dumps(plain(result),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cube_midpoint_before':before_mid,'cube_fits':{k:{kk:vv for kk,vv in v.items() if 'reprojection' in kk} for k,v in fits.items()},
                      'cube_half_mae_before_after':{k:[base['pairs'][k]['half_edge_mae_mm'],v['half_edge_mae_mm']] for k,v in pairs.items()},'board_summary':summary,'checks':checks}))
    return result


if __name__=='__main__':run()
