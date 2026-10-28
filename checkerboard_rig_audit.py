"""Separate planar-board pose route; cube and ground-plane metric validation."""
import itertools,json
from pathlib import Path
import cv2
import numpy as np
from edge_calibration import OUT,INTRINSICS,camera_parameters,project,triangulate,lengths,serialise,fit_camera,register_cube

BOXES={1:(600,870,1000,960),2:(710,680,1030,850),3:(380,730,650,890),4:(490,420,780,550)}

def detect(i,frame=120):
    im=cv2.imread(str(OUT/f'edgertronic_static_checkerboard_{i}_f{frame}.png'))
    x,y,xx,yy=BOXES[i];g=cv2.cvtColor(im[y:yy,x:xx],cv2.COLOR_BGR2GRAY);g=cv2.resize(g,None,fx=4,fy=4)
    ok,uv=cv2.findChessboardCornersSB(g,(7,4),flags=cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY|cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        ok,uv=cv2.findChessboardCorners(g,(7,4),flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:uv=cv2.cornerSubPix(g,uv,(5,5),(-1,-1),(3,50,.001))
    if not ok:raise ValueError(f'No checkerboard camera {i}, frame {frame}')
    return uv.reshape(4,7,2)/4+[x,y]

def fit_planar(obj,uv,k,d):
    train=np.arange(len(obj))%2==0
    ok,r,t=cv2.solvePnP(obj[train],uv[train],k,d,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('Planar PnP failed')
    r,t=cv2.solvePnPRefineLM(obj[train],uv[train],k,d,r,t)
    c={'K':k,'dist':d,'R':cv2.Rodrigues(r)[0],'t':t.reshape(3)}
    return c

def run():
    source=json.loads(INTRINSICS.read_text());a=json.loads(Path('edge_target_annotations.json').read_text())
    local={i:fit_camera(np.array([a['cube_coordinates'][j] for j in obs['fit']])*a['cube_side_mm'],list(obs['fit'].values()),*camera_parameters(source,int(i))) for i,obs in a['cameras'].items()}
    # Explicitly frozen registration from round 1, also consistent with the visible paired mid-rail faces.
    turns={'1':0,'2':3,'3':1,'4':0}
    ccube={i:register_cube(c,turns[i],a['cube_side_mm']) for i,c in local.items()}
    obj=np.array([[x*100,y*100,0] for y in range(4) for x in range(7)],float)
    variants={};raw={};fit_checks={}
    for i in range(1,5):
        k,d=camera_parameters(source,i);q=detect(i);raw[str(i)]=q.tolist();variants[str(i)]=[]
        for fx,fy in itertools.product([False,True],repeat=2):
            uv=q[:,::-1] if fx else q.copy();uv=uv[::-1] if fy else uv;uv=uv.reshape(-1,2)
            c=fit_planar(obj,uv,k,d)
            if c['R'][2,2]>0 and (obj@c['R'].T+c['t'])[:,2].min()>0:
                variants[str(i)].append((f'x{int(fx)}y{int(fy)}',c,uv))
    ref=next(v for v in variants['4'] if v[0]=='x0y0')
    cube4=ccube['4']
    # Cube -> camera 4 -> checkerboard reference. Only fixes coordinate registration.
    rc=ref[1]['R'].T@cube4['R'];tc=ref[1]['R'].T@(cube4['t']-ref[1]['t'])
    corners={};cube_labels={}
    for i,obs in a['cameras'].items():
        # Map each local cube point into common cube frame via the previously explicit horizontal registration.
        loc=np.array([a['cube_coordinates'][j] for j in obs['fit']])*a['cube_side_mm']
        orig_cam=local[i]
        world=((loc@orig_cam['R'].T+orig_cam['t'])-ccube[i]['t'])@ccube[i]['R']
        corners[i]=world@rc.T+tc
        cube_labels[i]={j:tuple(np.rint(w/a['cube_side_mm']).astype(int)) for j,w in zip(obs['fit'],world)}
    cams={};selected={};scores={}
    for i in variants:
        ranking=[]
        for name,c,uv in variants[i]:
            e=project(corners[i],c)-np.array(list(a['cameras'][i]['fit'].values()))
            ranking.append((float(np.sqrt(np.mean(np.sum(e**2,1)))),name,c,uv))
        ranking.sort(key=lambda x:x[0]);score,name,c,uv=ranking[0];cams[i]=c;selected[i]=name;scores[i]=[{'variant':v[1],'cube_projection_rms_px':v[0]} for v in ranking]
        e=np.linalg.norm(project(obj,c)-uv,axis=1);idx=np.arange(len(obj));fit_checks[i]={'train_rms_px':float(np.sqrt(np.mean(e[idx%2==0]**2))),'heldout_rms_px':float(np.sqrt(np.mean(e[idx%2==1]**2))),'positive_depth':bool(np.all((obj@c['R'].T+c['t'])[:,2]>0))}
    points={};reproj={}
    for label in itertools.product(range(2),repeat=3):
        observations=[];used=[]
        for i in cams:
            for j,l in cube_labels[i].items():
                if l==label:observations.append(a['cameras'][i]['fit'][j]);used.append(cams[i])
        if len(used)>=2:
            x,e,z=triangulate(observations,used);points[label]=x;reproj[str(label)]={'views':len(used),'rms_px':float(np.sqrt(np.mean(np.sum(e**2,1)))),'min_depth_mm':min(z)}
    edges=[]
    for p,q in itertools.combinations(points,2):
        if sum(abs(x-y) for x,y in zip(p,q))==1:
            ll=float(np.linalg.norm(points[p]-points[q]));edges.append({'labels':[p,q],'length_mm':ll,'error_percent':100*(ll/a['cube_side_mm']-1)})
    plane=[]
    for n in [2,4]:
        for ids in itertools.combinations(cams,n):
            xyz=[];ee=[];zz=[]
            for name in ['O','L','S']:
                x,e,z=triangulate([a['cameras'][i]['plane'][name] for i in ids],[cams[i] for i in ids]);xyz.append(x);ee.extend(e);zz.extend(z)
            ll=lengths(xyz);plane.append({'cameras':ids,'lengths_mm':ll,'error_percent':{k:100*(v/a['plane_distances_mm'][k]-1) for k,v in ll.items()},'rms_px':float(np.sqrt(np.mean(np.sum(np.array(ee)**2,1)))),'min_depth_mm':min(zz)})
    result={'selected_grid_reversals':selected,'grid_registration_scores':scores,'checkerboard_projection':fit_checks,'cube_reprojection':reproj,'cube_edges':edges,'plane_checks':plane,'cameras':{i:serialise(c) for i,c in cams.items()},'detections':raw,'engineering_pass':bool(all(v['train_rms_px']<=2 and v['heldout_rms_px']<=2 and v['positive_depth'] for v in fit_checks.values()) and all(abs(v['error_percent'])<=1 for v in edges) and all(max(abs(e) for e in v['error_percent'].values())<=1 and v['rms_px']<=2 and v['min_depth_mm']>0 for v in plane)),'scientific_data_gate_pass':False,'speed_error_reduction_kmh':None}
    (OUT/'checkerboard-rig-results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['selected_grid_reversals','checkerboard_projection','cube_edges','plane_checks']},indent=2))
    return result

if __name__=='__main__':run()
