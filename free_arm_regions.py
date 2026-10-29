"""Train-only multi-frame region IK, preserving the observed free-arm root motion."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from prime17_hand_alignment import Source, box_residual, project, rows_for_analysis
from physical_release_comparison import launch_states
from hand_speed_derivative import metrics_for_state

ROOT = Path(__file__).resolve().parent
KNOTS = np.array([940., 950., 960.])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(x):
    x = np.asarray(x, float)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm < 1e-12):
        raise ValueError('Invalid direction')
    return x/norm


def minimal_rotation(a, b):
    a, b = unit(a), unit(b)
    cross = np.cross(a, b); sine = np.linalg.norm(cross); cosine = np.clip(a@b, -1, 1)
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        axis = unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))
        return Rotation.from_rotvec(np.pi*axis).as_matrix()
    return Rotation.from_rotvec(cross/sine*np.arctan2(sine, cosine)).as_matrix()


def retarget(source_points, residuals, delta, lengths):
    """Preserve source shoulder translation; restore exact segment lengths."""
    s0, e0, w0 = np.moveaxis(np.asarray(source_points), -2, 0)
    s = s0+delta
    e = s+lengths[0]*unit(e0+residuals[..., 0, :]-s)
    w = e+lengths[1]*unit(w0+residuals[..., 1, :]-e)
    return np.stack([s, e, w], axis=-2)


def observations():
    load = lambda name: json.loads((ROOT/name).read_text(encoding='utf-8'))['observations']
    train = load('free_arm_new_regions.json')+load('counterview_regions.json')
    evaluation = rows_for_analysis()+load('long_hand_regions.json')
    key = lambda r: (r['camera'], r['frame'], r['segment'])
    if len({key(r) for r in train}) != 12 or set(map(key, train)) & set(map(key, evaluation)):
        raise ValueError('Split overlap or changed training count')
    return train, evaluation


def source_points(source, frames):
    return np.stack([source.origins[name](np.asarray(frames)+1.) for name in ['l_uarm','l_larm','l_hand']], axis=-2)


class Reconstruction:
    def __init__(self, source, delta, displacements, lengths):
        self.source, self.delta, self.lengths = source, np.asarray(delta), np.asarray(lengths)
        self.spline = CubicSpline(KNOTS, np.asarray(displacements), bc_type='clamped')

    def points(self, frames):
        frames = np.asarray(frames)
        residuals = self.spline(np.clip(frames, KNOTS[0], KNOTS[-1]))
        return retarget(source_points(self.source, frames), residuals, self.delta, self.lengths)

    def hand(self, frames):
        frames = np.atleast_1d(frames)
        old, new = source_points(self.source, frames), self.points(frames)
        rotation = self.source.hand(frames+1).as_matrix()
        return np.array([minimal_rotation(a[2]-a[1], b[2]-b[1])@r for a,b,r in zip(old,new,rotation)])


def fit(source, rows):
    original = source_points(source, KNOTS)
    lengths = np.median(np.linalg.norm(np.diff(original, axis=1), axis=2), axis=0)
    index = {'l_larm':1, 'l_hand':2}
    def unpack(x):
        delta, limb = x[:3]/1000., x[3:].reshape(3,2,3)/1000.
        return delta, limb
    def objective(x):
        delta, limb = unpack(x)
        loss = []
        for row in rows:
            k = int(np.flatnonzero(KNOTS == row['frame'])[0])
            point = limb[k,index[row['segment']]-1]
            uv = project(point,source.cameras[row['camera']])[0]
            loss.append(np.sum(box_residual(uv,row['box'])**2))
        displacement = limb-original[:,1:]
        return float(np.mean(loss)+.001*np.mean(np.sum((displacement/.01)**2,axis=-1))
                     +.001*np.sum((delta/.01)**2)
                     +.01*np.mean(np.sum((np.diff(displacement,axis=0)/.01)**2,axis=-1)))
    def constraints(x):
        delta, limb = unpack(x)
        shoulder = original[:,0]+delta
        # Millimetres improve conditioning without changing constraints.
        return (np.column_stack([np.linalg.norm(limb[:,0]-shoulder,axis=1)-lengths[0],
                                 np.linalg.norm(limb[:,1]-limb[:,0],axis=1)-lengths[1]])*1000).ravel()
    x0 = np.r_[np.zeros(3),original[:,1:].ravel()*1000]
    lower, upper = x0-200., x0+200.
    lower[:3], upper[:3] = -30., 30.
    fitted = minimize(objective,x0,method='SLSQP',bounds=list(zip(lower,upper)),
                      constraints={'type':'eq','fun':constraints},options={'maxiter':2000,'ftol':1e-10})
    delta, limb = unpack(fitted.x)
    model = Reconstruction(source,delta,limb-original[:,1:],lengths)
    info = {'success':bool(fitted.success),'message':str(fitted.message),'iterations':int(fitted.nit),
            'objective_before':objective(x0),'objective_after':objective(fitted.x),
            'max_constraint_error_mm':float(np.max(np.abs(constraints(fitted.x)))),
            'root_translation_mm':(delta*1000).tolist(),'segment_lengths_mm':(lengths*1000).tolist(),
            'knot_displacements_mm':((limb-original[:,1:])*1000).tolist(),
            'knots':KNOTS.tolist()}
    return model,info


def summary(source, model, rows, margin=0):
    distances=[]
    for row in rows:
        points = source_points(source,row['frame']) if model is None else model.points(row['frame'])
        point=points[{'l_larm':1,'l_hand':2}[row['segment']]]
        uv=project(point,source.cameras[row['camera']])[0]
        distances.append(float(np.linalg.norm(box_residual(uv,row['box'],margin))))
    distances=np.array(distances)
    stat=lambda x:{'count':len(x),'mean_outside_px':float(x.mean()),'max_outside_px':float(x.max()),'inside':int((x<1e-9).sum())}
    return {**stat(distances),'per_camera':{str(c):stat(distances[[r['camera']==c for r in rows]]) for c in sorted({r['camera'] for r in rows})},
            'region_not_point_accuracy':True}


def ball_metric(source, model):
    condition=next(r for r in source.observation['conditions'] if r['threshold']==80 and r['c3d_frame_offset']==1)
    ball=np.array([r['ball_rig_m'] for r in condition['positions']]);frames=np.arange(973,977)
    if model is None:
        origin=source.origins['l_hand'](frames+1);rotation=source.hand(frames+1).as_matrix()
    else:
        origin=model.points(frames)[:,2];rotation=model.hand(frames)
    states=launch_states((frames-973)/source.observation['rate_hz'],ball[:4],origin,rotation)
    return metrics_for_state(*states['B'],source.gravity,ball[4:],float(source.observation['rate_hz']),62.7*1.609344)


def run():
    freeze=json.loads((ROOT/'free_arm_annotation_freeze.json').read_text(encoding='utf-8'))
    for path,digest in freeze['sha256'].items():
        if sha(path)!=digest: raise ValueError('Frozen input changed: '+path)
    source=Source();train,test=observations();model,solver=fit(source,train)
    models={}
    for name,m in [('source',None),('region_reconstruction',model)]:
        models[name]={'training':summary(source,m,train),'evaluation':summary(source,m,test),
                      'boundary_sensitivity':{str(s):summary(source,m,test,s) for s in [-3,0,3]},
                      'B_ball_evaluation':ball_metric(source,m)}
    before,after=models['source'],models['region_reconstruction']
    frames=np.linspace(940,979,391);points=model.points(frames);original=source_points(source,frames)
    errors=np.abs(np.linalg.norm(np.diff(points,axis=1),axis=2)-model.lengths)
    checks={'solver_success':solver['success'],'root_not_at_boundary':bool(np.max(np.abs(model.delta))<.03-1e-5),
            'lengths_preserved':bool(errors.max()<1e-10),
            'evaluation_mean_nonworsening':after['evaluation']['mean_outside_px']<=before['evaluation']['mean_outside_px']+1e-9,
            'all_evaluation_cameras_nonworsening':all(after['evaluation']['per_camera'][c]['mean_outside_px']<=v['mean_outside_px']+1e-9 for c,v in before['evaluation']['per_camera'].items()),
            'boundary_sensitivity_nonworsening':all(after['boundary_sensitivity'][str(s)]['mean_outside_px']<=before['boundary_sensitivity'][str(s)]['mean_outside_px']+1e-9 for s in [-3,0,3]),
            'ball_three_metrics_nonworsening':all(after['B_ball_evaluation'][k]<=before['B_ball_evaluation'][k]+1e-9 for k in ['mean_absolute_speed_discrepancy_kmh','mean_velocity_vector_discrepancy_kmh','position_rmse_mm'])}
    result={'reference':'OBP_movement_BaseballThrow_001','protocol':'FREE_ARM_REGION_PROTOCOL.md','solver':solver,'models':models,
            'training_rows':train,'evaluation_rows':test,'engineering_checks':checks,'engineering_candidate_accepted':all(checks.values()),
            'length_error_max_mm':float(errors.max()*1000),
            'source_position_difference_max_mm':(np.linalg.norm(points-original,axis=2).max(axis=0)*1000).tolist(),
            'source_position_rmse_mm':(np.sqrt(np.mean(np.sum((points-original)**2,axis=2),axis=0))*1000).tolist(),
            'freeze':freeze,'code_sha256':sha(__file__),'actual_athlete_gain_kmh':None,'force_work_change':None,
            'anatomical_accuracy_verified':False,'scientific_stage_passed':False,
            'interpretation':'Development-only approximate visible-region reconstruction. B includes train-observed ball positions; not body-only launch prediction or a posture intervention.'}
    (ROOT/'free_arm_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'solver':solver,'models':{n:{'train':m['training']['mean_outside_px'],'eval':m['evaluation']['mean_outside_px'],'ball':m['B_ball_evaluation']} for n,m in models.items()},'checks':checks}))
    return result


if __name__=='__main__':run()
