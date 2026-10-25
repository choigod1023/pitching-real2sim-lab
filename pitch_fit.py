"""Constrained direction retargeting, not subject-scaled or force-driven pitching."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from muscle_model_check import load_arm
from pilot import regular_time, vector

NAMES = ['elv_angle_r', 'shoulder_elv_r', 'shoulder_rot_r', 'elbow_flexion_r']


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate segment or chest frame')
    return v/norm


def chest_frames(right, left, proximal, distal):
    up = normalize(proximal-distal)
    side = right-left
    side = normalize(side-np.sum(side*up, axis=-1, keepdims=True)*up)
    anterior = np.cross(up, side)
    return np.stack([anterior, up, side], axis=-1)


class ArmMap:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ids = np.array([model.joint(name).id for name in NAMES])
        self.address = model.jnt_qposadr[self.ids]
        self.lower = model.jnt_range[self.ids, 0].copy()
        self.upper = model.jnt_range[self.ids, 1].copy()
        self.constraints = []
        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                raise ValueError('Unsupported model equality')
            dep, src = model.eq_obj1id[i], model.eq_obj2id[i]
            coeff = model.eq_data[i, :5]
            if src < 0 or np.any(coeff[2:] != 0):
                raise ValueError('Expected affine two-joint equalities')
            self.constraints.append((model.jnt_qposadr[dep], model.jnt_qposadr[src], coeff[0], coeff[1]))
            if src in self.ids and model.jnt_limited[dep] and abs(coeff[1]) > 1e-12:
                k = list(self.ids).index(src)
                limits = sorted((model.jnt_range[dep]-coeff[0])/coeff[1])
                self.lower[k] = max(self.lower[k], limits[0])
                self.upper[k] = min(self.upper[k], limits[1])
        mujoco.mj_forward(model, self.data)
        self.root_rotation = self.data.xmat[model.body('myoarm_r_root').id].reshape(3,3).copy()
        self.body_ids = [model.body(name).id for name in ['humerus_r','ulna_r','lunate_r']]

    def set_pose(self, angles):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.address] = angles
        # This pinned model has direct, acyclic affine dependencies.
        for dep, src, offset, slope in self.constraints:
            self.data.qpos[dep] = offset+slope*self.data.qpos[src]
        mujoco.mj_forward(self.model, self.data)
        shoulder, elbow, wrist = self.data.xpos[self.body_ids]
        return np.stack([normalize(elbow-shoulder), normalize(wrist-elbow)])

    def fit(self, targets, prior=None):
        seed = np.array([0., 1.5, 0., 1.5]) if prior is None else prior
        def residual(q):
            # Very weak continuity tie-breaker; direction residual is primary.
            return np.r_[(self.set_pose(q)-targets).ravel(), .001*(q-seed)]
        seeds = [seed]
        if prior is None:
            seeds += [np.array([a, 1.5, r, 1.5]) for a in [-1.,1.] for r in [-1.,1.]]
        solutions = [least_squares(residual, np.clip(s,self.lower+1e-8,self.upper-1e-8),
                                   bounds=(self.lower,self.upper), max_nfev=100,
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8) for s in seeds]
        best = min(solutions, key=lambda r: np.sum(r.fun**2))
        actual = self.set_pose(best.x)
        angle_error = np.degrees(np.arccos(np.clip(np.sum(actual*targets,axis=1),-1,1)))
        return best.x, angle_error, best.success

    def fit_sequence(self, targets, initial, weight):
        """Joint constrained solve with dimensionless second-difference penalty.

        Weight is an engineering sensitivity setting, not a physiological prior.
        """
        n = len(targets)
        if n < 3 or not np.isfinite(weight) or weight <= 0:
            raise ValueError('At least three frames and positive temporal weight required')
        sparsity = lil_matrix((n*6+(n-2)*4,n*4),dtype=int)
        for i in range(n):
            sparsity[i*6:(i+1)*6,i*4:(i+1)*4] = 1
        for i in range(n-2):
            for j in range(4):
                sparsity[n*6+i*4+j,[i*4+j,(i+1)*4+j,(i+2)*4+j]] = 1
        def residual(flat):
            q = flat.reshape(n,4)
            fit = np.array([self.set_pose(a)-b for a,b in zip(q,targets)])
            return np.r_[fit.ravel(),weight*np.diff(q,n=2,axis=0).ravel()]
        result = least_squares(residual,initial.ravel(),
                               bounds=(np.tile(self.lower,n),np.tile(self.upper,n)),
                               jac_sparsity=sparsity.tocsr(),max_nfev=150,
                               ftol=1e-7,xtol=1e-7,gtol=1e-7)
        return result.x.reshape(n,4),bool(result.success)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent.parent/'openbiomechanics/baseball_pitching/data')
    parser.add_argument('--vendor', type=Path, default=Path('artifacts/vendor'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/pitch-fit'))
    parser.add_argument('--pitch')
    parser.add_argument('--temporal-weight', type=float, default=0.)
    parser.add_argument('--diagnostic-rotation-span-deg', type=float,
                        help='Symmetric shoulder rotation range for sensitivity only; not a validated anatomical correction')
    args = parser.parse_args()
    if not np.isfinite(args.temporal_weight) or args.temporal_weight < 0:
        raise ValueError('Temporal weight must be finite and nonnegative')
    meta = pd.read_csv(args.data_root/'metadata.csv')
    poi = pd.read_csv(args.data_root/'poi/poi_metrics.csv')
    meta = meta.merge(poi[['session_pitch','p_throws']], on='session_pitch', validate='one_to_one')
    eligible = meta[meta.p_throws=='R'].sort_values(['user','session_pitch'])
    if args.pitch:
        eligible = eligible[eligible.session_pitch==args.pitch]
    if eligible.empty:
        raise ValueError('No matching right-handed pitch')
    m = eligible.iloc[0]
    all_data = pd.read_csv(args.data_root/'full_sig/landmarks.csv')
    g = all_data[all_data.session_pitch==m.session_pitch].sort_values('time').copy()
    nominal = regular_time(g)
    start, release = float(g.fp_10_time.iloc[0]), float(g.BR_time.iloc[0])
    keep = (nominal >= start) & (nominal <= release)
    g = g.loc[keep].copy()
    times = nominal[keep]
    if len(g)<10:
        raise ValueError('Insufficient foot-contact to release samples')
    shoulder, elbow, wrist = [vector(g,n+'_jc') for n in ['shoulder','elbow','wrist']]
    frames = chest_frames(shoulder,vector(g,'glove_shoulder_jc'),vector(g,'thorax_prox'),vector(g,'thorax_dist'))
    model, compile_warnings = load_arm(args.vendor)
    if args.diagnostic_rotation_span_deg is not None:
        if not 0 < args.diagnostic_rotation_span_deg <= 180:
            raise ValueError('Diagnostic shoulder rotation span must be in (0,180] degrees')
        span = np.radians(args.diagnostic_rotation_span_deg)
        model.joint('shoulder_rot_r').range[:] = [-span,span]
    mapping = ArmMap(model)
    observed = np.stack([normalize(elbow-shoulder),normalize(wrist-elbow)],axis=1)
    local = np.einsum('nji,nkj->nki',frames,observed)
    targets = np.einsum('ij,nkj->nki',mapping.root_rotation,local)
    poses, errors, converged, contacts = [], [], [], []
    prior = None
    for target in targets:
        prior, error, ok = mapping.fit(target,prior)
        poses.append(mapping.data.qpos.copy()); errors.append(error)
        converged.append(bool(ok));contacts.append(mapping.data.ncon)
    poses = np.array(poses); errors = np.array(errors)
    if args.temporal_weight:
        fitted, ok = mapping.fit_sequence(targets,poses[:,mapping.address],args.temporal_weight)
        poses,errors,contacts = [],[],[]
        for q,target in zip(fitted,targets):
            actual=mapping.set_pose(q)
            poses.append(mapping.data.qpos.copy())
            errors.append(np.degrees(np.arccos(np.clip(np.sum(actual*target,axis=1),-1,1))))
            contacts.append(mapping.data.ncon)
        poses=np.array(poses);errors=np.array(errors);converged=[ok]*len(poses)
    limited = model.jnt_limited.astype(bool)
    violation = np.maximum(model.jnt_range[:,0]-poses,poses-model.jnt_range[:,1])[:,limited]
    report = {'stage':'chest_relative_direction_retargeting', 'pitch':str(m.session_pitch),
              'athlete':int(m.user), 'frames':len(g), 'start_event':'fp_10_time', 'end_event':'BR_time',
              'independent_fitted_joints':NAMES, 'forearm_rotation_and_hand':'held at model default; not identified',
              'mean_direction_error_deg':errors.mean(axis=0).tolist(),
              'max_direction_error_deg':errors.max(axis=0).tolist(),
              'p95_direction_error_deg':np.percentile(errors,95,axis=0).tolist(),
              'max_joint_limit_violation_rad':float(max(0,violation.max())),
              'fitted_joint_bound_frame_counts':{name:int(np.count_nonzero((poses[:,addr]-lo<1e-5)|(hi-poses[:,addr]<1e-5)))
                   for name,addr,lo,hi in zip(NAMES,mapping.address,mapping.lower,mapping.upper)},
              'diagnostic_rotation_span_deg':args.diagnostic_rotation_span_deg,
              'temporal_second_difference_weight':args.temporal_weight,
              'range_change_anatomically_validated':False,
              'optimizer_success_fraction':float(np.mean(converged)),
              'frames_with_contact':int(np.count_nonzero(contacts)),
              'observed_median_segment_lengths_m':[float(np.median(np.linalg.norm(v,axis=1))) for v in [elbow-shoulder,wrist-elbow]],
              'compile_warnings':compile_warnings,
              'subject_geometry_scaled':False, 'moving_chest_dynamics':False,
              'force_driven_tracking':False, 'ball_model':False,
              'scope':'Direction fit on a fixed generic model. Not measured position/velocity reproduction; not a ball-speed prediction.'}
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'trajectory.npz',qpos=poses,time=times,target_directions=targets,chest_frames=frames,observed_shoulder=shoulder,
             model_joint_ranges=model.jnt_range.copy())
    (args.output/'fit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
