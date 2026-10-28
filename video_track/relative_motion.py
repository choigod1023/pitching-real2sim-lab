"""Observed hand/ball velocity decomposition, NOT an intervention predictor.

    b = h + R r
    v_b = v_h + omega_world x (R r) + R r_dot

Position/rotation interpolants are fitted to the observed trajectories, without
radar fitting or smoothing. The full decomposition reconstructs observed ball
velocity by algebra; its zero closure error is not prediction accuracy. A constant
local vector fitted to early frames is separately evaluated on later frames.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from hand_ball_observation import relative_position
from relative_ball_state import ball_velocity
from .theia import load_hand_pose

FIT_RANGE = (956, 963)
EVAL_RANGE = (964, 973)  # analysis window, NOT confirmed contact frames
FPS = 360.0


def motion_curves(time, hand_m, rotations, ball_m):
    """Continuous observed poses; reject missing/degenerate inputs, never extrapolate."""
    time, hand_m, rotations, ball_m = map(lambda a: np.asarray(a, float), (time, hand_m, rotations, ball_m))
    n = time.size
    if (time.ndim != 1 or n < 5 or hand_m.shape != (n, 3) or ball_m.shape != (n, 3)
            or rotations.shape != (n, 3, 3) or not all(np.isfinite(a).all() for a in (time, hand_m, rotations, ball_m))
            or np.any(np.diff(time) <= 0)):
        raise ValueError("Need increasing times and at least five finite paired poses")
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4, rtol=0)):
        raise ValueError("Rotations must be proper orthogonal matrices")
    hand = CubicSpline(time, hand_m, extrapolate=False)
    ball = CubicSpline(time, ball_m, extrapolate=False)
    rot = RotationSpline(time, Rotation.from_matrix(rotations))

    def evaluate(t):
        t = np.atleast_1d(np.asarray(t, float))
        if not np.isfinite(t).all() or np.any(t < time[0]) or np.any(t > time[-1]):
            raise ValueError("No extrapolation outside observed interval")
        h, b, R = hand(t), ball(t), rot(t).as_matrix()
        vh, vb = hand(t, 1), ball(t, 1)
        omega = np.einsum("nij,nj->ni", R, rot(t, 1))
        lever = b - h
        r = relative_position(b, h, R)
        v_rotation = np.cross(omega, lever)
        # Derived from the same observations, explicitly not an independent prediction.
        v_relative = vb - vh - v_rotation
        r_dot = np.einsum("nji,nj->ni", R, v_relative)
        return dict(hand_m=h, ball_m=b, rotation=R, local_ball_m=r,
                    hand_velocity_mps=vh, ball_velocity_mps=vb, omega_world_rad_s=omega,
                    rotation_velocity_mps=v_rotation, relative_velocity_world_mps=v_relative,
                    relative_velocity_local_mps=r_dot)

    return evaluate


def constant_grip(state, local_vector_m):
    """Rigid local-vector baseline, using observed hand motion only at evaluation."""
    r0 = np.asarray(local_vector_m, float)
    if r0.shape != (3,) or not np.isfinite(r0).all():
        raise ValueError("Need a finite 3D local vector")
    lever = np.einsum("nij,j->ni", state["rotation"], r0)
    return (state["hand_m"] + lever,
            state["hand_velocity_mps"] + np.cross(state["omega_world_rad_s"], lever))


def window_speed(time, positions, width):
    """Centred local linear-fit speed, sensitivity only; endpoints remain NaN."""
    if width < 3 or width % 2 != 1:
        raise ValueError("Window must be odd and at least three")
    half, out = width // 2, np.full(len(time), np.nan)
    for i in range(half, len(time) - half):
        t = time[i-half:i+half+1] - time[i]
        B = np.column_stack((np.ones(width), t))
        v = np.linalg.lstsq(B, positions[i-half:i+half+1], rcond=None)[0][1]
        out[i] = np.linalg.norm(v) * 3.6
    return out


def analyze(frames, hand_m, rotations, ball_m, fps=FPS, fit_range=FIT_RANGE, eval_range=EVAL_RANGE):
    frames = np.asarray(frames)
    if (frames.ndim != 1 or not np.isfinite(frames).all()
            or not np.equal(frames, np.floor(frames)).all() or not np.all(np.diff(frames) == 1)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Require consecutive integer frames and positive fps")
    time = (frames - frames[0]) / fps
    fit = (frames >= fit_range[0]) & (frames <= fit_range[1])
    held = (frames >= eval_range[0]) & (frames <= eval_range[1])
    if (fit_range[0] > fit_range[1] or eval_range[0] > eval_range[1]
            or fit_range[1] >= eval_range[0] or fit.sum() != fit_range[1]-fit_range[0]+1
            or held.sum() != eval_range[1]-eval_range[0]+1):
        raise ValueError("Need complete disjoint chronological fit/evaluation windows")
    curves = motion_curves(time, hand_m, rotations, ball_m)
    state = curves(time)
    r0 = state["local_ball_m"][fit].mean(axis=0)
    fixed_p, fixed_v = constant_grip(state, r0)
    state.update(fixed_ball_m=fixed_p, fixed_ball_velocity_mps=fixed_v)
    vb = state["ball_velocity_mps"]
    speed = np.linalg.norm(vb, axis=1)
    direction = np.divide(vb, speed[:, None], out=np.zeros_like(vb), where=speed[:, None] > 1e-10)
    components = [state[k] for k in ("hand_velocity_mps", "rotation_velocity_mps", "relative_velocity_world_mps")]
    projected = np.stack([np.sum(v * direction, axis=1) * 3.6 for v in components], axis=1)
    # An independent finite difference of the continuous local-vector curve checks
    # rotation convention/sign; neither this nor algebraic closure validates the data.
    eps = 1e-6
    fd = (curves(time[1:-1]+eps)["local_ball_m"] - curves(time[1:-1]-eps)["local_ball_m"]) / (2*eps)
    fd_error = np.linalg.norm(fd-state["relative_velocity_local_mps"][1:-1], axis=1)
    poserr = np.linalg.norm(fixed_p-np.asarray(ball_m), axis=1)*1000
    verr = np.linalg.norm(fixed_v-vb, axis=1)*3.6
    for w in (3, 5, 7, 9):
        state[f"ball_speed_window{w}_kmh"] = window_speed(time, np.asarray(ball_m), w)
    composed = np.array([ball_velocity(vh, R, w, r, dr)["ball_velocity_mps"] for vh, R, w, r, dr in
                         zip(state["hand_velocity_mps"], state["rotation"], state["omega_world_rad_s"],
                             state["local_ball_m"], state["relative_velocity_local_mps"])])
    summary = dict(fit_frames=list(fit_range), evaluation_frames=list(eval_range),
                   fixed_local_vector_mm=(r0*1000).tolist(),
                   fixed_heldout_position_rmse_mm=float(np.sqrt(np.mean(poserr[held]**2))),
                   fixed_heldout_velocity_vector_mae_kmh=float(verr[held].mean()),
                   fixed_heldout_speed_mae_kmh=float(np.abs(np.linalg.norm(fixed_v[held], axis=1)-speed[held]).mean()*3.6),
                   formula_closure_max_mps=float(np.max(np.linalg.norm(composed-vb, axis=1))),
                   relative_derivative_fd_max_mps=float(fd_error.max()))
    rows = []
    for i, f in enumerate(frames):
        row = dict(frame=int(f), local_ball_mm=(state["local_ball_m"][i]*1000).tolist(),
                   distance_mm=float(np.linalg.norm(state["local_ball_m"][i])*1000),
                   fixed_position_error_mm=float(poserr[i]),
                   observed_spline_speed_kmh=float(speed[i]*3.6),
                   fixed_speed_kmh=float(np.linalg.norm(fixed_v[i])*3.6),
                   relative_speed_norm_kmh=float(np.linalg.norm(components[2][i])*3.6),
                   signed_speed_components_kmh=dict(zip(("hand_translation", "hand_rotation", "relative_ball_motion"), projected[i].tolist())))
        row["window_speeds_kmh"] = {str(w): (float(state[f"ball_speed_window{w}_kmh"][i])
                                                    if np.isfinite(state[f"ball_speed_window{w}_kmh"][i]) else None) for w in (3, 5, 7, 9)}
        rows.append(row)
    return summary, rows, state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", type=Path, required=True)
    ap.add_argument("--theia-c3d", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    track = json.loads(a.track.read_text())
    frames = np.array(sorted(int(f) for f in track))
    ball = np.array([track[str(f)]["X_mm"] for f in frames]) / 1000
    alignment = json.loads(a.alignment.read_text())
    nominal = alignment["c3d_frame_minus_video_frame"]
    pose = load_hand_pose(a.theia_c3d, a.alignment)
    results = {}
    a.out.mkdir(parents=True, exist_ok=True)
    for offset in (0, 1, 2):
        summary, rows, state = analyze(frames, *pose(offset, frames), ball)
        results[str(offset)] = dict(summary=summary, frames=rows)
        np.savez_compressed(a.out / f"relative_offset{offset}.npz", frames=frames, **state)
    report = dict(trial=a.theia_c3d.stem, segment="l_hand_4X4", fps=FPS,
                  nominal_c3d_minus_video_frame=nominal,
                  provenance={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in
                              (("track_sha256", a.track), ("theia_sha256", a.theia_c3d), ("alignment_sha256", a.alignment))},
                  alignment_note=alignment.get("note"),
                  scope="Observed kinematic decomposition, not ball/contact dynamics or independent prediction",
                  full_reconstruction_is_algebraic_identity=True, radar_used=False,
                  derivatives_use_future_observations=True, real_time_predictor=False,
                  release_identified=False, finger_acceleration_identified=False,
                  transfer_to_3034_validated=False, candidate_optimization_ready=False,
                  derivative_method="Unsmoothed CubicSpline positions + RotationSpline orientation; 3/5/7/9-frame speed sensitivity reported",
                  window_scope="Fixed diagnostic windows, not confirmed hand contact or release",
                  results_by_c3d_offset=results)
    (a.out / "relative_motion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    compact = {k: v for k, v in report.items() if k != "results_by_c3d_offset"}
    compact["results_by_c3d_offset"] = {}
    for offset, value in results.items():
        by_frame = {r["frame"]: r for r in value["frames"]}
        displacement = np.array(by_frame[974]["local_ball_mm"]) - by_frame[970]["local_ball_mm"]
        compact["results_by_c3d_offset"][offset] = dict(
            summary=value["summary"], local_displacement_970_to974_mm=float(np.linalg.norm(displacement)),
            diagnostic_frames=[by_frame[f] for f in (970, 973, 974, 975)])
    (a.out / "relative_motion_summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for offset, value in results.items():
        print(json.dumps(dict(offset=offset, **value["summary"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
