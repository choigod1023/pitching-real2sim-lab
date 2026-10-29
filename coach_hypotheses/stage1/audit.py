"""Outcome-free dimensional/sign checks; never selects a predictor by velocity."""
import numpy as np
import pandas as pd


def knee_sign_audit(data):
    angles = pd.read_csv(data / 'full_sig/joint_angles.csv', usecols=[
        'session_pitch', 'time', 'lead_knee_angle_x', 'fp_10_time', 'BR_time'])
    velocities = pd.read_csv(data / 'full_sig/joint_velos.csv', usecols=[
        'session_pitch', 'time', 'lead_knee_velo_x'])
    joined = angles.merge(velocities, on=['session_pitch', 'time'], validate='one_to_one')
    records = []
    for key, frame in joined.groupby('session_pitch', sort=True):
        time = frame.time.to_numpy(float)
        if np.any(np.diff(time) <= 0):
            raise ValueError('invalid_sign_audit_time')
        extension = -np.gradient(frame.lead_knee_angle_x.to_numpy(float), time)
        provided = frame.lead_knee_velo_x.to_numpy(float)
        mask = (time >= frame.fp_10_time.iloc[0]) & (time <= frame.BR_time.iloc[0]) & np.isfinite(extension) & np.isfinite(provided)
        if mask.sum() < 11:
            continue
        records.append({'session_pitch': key,
                        'r_raw_velocity_vs_negative_angle_derivative': float(np.corrcoef(provided[mask], extension[mask])[0, 1]) if np.std(extension[mask]) > 1e-8 and np.std(provided[mask]) > 1e-8 else None,
                        'mae_raw_vs_extension_dps': float(np.mean(np.abs(provided[mask] - extension[mask]))),
                        'mae_negated_raw_vs_extension_dps': float(np.mean(np.abs(-provided[mask] - extension[mask])))})
    return pd.DataFrame(records)
