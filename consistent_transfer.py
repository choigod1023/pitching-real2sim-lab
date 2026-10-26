"""Fit relative ball motion in the same hand frame used for prediction.

This remains an optical development experiment, not a contact/athlete model.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from frozen_transfer import fit_chain, run


def fit_consistent_chain(times, ball, origin, rotation):
    model = fit_chain(times, ball, origin, rotation)
    t = np.asarray(times, float) - model['t0']
    design = np.c_[np.ones(len(t)), t]
    fitted_rotation = model['r0'] @ Rotation.from_rotvec(t[:, None] * model['omega']).as_matrix()
    fitted_origin = design @ model['origin']
    relative = np.einsum('nji,nj->ni', fitted_rotation, np.asarray(ball) - fitted_origin)
    model['relative'] = np.linalg.lstsq(design, relative, rcond=None)[0]
    return model


if __name__ == '__main__':
    run(fit_consistent_chain, 'consistent-transfer', 'CONSISTENT_TRANSFER_PROTOCOL.md')
