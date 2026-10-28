"""Ball velocity with relative hand motion; not an estimator of unobserved slip."""
import numpy as np


def rotation_matrix(value):
    r=np.asarray(value,dtype=float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1.,atol=1e-8):
        raise ValueError('Expected a proper hand-to-world rotation')
    return r


def vec(value):
    v=np.asarray(value,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Expected finite 3-vector')
    return v


def local_offset(wrist_world,rotation,ball_world):
    return rotation_matrix(rotation).T@(vec(ball_world)-vec(wrist_world))


def ball_velocity(wrist_velocity,rotation,omega_world,offset_local,relative_velocity_local):
    """v_b = v_w + omega_world cross (R r) + R r_dot.

    relative_velocity_local must be observed or predicted by another model.
    Supplying observed future ball motion is reconstruction, not prediction.
    The rigid attachment baseline is explicitly returned for comparison.
    """
    r=rotation_matrix(rotation)
    rigid=vec(wrist_velocity)+np.cross(vec(omega_world),r@vec(offset_local))
    relative=r@vec(relative_velocity_local)
    return {'rigid_velocity_mps':rigid,'relative_velocity_mps':relative,'ball_velocity_mps':rigid+relative}
