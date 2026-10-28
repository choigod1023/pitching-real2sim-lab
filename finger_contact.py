"""Minimal synthetic sphere contact kernel, NOT anatomical pitching validation.

Row 0 is a ball; row 1 is a free sphere proxy, not a finger segment.
Stiffness, damping, friction and proxy dimensions are engineering assumptions.
This is not a reproduction of Koike's multi-segment finger model.
State rows are [position(3), velocity(3), world angular velocity(3)].
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContactParams:
    masses: tuple = (.145, .03)
    radii: tuple = (.0364, .008)
    stiffness: float = 1e4
    damping: float = 2.
    tangent_damping: float = 2.
    friction: float = .4
    gravity: tuple = (0., 0., 0.)

    def __post_init__(self):
        for values in (self.masses, self.radii):
            if len(values) != 2 or not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError('Two finite positive masses/radii required')
        for value in (self.stiffness, self.damping, self.tangent_damping, self.friction):
            if not np.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative contact coefficients required')
        if len(self.gravity) != 3 or not np.isfinite(self.gravity).all():
            raise ValueError('Finite three-vector gravity required')

    @property
    def inertia(self):
        return .4*np.asarray(self.masses)*np.asarray(self.radii)**2


def _state(state):
    state = np.asarray(state, dtype=float)
    if state.shape != (2, 9) or not np.isfinite(state).all():
        raise ValueError('Finite state with shape (2,9) required')
    return state


def contact(state, params):
    state = _state(state)
    x, v, w = state[:, :3], state[:, 3:6], state[:, 6:9]
    offset = x[0]-x[1]
    distance = np.linalg.norm(offset)
    if distance <= 1e-12:
        raise ValueError('Coincident sphere centers have undefined contact normal')
    n = offset/distance
    delta = max(0., sum(params.radii)-distance)
    # Same world application point on both bodies preserves angular momentum.
    point = .5*(x[0]-params.radii[0]*n+x[1]+params.radii[1]*n)
    arms = point-x
    u = v[0]+np.cross(w[0], arms[0])-v[1]-np.cross(w[1], arms[1])
    vn = float(u@n)
    ut = u-vn*n
    speed = np.linalg.norm(ut)
    fn = max(0., params.stiffness*delta-params.damping*vn) if delta > 0 else 0.
    ft = -min(params.friction*fn, params.tangent_damping*speed)*ut/speed if speed > 0 else np.zeros(3)
    force = fn*n+ft
    # Includes positive loss when tensile normal damping would be clipped.
    loss = (params.stiffness*delta-fn)*vn-float(ft@ut) if delta > 0 else 0.
    return {'force_on_ball': force, 'torque_ball': np.cross(arms[0], force),
            'torque_proxy': np.cross(arms[1], -force), 'penetration_m': delta,
            'normal_force_n': fn, 'tangent_force_n': ft, 'dissipation_w': loss,
            'point_m': point, 'relative_velocity_mps': u}


def derivative(state, params):
    state = _state(state)
    info = contact(state, params)
    result = np.zeros((2, 9))
    result[:, :3] = state[:, 3:6]
    result[:, 3:6] = np.asarray(params.gravity)
    result[0, 3:6] += info['force_on_ball']/params.masses[0]
    result[1, 3:6] -= info['force_on_ball']/params.masses[1]
    result[0, 6:9] = info['torque_ball']/params.inertia[0]
    result[1, 6:9] = info['torque_proxy']/params.inertia[1]
    return result


def rk4_step(state, dt, params):
    state = _state(state)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite time step required')
    a = derivative(state, params)
    b = derivative(state+.5*dt*a, params)
    c = derivative(state+.5*dt*b, params)
    d = derivative(state+dt*c, params)
    return _state(state+dt*(a+2*b+2*c+d)/6)


def totals(state, params):
    state = _state(state)
    x, v, w = state[:, :3], state[:, 3:6], state[:, 6:9]
    masses = np.asarray(params.masses)
    momentum = masses[:, None]*v
    kinetic = .5*np.sum(masses*np.sum(v*v, axis=1)+params.inertia*np.sum(w*w, axis=1))
    potential = -float(np.sum(masses*(x@np.asarray(params.gravity))))
    delta = max(0., sum(params.radii)-np.linalg.norm(x[0]-x[1]))
    elastic = .5*params.stiffness*delta**2
    return {'kinetic_j': float(kinetic), 'potential_j': potential, 'elastic_j': float(elastic),
            'energy_j': float(kinetic+potential+elastic), 'linear_momentum': momentum.sum(axis=0),
            'angular_momentum': (np.cross(x, momentum)+params.inertia[:, None]*w).sum(axis=0)}
