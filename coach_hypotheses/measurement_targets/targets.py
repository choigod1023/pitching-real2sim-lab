"""Condition existing stage2 inputs without fitting to recorded outcomes."""
from itertools import combinations
from math import factorial

import numpy as np

from ..stage2.uncertainty import speeds

FACTORS = ('distance', 'release', 'filter', 'relative_motion')
SEEDS = (20261029, 20261030)
N = 131072


def conditions():
    rows = []
    for mask in range(16):
        fixed = tuple(name for i, name in enumerate(FACTORS) if mask & (1 << i))
        rows.append({'name': f'fixed_{mask:02d}', 'fixed': fixed,
                     'distance_halfwidth_mm': 0. if 'distance' in fixed else 50.})
    for width in (10., 5., 2., 1.):
        for fixed in ((), FACTORS[1:]):
            label = 'alone' if not fixed else 'other_three_fixed'
            rows.append({'name': f'distance_pm{int(width)}mm_{label}',
                         'fixed': fixed, 'distance_halfwidth_mm': width})
    return rows


def common_samples(n, seed):
    if not isinstance(n, int) or n < 16:
        raise ValueError('need_integer_n_at_least_16')
    return np.random.default_rng(seed).random((2, n, 7))


def conditioned_samples(unit_samples, fixed=(), distance_halfwidth_mm=50.):
    u = np.asarray(unit_samples, float)
    if u.ndim != 2 or u.shape[1] != 7 or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('unit_samples_must_be_N_by_7')
    if len(set(fixed)) != len(fixed) or not set(fixed).issubset(FACTORS):
        raise ValueError('unknown_or_duplicate_fixed_factor')
    if not np.isfinite(distance_halfwidth_mm) or not 0 <= distance_halfwidth_mm <= 50:
        raise ValueError('distance_halfwidth_mm_outside_0_50')
    result = u.copy()
    result[:, 0] = .5 + (u[:, 0]-.5)*(distance_halfwidth_mm/50.)
    for name, index in zip(FACTORS, (0, 2, 3, 1)):
        if name in fixed:
            result[:, index] = .5
    # Zero relative-local offset is passed explicitly to speeds; these columns have no effect.
    result[:, 4:] = .5
    return result


def evaluate(case, unit_samples, fixed=(), distance_halfwidth_mm=50.):
    u = conditioned_samples(unit_samples, fixed, distance_halfwidth_mm)
    return speeds(case, u, offset_radius_m=0., relative_motion=True)


def summarize(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('need_nonempty_finite_vector')
    q = np.quantile(values, [.025, .5, .975])
    return {'mean_kmh': float(values.mean()), 'q025_kmh': float(q[0]),
            'median_kmh': float(q[1]), 'q975_kmh': float(q[2]),
            'width_kmh': float(q[2]-q[0])}


def numerical_difference(a, b):
    return {'mean_difference_kmh': abs(a['mean_kmh']-b['mean_kmh']),
            'max_endpoint_difference_kmh': max(abs(a[k]-b[k]) for k in ('q025_kmh', 'q975_kmh')),
            'width_difference_kmh': abs(a['width_kmh']-b['width_kmh'])}


def width_shapley(widths):
    """Allocate baseline-to-all-fixed width reduction over all orders, retaining interactions."""
    if set(widths) != set(range(16)) or not np.isfinite(list(widths.values())).all():
        raise ValueError('need_all_16_finite_widths')
    output = {}
    for i, name in enumerate(FACTORS):
        others = [j for j in range(4) if j != i]
        value = 0.
        for size in range(4):
            weight = factorial(size)*factorial(3-size)/factorial(4)
            for subset in combinations(others, size):
                mask = sum(1 << j for j in subset)
                value += weight*(widths[mask]-widths[mask | (1 << i)])
        output[name] = float(value)
    return output


def local_distance_sensitivity(case):
    nominal = np.full((1, 7), .5)
    base = float(speeds(case, nominal, 0., True)[0])
    checks = []
    for epsilon_mm in (1., .5, .25):
        u = np.full((2, 7), .5)
        u[:, 0] += np.array([-1., 1.])*(epsilon_mm/100.)
        values = speeds(case, u, 0., True)
        checks.append({'epsilon_mm': epsilon_mm,
                       'slope_kmh_per_mm': float((values[1]-values[0])/(2*epsilon_mm))})
    # v = w + R_dot * [.15,0,0]; d||v||/dd = v.dot(R_dot[:,0])/||v||.
    derivative = case['rotation_derivative'][1, 1, :, 0]
    velocity = case['wrist_velocity'][1] + .15*derivative
    norm = float(np.linalg.norm(velocity))
    if norm <= 0:
        raise ValueError('zero_nominal_speed')
    analytic = float(np.dot(velocity, derivative)/norm*3.6/1000.)
    slope = max(abs(row['slope_kmh_per_mm']) for row in checks)
    return {'nominal_kmh': base, 'finite_differences': checks,
            'analytic_slope_kmh_per_mm': analytic,
            'vector_derivative_norm_kmh_per_mm': float(np.linalg.norm(derivative)*3.6/1000.),
            'max_analytic_fd_difference_kmh_per_mm': max(abs(row['slope_kmh_per_mm']-analytic) for row in checks),
            'local_distance_mm_for_1kmh': None if slope == 0 else 1./slope}


def scenario_goal(rows):
    """New scenario criterion only; never an empirical accuracy verdict."""
    expected = {(f'3034_{i}', seed) for i in range(1, 6) for seed in SEEDS}
    if {(row['pitch'], row['seed']) for row in rows} != expected or len(rows) != 10:
        raise ValueError('need_five_pitches_both_seeds')
    maximum = max(row['summary']['width_kmh'] for row in rows)
    return {'max_width_kmh': maximum, 'all_five_both_seeds_width_le_2': maximum <= 2.,
            'scope': 'conditional scenario target, not actual accuracy'}
