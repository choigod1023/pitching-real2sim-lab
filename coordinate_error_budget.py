"""Noise propagation for a scalar coordinate derivative, not ball-speed accuracy.

Times are seconds, positions/noise standard deviations are metres. Ordinary
least squares uses actual timestamps (no resampling). Independent, zero-mean
position noise is assumed. Calibration error, correlated tracking error, timing
error and polynomial model bias are excluded. A window crossing release can mix
pre-release contact acceleration and free flight: lower noise need not mean lower
total error. Future samples make this an offline estimator, not a causal forecast.
"""
import json
import numbers

import numpy as np


def derivative_weights(timestamps, polynomial_order=1, evaluation_time=0.0):
    """Return weights w such that w @ position estimates its first derivative.

    Evaluation outside the sample interval is permitted (extrapolation) but can
    amplify noise and bias. Numerically rank-deficient fits are rejected.
    """
    t = np.asarray(timestamps, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError("timestamps must be a finite one-dimensional array of length >= 2")
    if not np.all(np.diff(t) > 0):
        raise ValueError("timestamps must be strictly increasing")
    if (isinstance(polynomial_order, (bool, np.bool_))
            or not isinstance(polynomial_order, numbers.Integral)
            or polynomial_order < 1 or polynomial_order >= t.size):
        raise ValueError("polynomial_order must be an integer in [1, number of samples - 1]")
    e = np.asarray(evaluation_time, dtype=float)
    if e.ndim != 0 or not np.isfinite(e):
        raise ValueError("evaluation_time must be a finite scalar")
    shifted = t - float(e)
    scale = np.max(np.abs(shifted))
    if not np.all(np.isfinite(shifted)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("timestamps cannot be stably scaled")
    design = np.polynomial.polynomial.polyvander(shifted / scale, polynomial_order)
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.eye(t.size), rcond=None)
    if rank != polynomial_order + 1:
        raise ValueError("polynomial design is numerically rank deficient")
    weights = coefficients[1] / scale
    if not np.all(np.isfinite(weights)):
        raise ValueError("non-finite derivative weights")
    return weights


def velocity_noise_sigma(timestamps, position_sigma_m, polynomial_order=1,
                         evaluation_time=0.0):
    """Scalar velocity-component SD (m/s), for independent coordinate noise.

    position_sigma_m may be a common scalar SD or one SD per sample. This is
    not the nonlinear 3D speed-magnitude uncertainty or an empirical error MAE.
    """
    weights = derivative_weights(timestamps, polynomial_order, evaluation_time)
    sigma = np.asarray(position_sigma_m, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(weights.shape, float(sigma))
    if sigma.shape != weights.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("position_sigma_m must be nonnegative finite scalar or match timestamps")
    result = float(np.linalg.norm(weights * sigma))
    if not np.isfinite(result):
        raise ValueError("non-finite propagated uncertainty")
    return result


def example_budgets(position_sigma_m=0.0005):
    """Illustrative engineering noise budgets; no observed pitch speeds used."""
    result = []
    for hz in (360, 1000):
        for count in (2, 11):
            t = (np.arange(count) - (count - 1) / 2) / hz
            value = velocity_noise_sigma(t, position_sigma_m)
            result.append(dict(sample_rate_hz=hz, sample_count=count,
                               span_ms=float(np.ptp(t) * 1000), polynomial_order=1,
                               position_sigma_m=position_sigma_m,
                               velocity_component_sigma_kmh=value * 3.6))
    return result


if __name__ == "__main__":
    print(json.dumps(example_budgets(), indent=2))
