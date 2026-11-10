"""Estimate the period of visible ball texture in a high-speed release video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares


WINDOWS = ((0, 28), (5, 45), (15, 55), (25, 75), (0, 75))
MASK_RADII = (0.70, 0.80, 0.90)
BALL_TRACE_COLUMNS = ('zero_based_frame', 'center_x_px', 'center_y_px', 'radius_px', 'arc_rmse_px')


def track_ball(
    clip: Path,
    first_frame: int,
    initial_circle: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the upper ball arc from an image-space seed through frame 99."""
    if not 0 <= first_frame <= 100 - max(end for _, end in WINDOWS):
        raise ValueError('First frame must leave 75 frames within the first 100 frames.')
    rolling = np.asarray(initial_circle, dtype=float)
    if rolling.shape != (3,) or not np.isfinite(rolling).all() or rolling[2] <= 8:
        raise ValueError('Initial circle must have finite x, y and a radius above 8 pixels.')

    capture = cv2.VideoCapture(str(clip))
    circles, chips = [], []
    yy, xx = np.mgrid[-1.1:1.1:48j, -1.1:1.1:48j]
    theta = np.deg2rad(np.linspace(-155, 35, 65))
    vectors = np.c_[np.cos(theta), np.sin(theta)]
    bound = np.array([5, 5, 4])
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {clip}')
        for frame_number in range(100):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Decode stopped at frame {frame_number}.')
            if frame_number < first_frame:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            radius = np.linspace(max(6, rolling[2] - 8), rolling[2] + 8, 65)
            coords = rolling[:2] + radius[None, :, None] * vectors[:, None, :]
            profile = map_coordinates(gray, [coords[..., 1], coords[..., 0]], order=1)
            edge = -np.gradient(gaussian_filter1d(profile, 3, axis=1), axis=1)
            maxima = edge.argmax(axis=1)
            ids = np.arange(len(theta))
            keep = (maxima > 5) & (maxima < len(radius) - 6) & (edge[ids, maxima] > .6)
            points = coords[ids[keep], maxima[keep]]
            if len(points) < 20:
                raise ValueError(f'Fewer than 20 visible arc points at frame {frame_number}.')

            def residual(circle: np.ndarray) -> np.ndarray:
                return np.linalg.norm(points - circle[:2], axis=1) - circle[2]

            fit = least_squares(
                residual, rolling, loss='soft_l1', f_scale=1,
                bounds=(rolling - bound, rolling + bound),
            )
            x, y, r = fit.x
            rolling = fit.x
            circles.append([frame_number, x, y, r, np.sqrt(np.mean(fit.fun**2))])
            chips.append(map_coordinates(gray, [y + r*yy, x + r*xx], order=1))
    finally:
        capture.release()
    return np.asarray(circles), np.asarray(chips), xx**2 + yy**2


def estimate_period(chips: np.ndarray, squared_distance: np.ndarray) -> list[dict]:
    """Measure seam recurrence across fixed disk masks and clip-relative windows."""
    if chips.ndim != 3 or chips.shape[1:] != squared_distance.shape:
        raise ValueError('Image patches and the distance grid must have matching spatial dimensions.')
    if len(chips) < max(end for _, end in WINDOWS):
        raise ValueError('At least 75 normalized patches are required.')
    if not np.isfinite(chips).all() or not np.isfinite(squared_distance).all():
        raise ValueError('Patches and distances must be finite.')

    specs = []
    median = np.median(chips, axis=0)
    for mask_radius in MASK_RADII:
        mask = squared_distance < mask_radius**2
        whitened = (chips - median)[:, mask]
        if whitened.shape[1] < 2:
            raise ValueError('Mask has fewer than two pixels.')
        whitened -= whitened.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('A normalized patch has no visible texture variation.')
        whitened /= norms
        for start, end in WINDOWS:
            series = whitened[start:end]
            lag_lo = 4
            lag_hi = min(38, len(series) - 5)
            acf = np.array([
                np.mean(np.sum(series[:-lag] * series[lag:], axis=1))
                for lag in range(lag_lo, lag_hi + 1)
            ])
            peaks = np.flatnonzero((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:])) + 1
            accepted = peaks[acf[peaks] > .45]
            if len(accepted) == 0:
                raise ValueError(f'No positive period candidate for mask {mask_radius}, {start}:{end}.')
            peak = int(accepted[0])
            alpha, beta, gamma = acf[peak - 1:peak + 2]
            shift = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            integer = peak + lag_lo
            half_lag = round(integer / 2)
            specs.append({
                'mask_radius_ball_radius': mask_radius,
                'relative_frame_start_inclusive': start,
                'relative_frame_end_exclusive': end,
                'integer_period_frames': integer,
                'period_frames_quadratic_diagnostic': float(integer + shift),
                'peak_integer_lag_correlation': float(beta),
                'half_period_nearest_lag_correlation': (
                    float(acf[half_lag - lag_lo]) if half_lag >= lag_lo else None
                ),
                'all_local_peaks_frames_and_correlation': [
                    [int(i + lag_lo), float(acf[i])] for i in peaks
                ],
            })
    return specs


def analyze_clip(
    clip: Path,
    first_frame: int,
    initial_circle: tuple[float, float, float],
    nominal_capture_fps: float,
) -> dict:
    """Convert seam periodicity to conditional RPM without instrument inputs."""
    if not np.isfinite(nominal_capture_fps) or nominal_capture_fps <= 0:
        raise ValueError('Nominal capture FPS must be positive and finite.')
    circles, chips, squared_distance = track_ball(clip, first_frame, initial_circle)
    specs = estimate_period(chips, squared_distance)
    periods, rpm = [], []
    for spec in specs:
        period = spec['period_frames_quadratic_diagnostic']
        periods.append(period)
        rpm.append(60 * nominal_capture_fps / period)
        spec['rpm_from_nominal_fps_integer_lag'] = (
            60 * nominal_capture_fps / spec['integer_period_frames']
        )
        spec['rpm_from_nominal_fps_quadratic_diagnostic'] = rpm[-1]
    return {
        'clip': clip.name,
        'nominal_capture_fps': nominal_capture_fps,
        'capture_fps_clip_metadata_verified': False,
        'ball_radial_fit_error_px_median': float(np.median(circles[:, 4])),
        'integer_period_frames': sorted(set(spec['integer_period_frames'] for spec in specs)),
        'quadratic_period_frames_min_median_max': [min(periods), float(np.median(periods)), max(periods)],
        'quadratic_rpm_diagnostic_min_median_max': [min(rpm), float(np.median(rpm)), max(rpm)],
        'ball_trace_columns': BALL_TRACE_COLUMNS,
        'ball_trace': circles.tolist(),
        'specifications': specs,
        'limitations': (
            'Monocular texture recurrence depends on appearance, motion blur and perspective. '
            'Subframe diagnostics and mask/window sensitivity do not establish measurement accuracy. '
            'RPM conversion assumes the nominal acquisition rate and one rotation per texture cycle.'
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--first-frame', type=int, required=True)
    parser.add_argument('--initial-circle', type=float, nargs=3, metavar=('X', 'Y', 'R'), required=True)
    parser.add_argument('--nominal-capture-fps', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze_clip(args.clip, args.first_frame, args.initial_circle, args.nominal_capture_fps)
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
