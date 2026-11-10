"""Estimate a camera-frame spin axis from consecutive textured sphere images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def sphere_images(video, trace, size=65):
    capture = cv2.VideoCapture(str(video))
    grid = np.linspace(-1, 1, size, dtype=np.float32)
    u, v = np.meshgrid(grid, grid)
    patches = []
    index = -1
    try:
        for frame, x, y, radius, _ in trace:
            while index < frame:
                success, image = capture.read()
                if not success:
                    raise ValueError(f"Unable to read video frame {frame}")
                index += 1
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            patch = cv2.remap(gray, x + radius * u, y + radius * v, cv2.INTER_LINEAR)
            detail = cv2.GaussianBlur(patch, (0, 0), .8) - cv2.GaussianBlur(patch, (0, 0), 3.)
            patches.append(detail)
    finally:
        capture.release()
    return np.stack(patches)


class TextureAgreement:
    def __init__(self, patches, start, end, mask_radius):
        frames, size, width = patches.shape
        if size != width or not 0 <= start < end < frames or not 0 < mask_radius <= .7:
            raise ValueError("Invalid sphere images or frame interval")
        coordinate = np.linspace(-1., 1., size)
        u, y = np.meshgrid(coordinate, coordinate)
        inside = u*u + y*y <= mask_radius**2
        self.points = np.column_stack((u[inside], -y[inside], np.sqrt(1 - u[inside]**2 - y[inside]**2)))
        self.images = patches[start + 1:end + 1]
        texture = patches[start:end, inside]
        centered = texture - texture.mean(axis=1, keepdims=True)
        scale = np.linalg.norm(centered, axis=1, keepdims=True)
        if np.any(scale <= 1e-7):
            raise ValueError("A source frame has no usable texture")
        self.source = centered / scale
        self.scale = (size - 1) / 2
        self.frame_axis = np.broadcast_to(np.arange(end-start)[:, None], centered.shape)

    def score(self, rotation_vector):
        points = Rotation.from_rotvec(rotation_vector).apply(self.points)
        if np.min(points[:, 2]) <= 0:
            return -1.
        x = self.scale * (points[:, 0] + 1.)
        y = self.scale * (1. - points[:, 1])
        coordinates = [self.frame_axis, np.broadcast_to(y, self.source.shape),
                       np.broadcast_to(x, self.source.shape)]
        target = map_coordinates(self.images, coordinates, order=1, mode="nearest", prefilter=False)
        target -= target.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(target, axis=1)
        agreement = np.sum(target * self.source, axis=1) / np.maximum(norm, 1e-7)
        return float(np.mean(agreement))


def fit_axis(agreement, radians_per_frame, candidates=160):
    if not 0 < radians_per_frame < .6 or candidates < 10:
        raise ValueError("Expected a small positive rotation and a spherical search grid")
    z = 1 - 2 * (np.arange(candidates) + .5) / candidates
    azimuth = np.arange(candidates) * np.pi * (3 - np.sqrt(5))
    angles = np.column_stack((azimuth, np.arcsin(z)))

    def axis(parameters):
        az, el = parameters
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def error(parameters):
        return -agreement.score(radians_per_frame * axis(parameters))

    index = np.argsort([error(candidate) for candidate in angles])[:3]
    fitted = [minimize(error, angles[i], method="Powell", options={"xtol": 2e-4, "ftol": 1e-6})
              for i in index]
    best = min(fitted, key=lambda result: result.fun)
    return axis(best.x), float(-best.fun)


def inspect(video, tracking):
    trace = tracking["ball_trace"]
    if len(trace) < 56 or np.any(np.diff(np.asarray(trace)[:, 0]) != 1):
        raise ValueError("Need at least 56 consecutive frames with measured ball circles")
    patches = sphere_images(video, trace[:56])
    rotation = 2 * np.pi / tracking["quadratic_period_frames_min_median_max"][1]
    train = TextureAgreement(patches, 2, 29, .60)
    test = TextureAgreement(patches, 30, 55, .60)
    axis, score = fit_axis(train, rotation)
    variants = {"fit_0.60": axis.tolist()}
    for label, objective in (
        ("fit_0.52", TextureAgreement(patches, 2, 29, .52)),
        ("fit_0.66", TextureAgreement(patches, 2, 29, .66)),
        ("test_fitted_separately", test),
    ):
        alternative, _ = fit_axis(objective, rotation)
        variants[label] = alternative.tolist()
    return {
        "first_tracked_frame": int(trace[0][0]),
        "fit_source_frames": [int(trace[2][0]), int(trace[29][0])],
        "test_source_frames": [int(trace[30][0]), int(trace[55][0])],
        "radians_per_frame": rotation, "camera_axes": variants,
        "train_correlation": score, "train_no_rotation": train.score(np.zeros(3)),
        "test_correlation": test.score(rotation * axis), "test_no_rotation": test.score(np.zeros(3)),
        "max_axis_sensitivity_deg": float(max(
            np.rad2deg(np.arccos(np.clip(np.dot(axis, variant), -1, 1))) for variant in variants.values()
        )),
        "optical_rpm_at_nominal_capture_fps": tracking["nominal_capture_fps"] * 60 * rotation / (2 * np.pi),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video_4", "track_4", "video_8", "track_8", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {
            pitch: pool.submit(inspect, video, json.loads(track.read_text(encoding="utf-8")))
            for pitch, video, track in ((4, args.video_4, args.track_4), (8, args.video_8, args.track_8))
        }
        report = {pitch: job.result() for pitch, job in jobs.items()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
