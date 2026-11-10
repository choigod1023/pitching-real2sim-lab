"""Test conditional rear-camera alignment on two linked Rapsodo pitches."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from experiments.linked_highspeed_release.study import launch_basis, predict_movement, read_pitches
from experiments.seam_axis_flight.study import optical_inputs, summarize_flight


PRINCIPAL = (960., 540.)
RADIUS_M = .0366
MODES = ("before", "rotation_only", "rotation_and_launch")
LATE_CENTER_FRAMES = {"4": (110, 130, 150, 170, 190, 210, 220, 230, 235),
                      "8": (110, 130, 150, 170, 190, 210, 220, 235, 245)}
LATE_RADIUS_FRAMES = {"4": (110, 130, 170, 190, 210, 215, 220),
                      "8": (110, 130, 150, 170, 190, 210, 220)}


def unit(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Expected a finite direction")
    return np.asarray(vector, dtype=float) / norm


def line(endpoints):
    p, q = np.column_stack((np.asarray(endpoints, dtype=float), np.ones(2)))
    cross = np.cross(p, q)
    return cross / np.linalg.norm(cross[:2])


def camera_basis(scene, focal, principal):
    """Rows map camera right/away/up into the inferred lane frame."""
    centerline = line(scene["plate_and_device_px"])
    vanishing = np.cross(centerline, line(scene["wall_primary_px"]))
    if abs(vanishing[2]) < 1e-10:
        raise ValueError("Longitudinal vanishing point at infinity")
    uv = vanishing[:2] / vanishing[2]
    cx, cy = principal
    forward = unit([(uv[0] - cx) / focal, 1., (cy - uv[1]) / focal])
    a, b, c = line(scene["vertical_primary_px"])
    vertical_plane = [focal * a, a * cx + b * cy + c, -focal * b]
    up = unit(np.cross(forward, vertical_plane))
    if up[2] < 0:
        up = -up
    return np.stack((np.cross(forward, up), forward, up)), uv


def direction_residual(endpoints, direction, focal, principal):
    """Held-out line endpoint distance from a line through its first point and the predicted vanishing point."""
    cx, cy = principal
    x, y, z = direction
    vanishing = [focal * x + cx * y, cy * y - focal * z, y]
    p, q = np.column_stack((np.asarray(endpoints, dtype=float), np.ones(2)))
    predicted = np.cross(p, vanishing)
    return float(abs(predicted @ q) / np.linalg.norm(predicted[:2]))


def positions(trace, focal, principal):
    """Approximate a spherical center from a perspective image circle."""
    data = np.asarray(trace, dtype=float)
    if data.ndim != 2 or data.shape[1] < 4 or len(data) < 3:
        raise ValueError("Expected rows containing frame, u, v, and radius")
    if not np.isfinite(data[:, :4]).all() or np.any(data[:, 3] <= 0) or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Expected ordered finite frames and positive radii")
    _, u, v, radius = data[:, :4].T
    cx, cy = principal
    return (RADIUS_M / radius)[:, None] * np.column_stack((u - cx, np.full(len(u), focal), cy - v))


def initial_velocity(trace, focal, principal, fps, start=1, count=20):
    data = np.asarray(trace, dtype=float)[start:start + count]
    if len(data) != count or not np.all(np.diff(data[:, 0]) == 1):
        raise ValueError("Initial velocity requires consecutive frames in the locked window")
    t = (data[:, 0] - data[:, 0].mean()) / fps
    return t @ positions(data, focal, principal) / (t @ t)


def cross_focal(trace, speed_m_s, fps):
    vx, depth_per_px, vz = initial_velocity(trace, 1., PRINCIPAL, fps)
    depth_squared = speed_m_s**2 - vx**2 - vz**2
    if depth_per_px <= 0 or depth_squared <= 0:
        raise ValueError("Anchor speed incompatible with apparent sphere motion")
    return float(np.sqrt(depth_squared) / depth_per_px)


def angular_pitch(pitch, direction):
    x, y, z = unit(direction)
    return {
        **pitch,
        "horizontal_angle_deg": float(np.rad2deg(np.arctan2(x, y))),
        "vertical_angle_deg": float(np.rad2deg(np.arctan2(z, np.hypot(x, y)))),
    }


def revised_inputs(pitch, observed_pitch, axis):
    observed, right, up = launch_basis(observed_pitch)
    axis = unit(axis)
    transverse = axis - (axis @ observed) * observed
    fraction = float(np.linalg.norm(transverse))
    transverse = unit(transverse)
    _, recorded_right, recorded_up = launch_basis(pitch)
    transported = (transverse @ right) * recorded_right + (transverse @ up) * recorded_up
    return fraction, transported, transverse


def corrected_flights(pitch, observed_pitch, field_axis, dt=.001):
    fraction, transported, transverse = revised_inputs(pitch, observed_pitch, field_axis)
    spin = fraction * pitch["rpm"]
    return fraction, {
        "rotation_only": predict_movement(pitch, dt=dt, spin_axis=transported, active_rpm=spin),
        "rotation_and_launch": predict_movement(observed_pitch, dt=dt, spin_axis=transverse, active_rpm=spin),
    }


def sensitivity(pitch, trace, scene, focal, optical_axis, fps):
    samples = []
    failures = 0
    for scale, dx, dy, start, count in product((.85, 1., 1.15), (-40, 0, 40), (-40, 0, 40), (1, 6), (20, 25)):
        f = focal * scale
        principal = (PRINCIPAL[0] + dx, PRINCIPAL[1] + dy)
        basis, _ = camera_basis(scene, f, principal)
        observed = angular_pitch(pitch, basis @ initial_velocity(trace, f, principal, fps, start, count))
        try:
            fraction, flights = corrected_flights(pitch, observed, basis @ optical_axis, dt=.004)
        except ValueError:
            failures += 1
            continue
        samples.append([100 * fraction, observed["horizontal_angle_deg"], observed["vertical_angle_deg"],
                        flights["rotation_and_launch"][0][1], 100 * flights["rotation_and_launch"][1][-1, 2]])
    values = np.asarray(samples)
    names = ("efficiency_pct", "horizontal_angle_deg", "vertical_angle_deg", "ivb_cm", "plate_height_cm")
    return {"cases": len(samples), "failures": failures,
            "ranges": dict(zip(names, np.column_stack((values.min(axis=0), values.max(axis=0))).tolist()))}


def project_path(path, frames, first_frame, origin, basis, focal, fps, dt=.001):
    """Project trajectory offsets at nominal frame times, without fitting the late images."""
    time = (np.asarray(frames) - first_frame) / fps
    # The last RK4 step is shortened at the plate, so use only whole steps.
    times = dt * np.arange(len(path) - 1)
    valid = (time >= 0.) & (time <= times[-1])
    shifts = np.column_stack([np.interp(time[valid], times, column[:-1]) for column in (path - path[0]).T])
    cam = origin + shifts @ basis
    projected = np.full((len(time), 2), np.nan)
    projected[valid] = PRINCIPAL + focal * cam[:, [0, 2]] / cam[:, 1, None] * [1., -1.]
    return projected


def validate_late(key, late, flights, trace, basis, focal, fps, height):
    fits = [{int(row[0]): np.asarray(row[:4]) for row in late["track"][method]} for method in ("upper", "circle")]
    frames = LATE_CENTER_FRAMES[key]
    variants = np.asarray([[fit[frame] for frame in frames] for fit in fits])
    observed = variants.mean(axis=0)[:, 1:3]
    origin = positions(trace, focal, PRINCIPAL)[0]
    scores = {}
    for mode, (_, path) in flights.items():
        projected = project_path(path, frames, trace[0][0], origin, basis, focal, fps)
        valid = np.isfinite(projected).all(axis=1)
        error = np.linalg.norm(projected[valid] - observed[valid], axis=1)
        scores[mode] = {
            "n": int(valid.sum()), "mean_error_px": float(error.mean()) if len(error) else None,
            "projected_px": [point.tolist() if keep else None for point, keep in zip(projected, valid)],
            "errors_px": error.tolist(),
        }
    radius_trace = np.mean([[fit[frame] for frame in LATE_RADIUS_FRAMES[key]] for fit in fits], axis=0)
    spatial = (positions(radius_trace, focal, PRINCIPAL) - origin) @ basis.T + [0., 0., height]
    return {
        "frames": frames, "observed_centers_px": observed.tolist(),
        "center_method_disagreement_mean_px": float(np.linalg.norm(variants[0, :, 1:3] - variants[1, :, 1:3], axis=1).mean()),
        "radius_frames": LATE_RADIUS_FRAMES[key], "video_path_yz_m_registered": spatial[:, 1:3].tolist(),
        "scores": scores,
    }


def compare(workbook, rotations, tracks, geometry, late=None):
    pitches = {str(p["pitch_id"]): p for p in read_pitches(workbook) if p["pitch_type"] == "FB" and p["video"]}
    if set(pitches) != {"4", "8"} or set(tracks) != set(pitches) or set(geometry["scenes"]) != set(pitches):
        raise ValueError("Expected linked pitches #4 and #8 with tracked circles and scene annotations")
    fps = geometry["nominal_capture_fps"]
    focals = {key: cross_focal(tracks[key], pitch["speed_mph"] * .44704, fps) for key, pitch in pitches.items()}
    rows = []
    for key, pitch in sorted(pitches.items()):
        other, = set(pitches) - {key}
        focal = focals[other]
        trace, scene = tracks[key], geometry["scenes"][key]
        basis, uv = camera_basis(scene, focal, PRINCIPAL)
        camera_velocity = initial_velocity(trace, focal, PRINCIPAL, fps)
        observed_pitch = angular_pitch(pitch, basis @ camera_velocity)
        cam = rotations[key]["camera_axes"]["fit_0.60"]
        optical_axis = np.array([cam[0], -cam[2], cam[1]])
        _, baseline_axis, baseline_fraction, _ = optical_inputs(cam, launch_basis(pitch))
        fraction, flights = corrected_flights(pitch, observed_pitch, basis @ optical_axis)
        flights["before"] = predict_movement(pitch, spin_axis=baseline_axis, active_rpm=baseline_fraction * pitch["rpm"])
        recorded = pitch["true_rpm"] / pitch["rpm"]
        path = positions(trace, focal, PRINCIPAL) @ basis.T
        # Draw only the observed segment, registered at the first tracked center.
        path += np.array([0., 0., pitch["release_height_ft"] * .3048]) - path[0]
        rows.append({
            "pitch_id": pitch["pitch_id"], "focal_px": focal, "focal_anchor_pitch_id": int(other),
            "camera_basis": basis.tolist(), "forward_vanishing_point_px": uv.tolist(),
            "heldout_wall_endpoint_residual_px": direction_residual(scene["wall_heldout_px"], basis[1], focal, PRINCIPAL),
            "heldout_vertical_endpoint_residual_px": direction_residual(scene["vertical_heldout_px"], basis[2], focal, PRINCIPAL),
            "recorded_speed_kmh": pitch["speed_mph"] * 1.609344,
            "image_fit_speed_kmh": float(np.linalg.norm(camera_velocity) * 3.6),
            "initial_fit_frames": [int(trace[1][0]), int(trace[20][0])],
            "recorded_horizontal_vertical_deg": [pitch["horizontal_angle_deg"], pitch["vertical_angle_deg"]],
            "image_horizontal_vertical_deg": [observed_pitch["horizontal_angle_deg"], observed_pitch["vertical_angle_deg"]],
            "recorded_efficiency_pct": 100 * recorded,
            "recorded_ivb_cm": pitch["ivb_in"] * 2.54, "recorded_plate_height_cm": pitch["plate_height_in"] * 2.54,
            "flights": {
                mode: {
                    **summarize_flight(pitch, *flights[mode]),
                    "efficiency_pct": 100 * (baseline_fraction if mode == "before" else fraction),
                    "efficiency_error_pp": 100 * ((baseline_fraction if mode == "before" else fraction) - recorded),
                } for mode in MODES
            },
            "video_path_yz_m_registered": path[:, 1:3].tolist(),
            "sensitivity": sensitivity(pitch, trace, scene, focal, optical_axis, fps),
        })
        if late is not None:
            mirrored = {**pitch, "horizontal_angle_deg": -pitch["horizontal_angle_deg"]}
            _, mirror_axis, mirror_fraction, _ = optical_inputs(cam, launch_basis(mirrored))
            mirror_flight = predict_movement(mirrored, spin_axis=mirror_axis, active_rpm=mirror_fraction * pitch["rpm"])
            rows[-1]["late"] = validate_late(key, late[key], {**flights, "mirrored_instrument_azimuth": mirror_flight},
                                             trace, basis, focal, fps, path[0, 2])
    metrics = ("efficiency_error_pp", "ivb_error_cm", "plate_height_error_cm")
    maes = {mode: {metric: float(np.mean([abs(row["flights"][mode][metric]) for row in rows])) for metric in metrics}
            for mode in MODES}
    return {"workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
            "capture_fps_assumed": fps, "eligible_pair_count": 0,
            "paired_speed_gap_kmh": abs(rows[0]["recorded_speed_kmh"] - rows[1]["recorded_speed_kmh"]),
            "cross_focal_px_by_anchor_pitch": focals, "mae": maes, "pitches": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("workbook", "rotation", "track_4", "track_8", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--geometry", type=Path, default=Path(__file__).with_name("scene.json"))
    parser.add_argument("--late-tracks", type=Path)
    args = parser.parse_args()
    paths = (args.rotation, args.track_4, args.track_8, args.geometry)
    rotations, track_4, track_8, geometry = (json.loads(path.read_text(encoding="utf-8-sig")) for path in paths)
    late = None
    if args.late_tracks is not None:
        late = json.loads(args.late_tracks.read_text(encoding="utf-8-sig"))
        paths += (args.late_tracks,)
    report = compare(args.workbook, rotations, {"4": track_4["ball_trace"], "8": track_8["ball_trace"]}, geometry, late)
    report["input_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
