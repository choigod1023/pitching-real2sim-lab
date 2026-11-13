"""Audit same-pitch image hand landmarks before connecting physical models.

The camera convention and Theia alignment follow the existing prime_bridge
adapter. This module deliberately imports no legacy dynamics and never uses a
detector's single-view world coordinates as observed metric coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TRIAL = "OBP_movement_BaseballThrow_001"
CAMERA_IDS = (17, 20, 21)
# Same convention as experiments/prime_bridge/bridge.py: model rotation = source rotation @ HAND_AXES.
HAND_AXES = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
EDGES = tuple((start + j, start + j + 1) for start in (1, 5, 9, 13, 17) for j in range(3)) + tuple((0, start) for start in (1, 5, 9, 13, 17))


def project(points, camera):
    """Return pixels and signed camera depths in the frozen pinhole model."""
    points = np.asarray(points, dtype=float)
    k, r, t = (np.asarray(a, dtype=float) for a in camera)
    if points.ndim != 2 or points.shape[1] != 3 or k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("Expected Nx3 points and K/R/t camera")
    if not all(np.isfinite(a).all() for a in (k, r, t)):
        raise ValueError("Finite camera required")
    coordinates = points @ r.T + t
    image = coordinates @ k.T
    pixels = np.full((len(points), 2), np.nan)
    valid = np.isfinite(image).all(axis=1) & (np.abs(image[:, 2]) > 1e-12)
    pixels[valid] = image[valid, :2] / image[valid, 2, None]
    return pixels, coordinates[:, 2]


def triangulate_pair(pixels, cameras):
    """DLT each landmark without filling missing points or fitting held views."""
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 3 or pixels.shape[0] != 2 or pixels.shape[2] != 2 or len(cameras) != 2:
        raise ValueError("Expected two views of Nx2 image coordinates")
    centers = [-np.asarray(r).T @ np.asarray(t) for _, r, t in cameras]
    if np.linalg.norm(centers[0] - centers[1]) < 1e-9:
        raise ValueError("Distinct camera centers required")
    matrices = [np.asarray(k) @ np.column_stack((r, t)) for k, r, t in cameras]
    points = np.full((pixels.shape[1], 3), np.nan)
    for j in range(len(points)):
        if not np.isfinite(pixels[:, j]).all():
            continue
        equations = np.stack([uv[a] * p[2] - p[a]
                              for uv, p in zip(pixels[:, j], matrices, strict=True) for a in (0, 1)])
        _, singular, vt = np.linalg.svd(equations)
        homogeneous = vt[-1]
        if singular[-2] <= 1e-10 or abs(homogeneous[-1]) <= 1e-10:
            continue
        points[j] = homogeneous[:3] / homogeneous[-1]
    return points


def longest_streak(frames, valid):
    """Count adjacent original frames, not adjacent rows of a sparse table."""
    frames, valid = np.asarray(frames), np.asarray(valid, dtype=bool)
    if frames.ndim != 1 or frames.shape != valid.shape or np.any(np.diff(frames) <= 0):
        raise ValueError("Ordered unique frames and one validity per frame required")
    longest = streak = 0
    previous = None
    for frame, keep in zip(frames, valid, strict=True):
        streak = (streak + 1 if previous is not None and frame == previous + 1 else 1) if keep else 0
        longest = max(longest, streak)
        previous = frame
    return int(longest)


def summary(values):
    """Summarize finite measurements; absent measurements remain null."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"n": int(len(values)), "median": float(np.median(values)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None}


def audit_multiview(frames, pixels, cameras, body_wrist):
    """Evaluate all predeclared two-view reconstructions in their third view."""
    frames = np.asarray(frames)
    pixels, body_wrist = np.asarray(pixels, dtype=float), np.asarray(body_wrist, dtype=float)
    if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer) or np.any(np.diff(frames) <= 0):
        raise ValueError("Ordered unique integer original frames required")
    if pixels.shape != (3, len(frames), 21, 2) or body_wrist.shape != (len(frames), 3) or len(cameras) != 3:
        raise ValueError("Expected three views, 21 landmarks, and same-frame source wrists")
    results, reconstructed = [], []
    for first, second, held in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        candidate = np.full((len(frames), 21, 3), np.nan)
        rows, passes, held_errors, wrist_deltas, sizes, lengths = [], [], [], [], [], []
        for i, frame in enumerate(frames):
            xyz = triangulate_pair(pixels[[first, second], i], [cameras[first], cameras[second]])
            candidate[i] = xyz
            projected = [project(xyz, camera) for camera in cameras]
            depths = np.asarray([item[1] for item in projected])
            valid_points = np.isfinite(xyz).all(axis=1)
            positive = valid_points & (depths > 0).all(axis=0)
            evaluation_valid = positive & np.isfinite(pixels[held, i]).all(axis=1)
            error = np.linalg.norm(projected[held][0][evaluation_valid] - pixels[held, i, evaluation_valid], axis=1)
            aggregate = summary(error)
            complete = bool(evaluation_valid.all())
            passed = complete and aggregate["median"] <= 5. and aggregate["max"] <= 10.
            wrist_delta = (float(np.linalg.norm(xyz[0] - body_wrist[i]) * 1000)
                           if positive[0] and np.isfinite(body_wrist[i]).all() else None)
            size = float(np.linalg.norm(xyz[12] - xyz[0]) * 1000) if positive[[0, 12]].all() else None
            bone = np.array([np.linalg.norm(xyz[b] - xyz[a]) * 1000 if positive[[a, b]].all() else np.nan
                             for a, b in EDGES])
            fit_errors = np.concatenate([np.linalg.norm(projected[c][0][positive] - pixels[c, i, positive], axis=1)
                                         for c in (first, second)])
            rows.append({"frame": int(frame), "triangulated_points": int(valid_points.sum()),
                         "positive_depth_points_all_three_views": int(positive.sum()),
                         "heldout_points": int(evaluation_valid.sum()), "heldout_error_px": aggregate,
                         "fit_error_px": summary(fit_errors), "source_wrist_difference_mm": wrist_delta,
                         "wrist_to_middle_tip_mm": size, "passes_image_geometry": bool(passed)})
            passes.append(passed)
            held_errors.extend(error.tolist())
            wrist_deltas.append(np.nan if wrist_delta is None else wrist_delta)
            sizes.append(np.nan if size is None else size)
            lengths.append(bone)
        lengths = np.asarray(lengths).reshape(len(frames), len(EDGES))
        bone_spreads = [float(np.ptp(column[np.isfinite(column)])) if np.isfinite(column).sum() >= 2 else np.nan
                        for column in lengths.T]
        streak = longest_streak(frames, passes)
        results.append({"fit_views": [CAMERA_IDS[first], CAMERA_IDS[second]], "heldout_view": CAMERA_IDS[held],
                        "frames": rows, "full_frame_geometry_pass_count": int(sum(passes)),
                        "consecutive_pass_frames": streak, "image_geometry_candidate": streak >= 3,
                        "heldout_error_px": summary(held_errors), "source_wrist_difference_mm": summary(wrist_deltas),
                        "wrist_to_middle_tip_mm": summary(sizes), "bone_length_temporal_range_mm": summary(bone_spreads)})
        reconstructed.append(candidate)
    report = {"trial": TRIAL, "camera_ids": list(CAMERA_IDS), "pairs": results,
              "any_image_geometry_candidate": any(row["image_geometry_candidate"] for row in results),
              "all_pairs_image_geometry_candidates": all(row["image_geometry_candidate"] for row in results),
              "body_attachment_validated": False, "physical_contact_validated": False,
              "dynamics_executed": False, "prediction": {"speed_kmh": None, "rpm": None, "ivb_cm": None, "hb_cm": None},
              "same_pitch_record": {"listed_speed_kmh": 62.7 * 1.609344,
                                     "speed_protocol_verified": False, "rpm": None, "ivb_cm": None, "hb_cm": None},
              "scope": "Provisional cameras and source-hand origin; image geometry is not anatomical/contact/force validation."}
    return report, np.asarray(reconstructed)


def read_prime_inputs(lab, frames, with_rotations=False):
    """Read the existing same-pitch C3D/alignment ledger and provisional rig."""
    import ezc3d

    lab, frames = Path(lab), np.asarray(frames, dtype=int)
    ledger_path = lab / "hand-ball-link/artifacts/hand-ball/observation.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    paths = {"body": lab / "tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d",
             "alignment": lab / "openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json",
             "ball": lab / "speed-journey/artifacts/release-observation/candidate_audit.json"}
    hashes = {"observation_ledger": hashlib.sha256(ledger_path.read_bytes()).hexdigest()}
    for key, path in paths.items():
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[key] != ledger["source_sha256"][key]["sha256"]:
            raise ValueError("Same-pitch source hash changed: " + key)
    alignment = json.loads(paths["alignment"].read_text(encoding="utf-8"))
    if ledger["trial"] != TRIAL or alignment["trial"] != TRIAL or alignment["c3d_frame_minus_video_frame"] != 1:
        raise ValueError("Same-pitch identity/time alignment mismatch")
    c3d = ezc3d.c3d(str(paths["body"]))
    if c3d["parameters"]["ROTATION"]["RATE"]["value"][0] != 360.:
        raise ValueError("Expected 360 Hz source rotations")
    raw = c3d["data"]["rotations"]
    if len(frames) and (frames.min() < 0 or frames.max() + 1 >= raw.shape[-1]):
        raise ValueError("Frame outside same-pitch body source")
    labels = c3d["parameters"]["ROTATION"]["LABELS"]["value"]
    hand = raw[:, :, labels.index("l_hand_4X4"), :][:, :, frames + 1]
    origins = hand[:3, 3].T
    r, t = np.asarray(alignment["rotation"]), np.asarray(alignment["translation"])
    if not np.isfinite(r).all() or np.max(np.abs(r.T @ r - np.eye(3))) > 1e-5 or np.linalg.det(r) <= 0:
        raise ValueError("Proper alignment rotation required")
    wrists = alignment["scale"] * (origins @ r.T) + t
    directory = lab / "openbiomechanics/computer_vision/calibration/results"
    intr_path, extr_path = directory / "calibration_results.json", directory / "optitrack_rig_provisional.json"
    intr, extr = json.loads(intr_path.read_bytes()), json.loads(extr_path.read_bytes())
    cameras = []
    for camera_id in CAMERA_IDS:
        k, pose = intr["calibrations"][f"optitrack_{camera_id}"], extr["camera_poses"][str(camera_id)]
        if np.any(k["distortion"]):
            raise ValueError("Frozen projection expects zero distortion")
        cameras.append((np.array([[k["fx_px"], 0., k["cx_px"]], [0., k["fy_px"], k["cy_px"]], [0., 0., 1.]]),
                        np.asarray(pose["reference_to_camera_rotation_matrix"]),
                        np.asarray(pose["reference_to_camera_translation_mm"]) / 1000.))
    hashes.update({"intrinsics": hashlib.sha256(intr_path.read_bytes()).hexdigest(),
                   "rig": hashlib.sha256(extr_path.read_bytes()).hexdigest()})
    metadata = {"source_sha256": hashes, "rig_status": extr["status"],
                "alignment_status": alignment["status"], "video_to_c3d_frame_offset": 1,
                "rate_hz": 360., "hand_origin_independently_verified": False}
    result = (cameras, wrists, metadata)
    if with_rotations:
        result += (np.einsum("ij,jkn->nik", r, hand[:3, :3]),)
    return result


def contiguous_velocity(frames, positions, rate_hz):
    """Differentiate finite, adjacent-frame runs only; never bridge an occlusion."""
    frames, positions = np.asarray(frames), np.asarray(positions, dtype=float)
    if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer) or np.any(np.diff(frames) <= 0):
        raise ValueError("Ordered unique integer frames required")
    if positions.ndim != 3 or positions.shape[0] != len(frames) or positions.shape[2] != 3 or not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("Expected NxJx3 positions and positive capture rate")
    velocity = np.full_like(positions, np.nan)
    for joint in range(positions.shape[1]):
        usable = np.flatnonzero(np.isfinite(positions[:, joint]).all(axis=1))
        cuts = np.flatnonzero(np.diff(frames[usable]) != 1) + 1
        for run in np.split(usable, cuts):
            if len(run) < 2:
                continue
            velocity[run, joint] = np.gradient(positions[run, joint], frames[run] / rate_hz, axis=0, edge_order=1)
    return velocity


def contact_input_adapter(frames, landmarks, hand_origins, hand_rotations, rate_hz=360.):
    """Prepare hand-relative fingertip kinematics, not measured contact pads.

    Rotations are local-to-rig columns in the source l_hand convention, not the
    legacy MuJoCo +X hand-axis convention. A rollout adapter must explicitly
    apply HAND_AXES, keep reaction forces, and account for local pad work.
    """
    frames = np.asarray(frames)
    landmarks, origins, rotations = (np.asarray(x, dtype=float) for x in (landmarks, hand_origins, hand_rotations))
    if landmarks.shape != (len(frames), 21, 3) or origins.shape != (len(frames), 3) or rotations.shape != (len(frames), 3, 3):
        raise ValueError("Expected same-frame 21-point hand and source hand frames")
    finite_rotation = np.isfinite(rotations).all(axis=(1, 2))
    selected_rotations = rotations[finite_rotation]
    if len(selected_rotations) and (np.max(np.abs(selected_rotations.transpose(0, 2, 1) @ selected_rotations - np.eye(3))) > 1e-5
                                    or np.any(np.linalg.det(selected_rotations) <= 0)):
        raise ValueError("Proper source hand rotations required")
    valid = np.isfinite(landmarks).all(axis=2) & np.isfinite(origins).all(axis=1)[:, None] & finite_rotation[:, None]
    local = np.einsum("nji,nkj->nki", rotations, landmarks - origins[:, None])
    local[~valid] = np.nan
    reconstructed = origins[:, None] + np.einsum("nij,nkj->nki", rotations, local)
    errors = np.linalg.norm(reconstructed - landmarks, axis=2) * 1000
    tips = local[:, [8, 12]]
    velocity = contiguous_velocity(frames, tips, rate_hz)
    arrays = {"landmarks_local_m": local, "fingertip_candidates_local_m": tips,
              "fingertip_local_velocity_m_s": velocity, "fingertip_velocity_valid": np.isfinite(velocity).all(axis=2),
              "fingertip_candidates_model_local_m": tips @ HAND_AXES,
              "fingertip_model_local_velocity_m_s": velocity @ HAND_AXES}
    contract = {"prepared": True, "executed": False, "trial": TRIAL, "landmark_ids": [8, 12],
                "coordinate_frame": "source Theia l_hand, local-to-rig rotation columns",
                "model_local_conversion": "source_local row vectors @ HAND_AXES; model rotation = source rotation @ HAND_AXES",
                "landmark_semantics": "Index/middle detector fingertip candidates; not measured surface contact/pad centers",
                "world_to_local_to_world_difference_mm": summary(errors),
                "finite_fingertip_velocity_frames": np.isfinite(velocity).all(axis=2).sum(axis=0).tolist(),
                "capture_rate_hz": float(rate_hz), "capture_time_alignment_independently_verified": False,
                "contact_pad_radius_m": None, "contact_normal_local": None, "tip_to_pad_offset_m": None,
                "measured_finger_force_n": None, "muscle_capacity_validated": False,
                "rollout_requirements": ["Validated three-view geometry and same-pitch body/hand frame identity",
                                         "Explicit l_hand-to-model hand-axis conversion",
                                         "Observed/uncertainty-bounded fingertip-to-surface-pad geometry",
                                         "Contiguous velocity support; stop at missing observations",
                                         "Ball-hand reaction forces and local finger positive/negative work",
                                         "Preserved initial ball energy and body force/work budgets"],
                "scope": "Input contract only; no free-ball contact or performance prediction was run."}
    return contract, arrays


def read_inference(path, method):
    """Adapt the frozen RGB detector manifest, retaining absent camera frames."""
    if method not in ("raw", "rotated"):
        raise ValueError("Expected one predeclared input method")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = {str(row["clip_id"]): row for row in data["clips"]}
    if len(clips) != len(data["clips"]):
        raise ValueError("Duplicate inference clip identity")
    tables, all_frames = [], set()
    for camera in CAMERA_IDS:
        key = f"prime{camera}"
        if key not in clips:
            raise ValueError("Missing same-pitch camera clip: " + key)
        table = {}
        for row in clips[key]["methods"][method]:
            frame = row["frame"]
            if type(frame) is not int or frame < 0 or frame in table:
                raise ValueError("Expected unique nonnegative original integer frames")
            selected = row["selected_index"]
            point = np.full((21, 2), np.nan)
            if selected is not None:
                if type(selected) is not int or not 0 <= selected < len(row["candidates"]):
                    raise ValueError("Invalid selected hand candidate")
                candidate = row["candidates"][selected]
                if candidate["eligible"] is not True:
                    raise ValueError("Ineligible hand candidate selected")
                point = np.asarray(candidate["pixel21"], dtype=float)
                if point.shape != (21, 2) or not np.isfinite(point).all():
                    raise ValueError("Selected hand must have 21 finite original-image points")
            table[frame] = point
        all_frames.update(table)
        tables.append(table)
    frames = np.asarray(sorted(all_frames), dtype=int)
    pixels = np.full((3, len(frames), 21, 2), np.nan)
    for i, table in enumerate(tables):
        for j, frame in enumerate(frames):
            if frame in table:
                pixels[i, j] = table[frame]
    return frames, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Detector JSON or NPZ with frames/pixels/camera_ids")
    parser.add_argument("--method", choices=("raw", "rotated"), default="raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".json":
        frames, pixels = read_inference(args.input, args.method)
    else:
        with np.load(args.input, allow_pickle=False) as data:
            if not np.array_equal(data["camera_ids"], CAMERA_IDS):
                raise ValueError("Expected camera order 17,20,21")
            frames, pixels = data["frames"], data["pixels"]
    cameras, wrists, provenance, rotations = read_prime_inputs(args.lab, frames, with_rotations=True)
    result, xyz = audit_multiview(frames, pixels, cameras, wrists)
    result.update(provenance)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["input_method"] = args.method
    contracts, local, tips, velocities, velocity_valid, model_tips, model_velocities = [], [], [], [], [], [], []
    for pair, points in zip(result["pairs"], xyz, strict=True):
        # Failed reconstructions stay in points for audit, but cannot enter the contact adapter.
        accepted = np.asarray([row["passes_image_geometry"] for row in pair["frames"]], dtype=bool)
        accepted &= pair["image_geometry_candidate"]
        contact_points = points.copy()
        contact_points[~accepted] = np.nan
        contract, arrays = contact_input_adapter(frames, contact_points, wrists, rotations, provenance.get("rate_hz", 360.))
        contract["image_geometry_candidate"] = pair["image_geometry_candidate"]
        contract["input_geometry_accepted_frames"] = int(accepted.sum())
        contract["status"] = "prepared_only_contact_not_validated" if pair["image_geometry_candidate"] else "prepared_not_run_geometry_gate"
        contracts.append(contract)
        local.append(arrays["landmarks_local_m"])
        tips.append(arrays["fingertip_candidates_local_m"])
        velocities.append(arrays["fingertip_local_velocity_m_s"])
        velocity_valid.append(arrays["fingertip_velocity_valid"])
        model_tips.append(arrays["fingertip_candidates_model_local_m"])
        model_velocities.append(arrays["fingertip_model_local_velocity_m_s"])
    result["contact_input_contracts"] = contracts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".private.npz"), frames=frames, points=xyz, source_wrist=wrists,
                        source_hand_rotations=rotations, landmarks_local_m=np.asarray(local),
                        fingertip_candidates_local_m=np.asarray(tips), fingertip_local_velocity_m_s=np.asarray(velocities),
                        fingertip_velocity_valid=np.asarray(velocity_valid), fingertip_candidates_model_local_m=np.asarray(model_tips),
                        fingertip_model_local_velocity_m_s=np.asarray(model_velocities))
    print(json.dumps({"pairs": [{k: row[k] for k in ("heldout_view", "consecutive_pass_frames", "image_geometry_candidate")}
                                for row in result["pairs"]], "dynamics_executed": False}))


if __name__ == "__main__":
    main()
