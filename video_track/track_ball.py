"""Eight-camera ball tracking for OBP-CV ``OBP_movement_BaseballThrow_001``.

No radar value is used. Forward tracking starts from the image-inspected pixel positions of
cameras 19 and 22 at frames 980, 981 and 985 (they define a first linear 3D path); the path is
projected into all eight cameras, ball detections refine it, and the loop repeats. Backward
tracking then follows the ball into the hand-contact frames: the search window is placed by the
attached-ball hypothesis (same hand-relative position as the next frame), while the position
itself comes only from image detections checked by multi-view reprojection.

Example (paths are placeholders)::

    python -m video_track.track_ball --cv-root "OBP CV Files" \
        --cal-dir openbiomechanics/computer_vision/calibration/results \
        --theia-c3d "OBP CV Files/theia3d_output/c3d/OBP_movement_BaseballThrow_001.c3d" \
        --alignment openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json --out out/track
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .camera import CAMERAS, load_cameras, project, triangulate
from .detect import detect_ball, find_video, read_min_channel
from .theia import load_hand

CLIP = "OBP_movement_BaseballThrow_001"
FPS = 360.0
# Image-inspected ball pixels from the earlier audit (decoded frame indices, 0-based).
SEED_FRAMES = (980, 981, 985)
SEED_UV = {19: lambda f: (785 - 20 * (f - 978), 519 + 5.6 * (f - 978)),
           22: lambda f: (715 + 17 * (f - 980), 298 - 0.8 * (f - 980))}


def seed_path(K, P):
    """Linear 3D path (mm per frame) through the triangulated seed positions."""
    pts = [triangulate({c: np.array(SEED_UV[c](f)) for c in (19, 22)}, K, P)[0] for f in SEED_FRAMES]
    A = np.column_stack([np.ones(len(pts)), np.array(SEED_FRAMES, float) - 980])
    coef = np.linalg.lstsq(A, np.array(pts), rcond=None)[0]
    return lambda f: coef[0] + coef[1] * (f - 980)


def track_forward(imgs, K, P, frames, path):
    track = {}
    for iteration in range(3):  # re-project from the fitted path and detect again
        track = {}
        for f in frames:
            det = {}
            for c in CAMERAS:
                d = detect_ball(imgs[(c, f)], project(path(f), K[c], P[c]))
                if d is not None and d["dist_px"] <= (30 if iteration == 0 else 14):
                    det[c] = d
            if len(det) < 3:
                continue
            uvs = {c: d["uv"] for c, d in det.items()}
            X, res = triangulate(uvs, K, P)
            if len(det) > 3 and max(res.values()) > 4.0:  # drop one outlier view
                worst = max(res, key=res.get)
                uvs.pop(worst)
                det.pop(worst)
                X, res = triangulate(uvs, K, P)
            track[f] = {"X_mm": X.tolist(), "cams": sorted(det), "residual_px": res, "uv": {c: det[c]["uv"].tolist() for c in det}}
        fs = np.array(sorted(track), float)
        if len(fs) >= 6:
            t = fs - fs.mean()
            coef = np.linalg.lstsq(np.column_stack([np.ones_like(t), t, t ** 2]), np.array([track[int(f)]["X_mm"] for f in fs]), rcond=None)[0]
            path = lambda f, coef=coef, m=fs.mean(): coef[0] + coef[1] * (f - m) + coef[2] * (f - m) ** 2
    return track


def track_backward(imgs, K, P, start, stop, ball_start_mm, hand_m):
    """Frames ``start-1`` down to ``stop``; ``hand_m(frame)`` is the hand origin in metres."""
    rel = np.asarray(ball_start_mm) / 1000 - hand_m(start)
    track = {}
    for f in range(start - 1, stop - 1, -1):
        expect_mm = (hand_m(f) + rel) * 1000
        det = {}
        for c in CAMERAS:
            d = detect_ball(imgs[(c, f)], project(expect_mm, K[c], P[c]), half=30)
            if d is not None and d["dist_px"] <= 22:
                det[c] = d
        if len(det) < 3:
            break
        uvs = {c: d["uv"] for c, d in det.items()}
        X, res = triangulate(uvs, K, P)
        if len(det) > 3 and max(res.values()) > 4.0:
            worst = max(res, key=res.get)
            uvs.pop(worst)
            det.pop(worst)
            X, res = triangulate(uvs, K, P)
        track[f] = {"X_mm": X.tolist(), "cams": sorted(det), "residual_px": res, "uv": {c: det[c]["uv"].tolist() for c in det}}
        rel = X / 1000 - hand_m(f)
    return track


def speed_fits(track):
    fs = np.array(sorted(f for f in track if f >= 974), float)
    Xs = np.array([track[int(f)]["X_mm"] for f in fs])

    def fit(sel, degree):
        t = (fs[sel] - fs[sel].mean()) / FPS
        B = np.column_stack([t ** d for d in range(degree + 1)])
        coef = np.linalg.lstsq(B, Xs[sel], rcond=None)[0]
        return {"speed_mid_kmh": float(np.linalg.norm(coef[1]) * 3.6 / 1000),
                "position_rmse_mm": float(np.sqrt(np.mean(np.sum((Xs[sel] - B @ coef) ** 2, axis=1)))),
                "frames": [int(fs[sel][0]), int(fs[sel][-1])]}

    out = {"linear_all": fit(np.ones(len(fs), bool), 1)}
    for lo, hi in ((974, 990), (980, 996), (986, 1002)):
        sel = (fs >= lo) & (fs <= hi)
        out[f"quadratic_{lo}_{hi}"] = fit(sel, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-root", type=Path, required=True, help="folder 'OBP CV Files' (contains compressed_optitrack)")
    ap.add_argument("--cal-dir", type=Path, required=True)
    ap.add_argument("--theia-c3d", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    K, P = load_cameras(a.cal_dir)
    first, stop = 956, 1006
    imgs = {}
    for c in CAMERAS:
        for f, im in read_min_channel(find_video(a.cv_root, CLIP, c), first, stop).items():
            imgs[(c, f)] = im
    track = track_forward(imgs, K, P, list(range(974, 1004)), seed_path(K, P))
    hand = load_hand(a.theia_c3d, a.alignment)
    track.update(track_backward(imgs, K, P, 974, first, track[974]["X_mm"], lambda f: hand(1, f)[0]))
    res = [r for v in track.values() for r in v["residual_px"].values()]
    summary = {"frames": [min(track), max(track)], "n_frames": len(track),
               "reprojection_px": {"median": float(np.median(res)), "p95": float(np.percentile(res, 95)), "max": float(np.max(res))},
               "speed_fits": speed_fits(track),
               "scope": "image-derived, provisional calibration, unvalidated synchronization; radar value not used"}
    (a.out / "track.json").write_text(json.dumps({str(f): v for f, v in sorted(track.items())}, indent=1))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
