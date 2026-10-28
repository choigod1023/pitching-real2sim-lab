"""Bright-blob ball detection and video reading (the ball is the brightest white object)."""
from pathlib import Path

import cv2
import numpy as np


def find_video(cv_root, clip, cam):
    """Path of one camera file, for example ``OBP_movement_BaseballThrow_001`` and camera 19."""
    matches = sorted((Path(cv_root) / "compressed_optitrack").glob(f"{clip}-Camera {cam} *.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file for {clip} camera {cam}, found {len(matches)}")
    return matches[0]


def read_min_channel(path, first, stop):
    """Frames ``first <= f < stop`` as minimum-over-colour images (keeps white, drops colours)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = {}
    for f in range(first, stop):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {f} of {path.name}")
        frames[f] = image.min(axis=2)
    cap.release()
    return frames


def detect_ball(image, expect, half=34, min_peak=110, floor=70.0, fraction=0.55, area=(6, 700)):
    """Nearest bright connected component to ``expect`` inside a square window.

    Returns ``{"uv", "area", "dist_px"}`` or ``None``. The centre is the intensity-weighted
    centroid above the local threshold ``max(floor, fraction * window_max)``.
    """
    x0, y0 = int(expect[0]) - half, int(expect[1]) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half >= image.shape[1] or y0 + 2 * half >= image.shape[0]:
        return None
    roi = image[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(float)
    if roi.max() < min_peak:
        return None
    threshold = max(floor, fraction * roi.max())
    n, labels, stats, _ = cv2.connectedComponentsWithStats((roi > threshold).astype(np.uint8), connectivity=8)
    best = None
    for k in range(1, n):
        if not area[0] <= stats[k, cv2.CC_STAT_AREA] <= area[1]:
            continue
        ys, xs = np.nonzero(labels == k)
        w = roi[ys, xs] - threshold
        uv = np.array([(xs * w).sum() / w.sum() + x0, (ys * w).sum() / w.sum() + y0])
        dist = float(np.linalg.norm(uv - np.asarray(expect, float)))
        if best is None or dist < best["dist_px"]:
            best = {"uv": uv, "area": int(stats[k, cv2.CC_STAT_AREA]), "dist_px": dist}
    return best
