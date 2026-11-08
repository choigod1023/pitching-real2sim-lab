"""Theia hand pose in the camera frame using the supplied contributor-fitted alignment."""
import json
from pathlib import Path

import ezc3d
import numpy as np


def load_hand_pose(c3d_path, alignment_path, segment="l_hand_4X4", fps=360.0):
    """Return a callable giving (Nx3 origins in metres, Nx3x3 rotations).

    Local axes are the source Theia segment axes, not anatomical finger axes.
    The alignment scales translations only; rotations remain dimensionless.
    Frame offsets are C3D indices minus decoded video indices. No interpolation,
    negative indexing, gap filling or orientation repair is performed here.
    """
    c = ezc3d.c3d(str(c3d_path))
    labels = c["parameters"]["ROTATION"]["LABELS"]["value"]
    a = c["data"]["rotations"][:, :, labels.index(segment), :]
    al = json.loads(Path(alignment_path).read_text())
    q, t, s = np.asarray(al["rotation"]), np.asarray(al["translation"]), al["scale"]
    rate = float(c["parameters"]["ROTATION"]["RATE"]["value"][0])
    if not np.isfinite(fps) or fps <= 0 or not np.isclose(rate, fps):
        raise ValueError("Theia/video rate mismatch")
    if al.get("trial", Path(c3d_path).stem) != Path(c3d_path).stem:
        raise ValueError("Alignment belongs to a different trial")
    if (q.shape != (3, 3) or t.shape != (3,) or not np.isfinite(q).all()
            or not np.isfinite(t).all() or not np.isfinite(s) or s <= 0
            or not np.allclose(q.T @ q, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(q), 1, atol=1e-5)):
        raise ValueError("Invalid similarity alignment")

    def pose(frame_offset, video_frames):
        frames = np.atleast_1d(video_frames)
        if (frames.ndim != 1 or not np.isfinite(frames).all()
                or not np.isfinite(frame_offset) or int(frame_offset) != frame_offset
                or not np.equal(frames, np.floor(frames)).all()):
            raise ValueError("Frame indices and offset must be integers")
        idx = frames.astype(int) + int(frame_offset)
        if np.any(idx < 0) or np.any(idx >= a.shape[-1]):
            raise ValueError("Frame outside Theia data")
        origins = (s * (q @ a[:3, 3, idx])).T + t
        rotations = np.einsum("ij,jkn->nik", q, a[:3, :3, idx])
        if (not np.isfinite(origins).all() or not np.isfinite(rotations).all()
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4, rtol=0)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4, rtol=0)):
            raise ValueError("Invalid hand pose; no repair or gap filling")
        return origins, rotations

    return pose


def load_hand(c3d_path, alignment_path, segment="l_hand_4X4"):
    """Backward-compatible estimated wrist origin, with the same frame convention."""
    pose = load_hand_pose(c3d_path, alignment_path, segment)

    def origin(frame_offset, video_frames):
        return pose(frame_offset, video_frames)[0]

    return origin
