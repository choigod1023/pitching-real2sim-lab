"""Theia3D hand origin placed in the camera reference frame via the published trial alignment."""
import json
from pathlib import Path

import ezc3d
import numpy as np


def load_hand(c3d_path, alignment_path, segment="l_hand_4X4"):
    """Return ``origin(frame_offset, video_frames)`` giving the hand origin in metres.

    ``frame_offset`` is the C3D index minus the video index (the alignment file documents 1).
    The origin is Theia's estimated wrist, not a measured joint centre.
    """
    c = ezc3d.c3d(str(c3d_path))
    labels = c["parameters"]["ROTATION"]["LABELS"]["value"]
    a = c["data"]["rotations"][:, :, labels.index(segment), :]
    al = json.loads(Path(alignment_path).read_text())
    q, t, s = np.asarray(al["rotation"]), np.asarray(al["translation"]), al["scale"]

    def origin(frame_offset, video_frames):
        idx = np.atleast_1d(video_frames) + frame_offset
        return (s * (q @ a[:3, 3, idx])).T + t

    return origin
