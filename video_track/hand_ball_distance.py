"""Distance between the video-tracked ball and the Theia hand origin, and the ball speed profile.

No radar value is used. The distance depends on the trial alignment between Theia and the camera
frame (median residual 1.7 cm in the published fit) and on the C3D-minus-video frame offset, so
offsets 0, 1 and 2 are all reported.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .theia import load_hand


def distances(track, hand, offsets=(0, 1, 2)):
    frames = sorted(int(f) for f in track)
    ball = np.array([track[str(f)]["X_mm"] for f in frames]) / 1000
    return frames, {o: np.linalg.norm(ball - hand(o, np.array(frames)), axis=1) * 1000 for o in offsets}


def speed_profile(track, window=5, fps=360.0):
    """Ball speed (km/h) from a sliding linear fit over ``window`` consecutive frames."""
    frames = sorted(int(f) for f in track)
    X = {f: np.array(track[str(f)]["X_mm"]) for f in frames}
    half, out = window // 2, {}
    for c in frames[half:-half]:
        w = [f for f in range(c - half, c + half + 1) if f in X]
        if len(w) < window:
            continue
        t = np.array(w, float) / fps
        coef = np.linalg.lstsq(np.column_stack([np.ones(window), t - t.mean()]), np.array([X[f] for f in w]), rcond=None)[0]
        out[c] = float(np.linalg.norm(coef[1]) * 3.6 / 1000)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", type=Path, required=True, help="track.json from video_track.track_ball")
    ap.add_argument("--theia-c3d", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    track = json.loads(a.track.read_text())
    frames, d = distances(track, load_hand(a.theia_c3d, a.alignment))
    speed = speed_profile(track)
    report = {"frames": frames, "distance_mm_by_c3d_offset": {str(o): v.round(1).tolist() for o, v in d.items()},
              "ball_speed_kmh_5frame": {str(f): round(v, 1) for f, v in speed.items()}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "hand_ball_distance.json").write_text(json.dumps(report, indent=1))
    for o, v in d.items():
        print(f"C3D offset {o}: " + " ".join(f"{f}:{x:.0f}" for f, x in zip(frames, v) if f <= 980))
    print("ball speed (km/h): " + " ".join(f"{f}:{v:.0f}" for f, v in speed.items() if f <= 990))


if __name__ == "__main__":
    main()
