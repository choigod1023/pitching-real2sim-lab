"""Error budget of the marker-based release-speed proxy for the five 3034 pitches.

Proxy: ball centre = wrist midpoint + offset * unit(RFIN - wrist midpoint), speed read at the
recorded release time. The offset comes from the video (not from the radar); the radar speed
is used only as the comparison. Reports how the bias changes with the video release frame,
the release-time shift and the filter cutoff.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .hand_ball_distance import distances
from .pitches import PITCHES, load_hand_markers, recorded_speed_kmh, release_time_s
from .theia import load_hand


def proxy_speed(markers, rate, release_s, offset_m, cutoff_hz=20.0, shift=0):
    sos = butter(4, cutoff_hz, fs=rate, output="sos")
    f = {k: sosfiltfilt(sos, v, axis=0) for k, v in markers.items()}
    mid = (f["RWRA"] + f["RWRB"]) / 2
    x = f["RFIN"] - mid
    x /= np.linalg.norm(x, axis=1)[:, None]
    v = np.gradient(mid + offset_m * x, 1 / rate, axis=0, edge_order=2)
    return float(np.linalg.norm(v[int(round(release_s * rate)) + shift]) * 3.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obp-data", type=Path, required=True, help="baseball_pitching/data")
    ap.add_argument("--track", type=Path, required=True)
    ap.add_argument("--theia-c3d", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = {p: load_hand_markers(a.obp_data, p) for p in PITCHES}
    radar = {p: recorded_speed_kmh(a.obp_data, p) for p in PITCHES}
    release = {p: release_time_s(a.obp_data, p) for p in PITCHES}

    def bias(offset, cutoff=20.0, shift=0):
        return float(np.mean([proxy_speed(*data[p], release[p], offset, cutoff, shift) - radar[p] for p in PITCHES]))

    track = json.loads(a.track.read_text())
    frames, dist = distances(track, load_hand(a.theia_c3d, a.alignment))
    report = {"fixed_15cm_bias_kmh": round(bias(0.15), 1), "release_frame_anchor": {}}
    for offset, d in dist.items():
        for f in (973, 974, 975, 976, 977):
            m = d[frames.index(f)] / 1000
            report["release_frame_anchor"][f"c3d_offset{offset}_frame{f}"] = {"distance_cm": round(m * 100, 1), "bias_kmh": round(bias(m), 1)}
    report["release_time_shift_at_20cm"] = {j: round(bias(0.20, 20.0, j), 1) for j in range(-3, 4)}
    report["cutoff_at_20cm"] = {fc: round(bias(0.20, fc), 1) for fc in (20, 30, 50)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "error_budget.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
