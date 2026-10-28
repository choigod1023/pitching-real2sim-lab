"""Load the five session-3034 pitches (marker positions, release time, recorded speed) from OBP data."""
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd

PITCHES = ("3034_1", "3034_2", "3034_3", "3034_4", "3034_5")
MPH_TO_KMH = 1.609344


def recorded_speed_kmh(data, pitch):
    meta = pd.read_csv(Path(data) / "metadata.csv").set_index("session_pitch")
    return float(meta.loc[pitch].pitch_speed_mph) * MPH_TO_KMH


def release_time_s(data, pitch):
    lm = pd.read_csv(Path(data) / "full_sig/landmarks.csv", usecols=["session_pitch", "BR_time"])
    return float(lm[lm.session_pitch.eq(pitch)].BR_time.iloc[0])


def load_hand_markers(data, pitch, names=("RFIN", "RWRA", "RWRB")):
    """Raw hand markers in metres at the C3D rate (no gap filling; missing data raises)."""
    data = Path(data)
    meta = pd.read_csv(data / "metadata.csv").set_index("session_pitch")
    paths = list((data / "c3d").rglob(meta.loc[pitch].filename_new))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one C3D for {pitch}, found {len(paths)}")
    c = ezc3d.c3d(str(paths[0]))
    labels = [v.strip() for v in c["parameters"]["POINT"]["LABELS"]["value"]]
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[c["parameters"]["POINT"]["UNITS"]["value"][0].strip().lower()]
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    out = {}
    for n in names:
        raw = c["data"]["points"][:, labels.index(n), :].T
        if not np.isfinite(raw).all() or np.any(raw[:, 3] < 0):
            raise ValueError(f"Missing marker {n}")
        out[n] = raw[:, :3] * scale
    return out, rate
