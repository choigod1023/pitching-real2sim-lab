"""Observed-direction geometry test; not predictive shoulder dynamics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ezc3d
from scipy.signal import butter, sosfiltfilt
from pilot import vector, regular_time
from trace_error_audit import DATA


def run():
    meta = pd.read_csv(DATA / "metadata.csv").set_index("session_pitch")
    all_landmarks = pd.read_csv(DATA / "full_sig/landmarks.csv")
    lengths, rows = {}, []
    for pitch in ["3034_1", "3034_2", "3034_3", "3034_4", "3034_5"]:
        g = all_landmarks[all_landmarks.session_pitch.eq(pitch)].sort_values("time")
        times = regular_time(g)
        paths = list((DATA / "c3d").rglob(meta.loc[pitch, "filename_new"]))
        if len(paths) != 1:
            raise ValueError("Nonunique capture")
        c = ezc3d.c3d(str(paths[0]))
        labels = [x.strip() for x in c["parameters"]["POINT"]["LABELS"]["value"]]
        rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
        scale = {"m": 1., "mm": .001, "cm": .01}[c["parameters"]["POINT"]["UNITS"]["value"][0].strip().lower()]
        raw = c["data"]["points"][:, labels.index("CLAV"), :].T
        if not np.isfinite(raw).all() or np.any(raw[:, 3] < 0):
            raise ValueError("Invalid CLAV samples")
        if len(raw) != len(times) or not np.allclose(times, np.arange(len(times))/rate, atol=.000051, rtol=0):
            raise ValueError("Sample alignment mismatch")
        clav = sosfiltfilt(butter(4, 20, fs=rate, output="sos"), raw[:, :3]*scale, axis=0)
        mask = (times >= float(g.fp_10_time.iloc[0])) & (times <= float(g.BR_time.iloc[0]))
        target = vector(g, "shoulder_jc")[mask]
        for name, anchor in [("thorax_prox", vector(g, "thorax_prox")[mask]), ("CLAV", clav[mask])]:
            delta = target-anchor
            distance = np.linalg.norm(delta, axis=1)
            if np.any(distance <= 0):
                raise ValueError("Degenerate link")
            if pitch == "3034_1":
                lengths[name] = float(distance.mean())
            predicted = anchor + delta / distance[:, None] * lengths[name]
            error = np.linalg.norm(predicted-target, axis=1)
            rows.append(dict(pitch=pitch, anchor=name, fixed_training_length_mm=1000*lengths[name],
                             rmse_mm=float(1000*np.sqrt(np.mean(error**2))), max_error_mm=float(1000*error.max())))
    report = dict(rows=rows, speed_used=False, observed_target_direction_used=True,
                  joint_center_identified=False, dynamics_validated=False,
                  scope="Geometry feasibility only; CLAV is a skin marker, not a confirmed sternoclavicular center")
    output = Path(__file__).resolve().parent / "artifacts/baseline-ground-balance/shoulder_preflight.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(row)
    return report


if __name__ == "__main__":
    run()
