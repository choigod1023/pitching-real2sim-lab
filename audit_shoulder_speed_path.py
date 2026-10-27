"""Oracle shoulder translation replacement: kinematic error attribution only."""
import json
from pathlib import Path
import numpy as np
import mujoco
from baseline_calibration import endpoint
from trace_error_audit import ObservedPitch
from torque_arm import derivatives


def run():
    observation = ObservedPitch("3034_4")
    rows = []
    for folder in ["baseline-calibration", "baseline-dequantized"]:
        for mode in ["nominal", "frozen_geometry_mass"]:
            root = Path(__file__).resolve().parent / "artifacts" / folder
            model = mujoco.MjModel.from_xml_path(str(root / (mode + ".xml")))
            with np.load(root / (mode + "_states.npz")) as states:
                q, times = states["reference"], states["time"]
            # Saved reference has no padding: exclude endpoints for a stable
            # interior derivative diagnostic, and explicitly report its time.
            velocity, _ = derivatives(model, q, float(np.median(np.diff(times))))
            i = len(times)-4
            vb = endpoint(model, q[i], velocity[i], "ball_center")
            vs = endpoint(model, q[i], velocity[i], "shoulder_jc")
            measured_vs = observation.points(times[i], 1)[0]
            replaced = vb + measured_vs-vs
            speed = lambda x: float(np.linalg.norm(x)*3.6)
            rows.append(dict(folder=folder, mode=mode, milliseconds_before_BR=float((observation.b-times[i])*1000),
                             reference_proxy_kmh=speed(vb), observed_shoulder_replacement_proxy_kmh=speed(replaced),
                             proxy_change_kmh=speed(replaced)-speed(vb),
                             shoulder_velocity_difference_kmh=speed(measured_vs-vs)))
    report = dict(pitch="3034_4", rows=rows, actual_ball_speed_prediction=False, force_constraints_checked=False,
                  scope="Replace shoulder translation velocity only using observed shoulder; relative arm/hand velocity fixed. Interior pre-release sample, not BR speed or new forward simulation.")
    output = Path(__file__).resolve().parent / "artifacts/baseline-ground-balance/shoulder_speed_path.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
