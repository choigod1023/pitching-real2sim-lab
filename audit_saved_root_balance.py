"""Compare saved baseline reference dynamics with measured ground wrench.
This checks numerical reconstruction, not ball-speed accuracy or foot contact.
"""
import json
from pathlib import Path
import numpy as np
import ezc3d
import mujoco
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from baseline_ground_balance import wrench_at_root
from trace_error_audit import ObservedPitch
from torque_arm import derivatives


def run():
    obs = ObservedPitch("3034_4")
    capture = ezc3d.c3d(obs.ledger["source_paths"]["c3d"], extract_forceplat_data=True)
    rate = float(capture["parameters"]["ANALOG"]["RATE"]["value"][0])
    sos = butter(4, 40, fs=rate, output="sos")
    rows = []
    for folder in ["baseline-calibration", "baseline-dequantized"]:
        for mode in ["nominal", "frozen_geometry_mass"]:
            root = Path(__file__).resolve().parent / "artifacts" / folder
            model = mujoco.MjModel.from_xml_path(str(root / (mode + ".xml")))
            with np.load(root / (mode + "_states.npz")) as states:
                q, times = states["reference"], states["time"]
            velocity, acceleration = derivatives(model, q, float(np.median(np.diff(times))))
            forces, moments, centers = [], [], []
            for plate in capture["data"]["platform"]:
                if (plate["unit_force"], plate["unit_moment"], plate["unit_position"]) != ("N", "Nm", "m"):
                    raise ValueError("Unexpected force platform units")
                samples = np.arange(plate["force"].shape[1]) / rate
                forces.append(CubicSpline(samples, sosfiltfilt(sos, plate["force"].T, axis=0))(times))
                moments.append(CubicSpline(samples, sosfiltfilt(sos, plate["moment"].T, axis=0))(times))
                centers.append(plate["corners"].mean(1))
            forces, moments = np.array(forces), np.array(moments)
            data = mujoco.MjData(model)
            required, residual = [], []
            for i in range(3, len(times) - 3):
                data.qpos[:], data.qvel[:], data.qacc[:] = q[i], velocity[i], acceleration[i]
                mujoco.mj_inverse(model, data)
                demand = data.qfrc_inverse[:6].copy()
                required.append(demand)
                residual.append(demand - wrench_at_root(model, data, forces[:, i], moments[:, i], centers))
            required, residual = np.array(required), np.array(residual)
            if not np.isfinite(residual).all():
                raise ValueError("Non-finite residual")
            def rms(x):
                return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
            rows.append(dict(folder=folder, mode=mode, required_force_n=rms(required[:, :3]),
                             residual_force_n=rms(residual[:, :3]), residual_torque_nm=rms(residual[:, 3:])))
    report = dict(pitch="3034_4", filter_hz=40, excluded_boundary_samples=3, rows=rows,
                  scope="Saved reference inverse dynamics; no foot contact validation or speed correction")
    output = Path(__file__).resolve().parent / "artifacts/baseline-ground-balance/rounding_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
