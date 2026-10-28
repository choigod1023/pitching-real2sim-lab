"""Force-based RK4 rollout of the observed arm/hand/attached-ball model for the five 3034 pitches.

Same model as ``grip_identifiability.py``; only the wrist-to-ball distance (grip) changes. The
reference states are rebuilt here (20 Hz marker filter, step 1/2880 s). The radar speed is used
only for the comparison columns. Run from the repository root: ``python -m video_track.simulate_grip``.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from coupled_payload import inverse_and_rollout, model_for, run as prepare_reference
from .pitches import PITCHES, recorded_speed_kmh

GRIPS_M = (0.15, 0.193, 0.200, 0.211)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obp-data", type=Path, required=True, help="baseball_pitching/data")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--grips-m", type=float, nargs="+", default=list(GRIPS_M))
    a = ap.parse_args()
    results, radar = {}, {p: recorded_speed_kmh(a.obp_data, p) for p in PITCHES}
    for p in PITCHES:
        folder = a.work_dir / p
        folder.mkdir(parents=True, exist_ok=True)
        prepare_reference(a.obp_data, p, folder, cutoff_hz=20.0, steps=(1 / 2880,))
        with np.load(folder / "reference-0.npz") as ref:
            q, lengths, dt = ref["qpos"].copy(), ref["lengths"].copy(), float(ref["dt"])
        results[p] = {}
        for g in a.grips_m:
            model = model_for(lengths, dt, grip=(g, 0.0, 0.0))
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
            out, _ = inverse_and_rollout(model, q, dt)
            results[p][g] = out["conditional_release_speed_kmh"]
    print("pitch  radar | " + " | ".join(f"{g * 100:.1f}cm (error)" for g in a.grips_m))
    for p in PITCHES:
        print(p, f"{radar[p]:6.1f} | " + " | ".join(f"{results[p][g]:6.1f} ({results[p][g] - radar[p]:+5.1f})" for g in a.grips_m))
    r = np.array([radar[p] for p in PITCHES])
    for g in a.grips_m:
        e = np.array([results[p][g] for p in PITCHES]) - r
        print(f"grip {g * 100:.1f} cm: mean bias {e.mean():+.2f}, MAE {np.abs(e).mean():.2f}, "
              f"bias-removed MAE {np.abs(e - e.mean()).mean():.2f} (mean-radar predictor MAE {np.abs(r - r.mean()).mean():.2f})")
    a.work_dir.mkdir(parents=True, exist_ok=True)
    (a.work_dir / "simulate_grip.json").write_text(json.dumps({p: {str(g): v for g, v in d.items()} for p, d in results.items()}, indent=1))


if __name__ == "__main__":
    main()
