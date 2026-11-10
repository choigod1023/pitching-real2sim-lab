"""Observed launch state with a controlled effective finger-contact moment arm."""
import argparse
import json
from itertools import product
from math import cos, pi, radians, sin, sqrt
from pathlib import Path

import numpy as np

from experiments.skenes_fastball_scenarios.study import lift_model, release_state
from pitch_outcomes import flight

MASS = .145
RADIUS = .0366
INERTIA = .4 * MASS * RADIUS**2
DURATION = .008
RPM = 60 / (2*pi)
REFERENCE_ANGLE = radians(12)
PRIMARY_SHIFT_MM = 4.
NOMINAL = (.10, .35, .85, .35)  # impulse, friction limit, active-spin fraction, drag


def impulse_fraction(phase):
    """Integral and second integral of a unit-area symmetric triangular pulse."""
    phase = np.asarray(phase)
    return (
        np.where(phase <= .5, 2 * phase**2, 1 - 2 * (1-phase)**2),
        np.where(phase <= .5, (2/3) * phase**3, phase - .5 + (2/3) * (1-phase)**3),
    )


def contact_pulse(position, velocity, rpm, axis_deg, shift_mm, impulse_ns, mu, active):
    direction = velocity / np.linalg.norm(velocity)
    _, axis = lift_model(rpm, active, axis_deg, np.linalg.norm(velocity))
    transverse = axis - np.dot(axis, direction) * direction
    transverse /= np.linalg.norm(transverse)
    observed_spin = rpm / RPM * (active*transverse + sqrt(1-active**2)*direction)
    lever_plane = np.cross(direction, transverse)
    impulse = impulse_ns * direction

    def arm(angle):
        return RADIUS * (-cos(angle)*direction + sin(angle)*lever_plane)

    reference_arm = arm(REFERENCE_ANGLE)
    current_arm = arm(REFERENCE_ANGLE + shift_mm/(1000*RADIUS))
    normal_ns = -float(np.dot(current_arm/RADIUS, impulse))
    tangent_ns = float(np.linalg.norm(impulse + normal_ns*current_arm/RADIUS))
    trace = dict(
        surface_shift_mm=shift_mm, impulse_ns=impulse_ns, normal_impulse_ns=normal_ns,
        tangent_impulse_ns=tangent_ns, mu_limit=mu,
        mu_required=tangent_ns/normal_ns if normal_ns > 0 else None,
        peak_force_n=2*impulse_ns/DURATION, arm_m=current_arm.tolist(),
    )
    if normal_ns <= 0 or tangent_ns > mu * normal_ns + 1e-12:
        return trace | {"status": "friction_rejected"}

    pre_velocity = velocity - impulse/MASS
    pre_position = position - DURATION*(pre_velocity + impulse/(2*MASS))
    pre_spin = observed_spin - np.cross(reference_arm, impulse)/INERTIA
    torque_impulse = np.cross(current_arm, impulse)
    phase = np.linspace(0., 1., 17)
    fraction, integrated = impulse_fraction(phase)
    v = pre_velocity + fraction[:, None]*impulse/MASS
    p = pre_position + phase[:, None]*DURATION*pre_velocity + integrated[:, None]*DURATION*impulse/MASS
    w = pre_spin + fraction[:, None]*torque_impulse/INERTIA
    rpm_total = np.linalg.norm(w, axis=1)*RPM
    rpm_active = np.linalg.norm(w - (w @ direction)[:, None]*direction, axis=1)*RPM
    return trace | dict(
        status="computed", release_speed_kmh=float(np.linalg.norm(v[-1])*3.6),
        transverse_axis=transverse.tolist(), active_rpm=float(rpm_active[-1]),
        rpm=float(rpm_total[-1]), t_ms=((phase-1)*DURATION*1000).tolist(),
        center_xyz_m=p.tolist(), velocity_m_s=v.tolist(), omega_rad_s=w.tolist(),
        rpm_trace=rpm_total.tolist(), torque_impulse_nms=torque_impulse.tolist(),
        rotational_work_j=float(.5*INERTIA*(w[-1]@w[-1] - pre_spin@pre_spin)),
    )


def simulate(pitch, shift_mm, impulse_ns, mu, active, cd, dt=.001):
    position, velocity, distance = release_state(pitch)
    row = pitch["row"]
    model = contact_pulse(position, velocity, float(row["release_spin_rate"]), float(row["spin_axis"]),
                          shift_mm, impulse_ns, mu, active)
    if model["status"] != "computed":
        return model
    p, v = model["center_xyz_m"][-1], model["velocity_m_s"][-1]
    cl, _ = lift_model(model["active_rpm"], 1., float(row["spin_axis"]), np.linalg.norm(v))
    axis = model["transverse_axis"]
    path, elapsed = flight(p, v, axis, cl, cd, distance, dt)
    no_lift, _ = flight(p, v, axis, 0., cd, distance, dt)
    return model | dict(
        cl=cl, flight_s=elapsed, dt_s=dt, flight_xyz_m=path[:, :3].tolist(),
        plate_xz_cm=(path[-1,[0,2]]*100).tolist(),
        lift_xz_cm=((path[-1,[0,2]]-no_lift[-1,[0,2]])*100).tolist(),
    )


def differences(baseline, candidate):
    if candidate["status"] != "computed":
        return None
    return {key: (np.asarray(candidate[key])-baseline[key]).tolist() for key in (
        "release_speed_kmh", "rpm", "active_rpm", "rotational_work_j", "plate_xz_cm", "lift_xz_cm",
    )}


def study(pair):
    low, high = pair["low"], pair["high"]
    baseline, candidate = [simulate(low, s, *NOMINAL, dt=.0005) for s in (0, PRIMARY_SHIFT_MM)]
    sweep = []
    for shift in (-6., -4., -2., 0., 2., 4., 6.):
        alt = simulate(low, shift, *NOMINAL, dt=.0005)
        sweep.append({"shift_mm": shift, "mu_required": alt["mu_required"],
                      "status": alt["status"], "delta": differences(baseline, alt)})
    grid = []
    for impulse, mu, active, cd in product((.06,.10,.14), (.25,.35,.50), (.70,.85,1.), (.30,.35,.40)):
        ref = simulate(low, 0., impulse, mu, active, cd)
        alt = simulate(low, PRIMARY_SHIFT_MM, impulse, mu, active, cd)
        grid.append(dict(impulse_ns=impulse, mu=mu, active=active, cd=cd,
                         status=alt["status"], delta=differences(ref, alt)))
    coarse = [simulate(low, s, *NOMINAL)["plate_xz_cm"] for s in (0, PRIMARY_SHIFT_MM)]
    convergence = max(float(np.max(np.abs(np.asarray(a) - b["plate_xz_cm"])))
                      for a,b in zip(coarse, (baseline,candidate), strict=True))
    observed = {key: [p[key] for p in (low,high)] for key in ("speed_kmh","spin_rpm","vertical_break_cm")}
    return dict(
        scope="conditional_effective_contact_no_measured_finger_path",
        play_ids=[p["play_id"] for p in (low,high)], observed=observed,
        nominal={"baseline":baseline,"candidate":candidate}, delta=differences(baseline,candidate),
        arc_sweep=sweep, parameter_grid=grid, max_timestep_change_cm=convergence,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cohort", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    cohort = json.loads(args.cohort.read_bytes())
    pair = next(p for game in cohort["games"] for p in game["chosen"])
    result = study(pair)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    accepted = [r["delta"] for r in result["parameter_grid"] if r["delta"] is not None]
    for key in ("rpm", "rotational_work_j", "lift_xz_cm"):
        values = np.array([r[key] for r in accepted])
        print(key, "range", np.min(values,axis=0), np.max(values,axis=0))
    print(f"admissible grid cases {len(accepted)}/{len(result['parameter_grid'])}")
    print(json.dumps({"delta":result["delta"],"max_timestep_change_cm":result["max_timestep_change_cm"]}))


if __name__ == "__main__":
    main()
