# Independent motion-search verification

Executed `../.venv/Scripts/python.exe audit_motion_search.py` on 2026-10-26. Local machine-readable output: `artifacts/constrained-motion-search/independent-audit.json`.

- Verified 423 distinct coarse evaluations against the 500-evaluation cap, normalized parameter bounds, frozen protocol and source hashes, deterministic 64-point Sobol initialization, identical initial position/velocity, and independently reconstructed budget predicates for every successful evaluation.
- Read optimizer implementation: six jointly varied parameters, 12 signed coordinate probes and three SLSQP starts. SLSQP terminated at baseline or numerically near baseline in 1, 13 and 20 iterations. This is a bounded nonlinear optimization run, not proof of a global optimum or of physiological infeasibility.
- Independently integrated baseline, selected result and diagnostic with RK4 at half step using stored controls. Maximum disagreement in positions, velocities, positive/negative work and endpoint speed was exactly zero in these replays. The independent work integration handles power sign crossings separately.
- Selected feasible result is the unchanged baseline: **113.820741 km/h; gain 0 km/h**. The diagnostic is not an accepted optimum.
- The fastest pose-screen-passing diagnostic reaches **121.316042 km/h (+7.495301 km/h)**. Independently computed actual baseline-relative maximum joint rotation is **1.210127 degrees**, maximum attached ball-center displacement **13.957593 mm**, and release velocity direction change **0.647229 degrees**. It passes the geometric screens but fails force/work constraints. Shoulder translation net work increases **17.393988 J**, shoulder rotation net work **35.952191 J**, and shoulder rotational torque-norm peak **165.479941 Nm**. These are model coordinates and demands, not measured muscle strength.
- Coarse-to-fine endpoint-speed changes are **0.000011342 km/h** for baseline and **0.000018190 km/h** for diagnostic. These assess integration resolution with fixed reference/control intervals, not measurement accuracy or reference-derivative convergence.

## Reference tracking is still imperfect

The 3-degree geometric screen compares each actual rollout with the baseline actual rollout. It does **not** imply tracking the measured reference within 3 degrees.

Baseline maximum reference-to-rollout rotation discrepancies are shoulder **0.623854 degrees**, elbow **0.263819 degrees**, marker roll **3.564607 degrees**, and wrist **2.299888 degrees**. For the diagnostic they are **0.691163**, **0.305535**, **4.262344**, and **2.932697 degrees**, respectively. Marker roll is not an anatomically calibrated forearm angle. The endpoint displacement screen is a rigidly attached ball-center comparison, not a measured hand-marker constraint.

The recorded speed for pitch 3034_4 is **128.425651 km/h**, leaving **14.604910 km/h** above the baseline model BR speed. Measurement time/location remain unresolved. Numerical replay agreement therefore does not establish zero real-world error, comfort, equal physiological capability, accuracy at the plate, or actual training benefit.
