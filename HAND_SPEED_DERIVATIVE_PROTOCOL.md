# Same-pitch launch-speed comparison — frozen before execution

Goal: directly test whether the hand–ball model's launch-velocity calculation can reduce subsequent speed/trajectory prediction discrepancies. This is an exploratory same-pitch development diagnostic while the original data gate remains unpassed; it is not permission to skip the gate or recommend posture changes.

## Fixed source and split

Use only existing Prime `OBP_movement_BaseballThrow_001` observations. Train video frames973–976, evaluate977–979, exactly as the existing physical free-ball comparison. Legacy primary condition: threshold80,C3D offset+1. Report all threshold60/80/100 ×offset0/1/2 combinations. No condition is chosen by agreement with speed or future ball positions. Different Edger/3034 trials are excluded.

Verify original ball/body/alignment hashes. Source62.7MPH label equals100.9058688km/h, but radar instrument/location/time definition is unresolved: its discrepancy is not certified absolute speed error. Triangulated evaluation-ball positions also have provisional calibration/detection uncertainty; their interval speed is a conditional reference, not ground truth.

## Models and source of change

- A: original rigid hand offset, retained as context.
- B: original affine hand-relative offset with last two training predicted positions differenced for launch velocity.
- C: original training-ball linear world trajectory, launch velocity from its fitted slope.
- D1: same B launch position and affine relative motion, but analytic velocity `v_origin + omega_world x (R*r) + R*r_dot`. Training origin/log-rotation use degree1 polynomials constrained through the observed last hand pose.
- D2 (**new primary**): same as D1 with degree2 origin/log-rotation. No future hand or ball is an input. Let u(t)=log(R_end^T R(t)); u(end)=0, hence omega_world(end)=R_end*u_dot(end). No unconstrained angular extrapolation or force tuning.

Only the launch velocity changes between B and D; verify their launch positions agree numerically. Do not change gravity, hand pose, release frame, mass, grip, camera calibration or measured speed. Frame976 launch is a diagnostic boundary, not a newly verified physical release.

Every state is passed to the same MuJoCo free sphere RK4 model; compare step sizes and the analytical gravity trajectory. Report km/h speed, vector and position discrepancy; source-label gap; all velocity components and failures.

## Decision rules

Report D2 versus B on identical inputs and evaluation frames. A conditional kinematic improvement requires all three speed MAE, velocity-vector MAE and position RMSE to improve; no speed-only selection. Added predictive value versus the ball-only C baseline requires the same non-worsening checks. D1 is a secondary diagnostic, not an alternative primary selected after seeing results.

No result here establishes a new-athlete generalization, force/work/comfort/athlete benefit, or completion of the data/scientific gate. The model uses observed training-ball positions to calibrate its relative state. A smaller source-label gap alone cannot validate the hand model.

## Numerical source check before outcome calculation

The float32 C3D rotations have maximum orthogonality residual5.4823e-8, consistent with their storage precision; the basis itself is proper to1.6e-15. To evaluate an SO(3) log, use nearest proper rotation by SVD **only** when the defect is <=16*float32 epsilon and determinant positive; reject larger defects. Apply the same numerical canonicalization to baseline B and D, retain B_raw and quantify any launch-state difference. This is storage-roundoff cleanup, not an anatomical pose adjustment or a relaxed scientific gate. The numerical issue was identified before evaluating new velocity outcomes.
