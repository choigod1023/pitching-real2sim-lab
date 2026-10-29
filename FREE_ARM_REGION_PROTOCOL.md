# Free-arm multi-frame image-region reconstruction

Frozen before inspecting projections of the new 20/21 observations or fitting.
Trial: Prime OBP_movement_BaseballThrow_001. This is observation reconstruction,
not athlete movement optimization. The original +1 time correspondence remains.

## Observations and exclusions

Use native 1280x720 raw frames, not skeleton projections, to record approximate
visible wrist/elbow surface regions in cameras 20/21 at frames 940 and 960.
The grid crops are merely magnified raw pixels. One AI reader, motion blur and
surface-to-centre ambiguity remain. Regions are not anatomical ground truth,
confidence intervals or exact bone centres. No shoulder-centre pixel is invented.
Reuse the previously frozen four camera20/21 frame950 regions unchanged.
All 12 observations are fitting data. Hold out every camera17 observation,
and all existing observations at non-fitting frames in cameras19/20/21.
Previously inspected data remain development data, not independent validation.

## One predetermined reconstruction

Frames940/950/960: preserve the source moving shoulder trajectory, allowing one
shared constant translation with each axis bounded +/-30mm. Source arm lengths
are fixed to the medians of those training frames. Optimize elbow/wrist positions
at the three knots, with +/-200mm per-axis computational bounds around source.
These are engineering bounds, not anatomical ROM or measurement uncertainty.
Use a single source-pose initialization, SLSQP, exact two length constraints per
knot, maximum2000 iterations and ftol1e-10. No multistart or result-driven tuning.

Objective: mean squared outside-region pixel distance + 0.001 times mean squared
source-position displacement normalized by10mm + 0.001 times squared root
translation normalized by10mm + 0.01 times mean squared consecutive displacement
changes normalized by10mm. All penalties are engineering regularization choices.
Neither observed ball coordinates, record speed, evaluation views nor future
frames enter optimization or hyperparameter selection.

Interpolate knot displacements with clamped cubic splines with zero endpoint
derivatives, hold endpoint displacements outside940..960, and project onto the
two exact bone lengths at each evaluated time. Preserve source root motion.
Retarget hand orientation by the minimal rotation between source/reconstructed
forearm directions; this preserves relative hand orientation under the chosen
transport but does not identify anatomical wrist rotation or pronation.

## Evaluation and rejection

Report training and held-out outside-region distances, each held-out camera,
and +/-3px annotation-boundary sensitivity. Report root displacement/boundary,
bone-length error and source-position differences. Zero outside a region is
not zero joint error. Use unchanged B hand-relative fit973..976 and optical ball
evaluation977..979, 80 threshold/+1 offset, RK4 free flight plus analytic checks.
Report speed MAE, vector discrepancy and position RMSE, not only one metric.
This still uses observed ball states to fit release; it is not body-only speed
prediction or an athlete intervention. Force/work and muscle feasibility remain
unverified. No new muscle-capacity or shoulder-force claim is authorized.

Engineering adoption requires solver success; exact bone lengths; root not at
bound; nonworsening held-out mean and every camera, all boundary sensitivities,
and all three ball metrics. Scientific gate rules remain unchanged independently.
If rejected, retain every observation and result and investigate the assumption
that failed. Do not alter regions, timing, grip, force or penalties to pass.
