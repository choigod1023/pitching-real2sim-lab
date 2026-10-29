# Dense camera1–3 connection audit

Continuation of PR98, not a change to its thresholds or data split. Frozen before stereo fitting. Raw source video files are reused read-only from the preceding worktree. All new caches/outputs live under this worktree's `artifacts/dense-camera-link`.

- Existing coarse overlap at576 motivates dense frames540..612 inclusive, step1, camera1 and3. Inspect all attempted frames and method failures, preserving source-hash and implementation-version provenance. Counts alone are not independent calibration accuracy.
- Primary detector: native-resolution SB without native cornerSubPix. Source720/subPix is the declared fallback comparison; report both. Do not move the fixed cube or ground-plane image annotations.
- For this new camera-pair experiment, frames divisible by5 are withheld; other frames train. This rule is separate from PR98's camera1–4 modulo60 rule, which is unchanged. Require >=8shared, >=5training, >=2withheld. If a method lacks observations, report it as insufficient rather than changing the split/counts.
- Hold original camera intrinsics fixed. Compare four global grid-axis reversal choices on training observations only and record all training scores. No speed or static-target lengths choose the permutation. Visually inspect representative paired observations.
- Report withheld board reprojection, every100mm adjacency, positive depths, board-normal angle span and translation span. Narrow adjacent-frame observations are development evidence, not independent athlete validation.
- Static cube registration is explicitly frozen to PR97's horizontal rotations camera1=0, camera3=1; it remains a provisional correspondence convention, not lab/C3D axes. Camera3 provides only six visible outer corners; do not invent the other two or claim all12 cube edges were validated if only a subset is reconstructible. Report each available edge and each projection.
- Keep <=2px and <=1% on all measured known lengths, positive depth. CS-200 mounting/sphere-centre identity remains UNKNOWN even if a conditional calculation matches its drawing. No hand/ball prediction or scientific advance until original gate checks pass.
- If fixed-intrinsic fitting fails, inspect ordering/time and original source geometry before any new model. ±1frame diagnostics, if run, must all be reported and may not be chosen to match radar or cube dimensions. The physical360Hz clock and67/70 versus69.1MPH reference remain unresolved.

Method sources already reviewed in preceding work: OpenCV4.13 calibration reference; Zhang TPAMI2000; Strobl/Hirzinger ICCV2011 Workshop (imperfect target calibration). New target-shape corrections require a separately frozen training-only experiment and independent unchanged metric checks.
