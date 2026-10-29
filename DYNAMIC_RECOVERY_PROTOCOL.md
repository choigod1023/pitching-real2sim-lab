# Dynamic checkerboard recovery — frozen before calculation

Continue the failed metric audit in PR97; do not accept a 3D throwing result until the original metric checks pass. Source processing code: public openbiomechanics `computer_vision/calibration/scripts/analyze_calibration.py`. Official numerical reference: https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html .

## Failure hypotheses and actual tests

1. Downsampling to width 720 and then running a native 11x11 `cornerSubPix` neighbourhood may move narrowly spaced corners. Compare that exact baseline with native-resolution SB detection and with enlarged target-region SB without subsequent native `cornerSubPix`. Enlargement provides no new image resolution. Measure displacement, minimum spacing and withheld metric results; do not infer causality from a low fitting RMS.
2. Same-index video frames or grid ordering may be inconsistent across cameras. Use zero frame shift as primary. Report board-pose consistency and overlap for all camera pairs; keep failed/insufficient pairs. Check 0/180-degree and axis-reversal candidates using training board observations, not cube or speed values. Any selected permutation must be recorded and visually inspected. If needed, +/-1 frame are engineering diagnostics, not a synchronization confidence interval.
3. If a detector change alone is inadequate, a second stage may refit intrinsics using only training frames and the existing per-camera model classes (camera1 zero distortion square pixels; camera4 k1 only square pixels). Fix the model class in advance; do not choose using cube or throwing speed. Preserve original intrinsics as a separate baseline and report coverage/parameter shifts.

## Frozen frame splits and checks

- Initial coverage probe: frames 0,24,... through 1320 in all four dynamic checkerboard videos. No hand/ball or radar values are used. Record every attempted frame, success/failure and runtime. Detection cache must be keyed by video hash, frame, method and implementation hash.
- First quantitative recovery uses camera1–4 at frames 0,12,...,552. Frame indices divisible by 60 are withheld; all others train stereo parameters. Require >=8 shared observations, >=5 training and >=2 withheld views. This development set is temporally correlated and is not independent athlete validation.
- Hold intrinsics fixed in the first comparison. The original cube manual observations and third-round terminal observations are frozen, hashed and shared by every method. Do not improve results by moving them after seeing the comparison. Unconfirmed CS-200 sphere/mount identities remain UNKNOWN.
- For withheld moving-board views, report epipolar and triangulated adjacent-spacing (100 mm) errors. For the separate static cube, report all 12 edges (698.5 mm), all corner reprojections and depths. For the separate ground-plane target report 150/200/250 mm distances conditionally, without using them in continuous fitting.
- Keep previous limits: <=2px relevant reprojection RMS, every known target length within 1%, positive depth. A detector or calibration is not accepted merely for beating the previous average. Geometry checks and original scientific gate remain required.
- Do not call these discrepancies true instrument error or km/h confidence bounds. No speed reference is read. Prime, Edgertronic and 3034 are separate trials and cannot be combined as a before/after correction.

## Recovery routine

For each genuine failure, save the source/inputs, scientific failure type, evidence-based recovery hypothesis, actual code/data change, numerical comparison, validation outcome and next executable action. Reuse completed results and raw source hashes. Run relevant code tests; record scientific UNKNOWN/FAIL separately. Continue the next stage in the same execution only if its gate passes. Stop and ask only for an actual missing decision/measurement after reasonable public alternatives have been tried. Normal work and recovered errors stay quiet in scheduled runs; report completion or unresolved actionable blockage.
