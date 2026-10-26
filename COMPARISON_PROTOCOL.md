# Frozen comparison protocol before this run

Development trial: Prime OBP_movement_BaseballThrow_001, NOT3034_1. Data have been inspected in earlier diagnostics; this is a fit-excluded temporal comparison, not untouched external validation.

Freeze before calculating model results:
- Training ball frames973,974,975,976 from cameras20/21. Evaluation ball frames977,978,979 are never used in fitting.
- All pre-existing detection thresholds60,80,100 and Theia offsets0,1,2. Primary display is previously used threshold80/offset1, never selected from results. Report all9 conditions.
- A: constant local ball offset fitted only to training hand-local ball positions. This is a calibrated fixed-grip comparator in the Theia frame, not the old3034_1 marker-frame15cm model. Do not transplant the latter axis without calibration.
- B: affine local ball offset r(t)=a+b*t fitted only on training positions; predict future world ball positions with observed evaluation hand transforms. Conditional kinematic forecast, not a force/muscle/contact model or independent motion prediction.
- C: constant world velocity extrapolation fitted only on training ball positions. Simple tracking comparator to check whether B adds value beyond extrapolation.
- Evaluate identical two future intervals977→978,978→979: mean predicted speed, mean observed candidate speed, mean absolute speed discrepancy, vector velocity discrepancy and3D position RMSE. Also separate discrepancy to the independently documented62.7mph scalar for this trial; metadata timing/location not established. No parameter fitted to radar.
- Camera19 detections from the prior holdout audit are fit-excluded third-view image checks. Shared provisional calibration and container-index alignment remain assumptions. Report failed threshold160 detections and allsuccessful thresholds100/130; do not tune to those pixels.
- No gain/causal athlete conclusion. No claim reduction of the old16.73km/h gap. Fewer than1km/h difference on these developmental candidates is not1km/h validated accuracy.

