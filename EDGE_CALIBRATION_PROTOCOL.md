# Edgertronic calibration audit — 2026-10-29

This protocol is frozen before fitting the newly downloaded static targets. It is an engineering calibration audit, not a replacement for the research-gates scientific criteria.

## Inputs and split

- Public source index: `https://docs.google.com/spreadsheets/d/1NhpF8DnfBdio_xsU7B44KNuHghtePDp-d3juvCYNm9Q/edit`. Readme B14 specifies 27.5 in (698.5 mm) between outer red joint centres; middle joints are halfway. The 27-point lattice is a coordinate system, **not a claim that all lattice sites have physical markers**.
- Use published checkerboard intrinsics without refitting them to throwing speed. Source: `openbiomechanics/computer_vision/calibration/results/calibration_results.json` in the adjacent public repository. Save its hash and exact parameters.
- Frame 120 cube outer-joint image centres fit each camera pose. Record manually placed image coordinates and occlusions. Frame 0 and 240 assess repeatability only: a stationary target does not provide independent geometry.
- Visible cube edge midpoints are excluded from pose fitting and used for projection checks. The separate CS-200 target is excluded from continuous pose/intrinsic fitting; its three marker mounting centres define 150/200/250 mm distances. Screw/marker centre visibility must be flagged; silhouette tips are not automatically mounting centres.
- A cube is rotationally symmetric. Keep its vertical direction fixed and explicitly enumerate the four horizontal rotations per camera. CS-200 epipolar consistency may select discrete cube registration, but must not adjust camera pose or scale. Because it participates in registration, its length checks are conditional validation, not fully independent external validation. Preserve competing registrations and their errors.

## Frozen engineering checks

- At least six visible, non-coplanar cube correspondences per camera; no invented occluded corners. Proper rotation, finite parameters and positive depth.
- Cube fitting RMS <= 2 px and held-out midpoint RMS <= 2 px for each camera used in reconstruction. These image thresholds are engineering checks, not an accuracy guarantee.
- Require all three CS-200 lengths within 1% of the drawing, with positive reconstructed depth and reprojection RMS <= 2 px. A 1% scale check is a necessary screen for roughly 1 km/h changes near 100 km/h, not sufficient evidence of 1 km/h speed accuracy. Report every camera pair, do not select one for matching speed.
- Report image-coordinate perturbations of +/-1 px as deterministic engineering sensitivity, not confidence intervals. No physical-speed claim from an unverified capture clock.
- If checks fail, retain the failed round. Corrections require identifiable wrong annotations, correspondence or source geometry; do not relax thresholds. Do not calculate an accepted hand–ball 3D result until the checks pass.

## Limits

The source workbook declares 360 Hz capture and genlock, while MP4 playback is 30 Hz; file-to-speed mapping (67/70 filename versus 69.1 MPH worksheet entry), actual capture timebase, radar measurement definition and subject identity are unresolved. New Edgertronic videos are different throws from the Prime/C3D and 3034 studies. No cross-dataset before/after error claim, intervention benefit or model calibration to radar speed is allowed.

Sources: [source calibration guide](https://github.com/drivelineresearch/openbiomechanics/blob/main/computer_vision/calibration/README.md), [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html). Local assets and hashes are listed in `artifacts/edge-calibration/calibration-source-manifest.json`; the CS-200 technical drawing is in the preceding paired-source worktree.
