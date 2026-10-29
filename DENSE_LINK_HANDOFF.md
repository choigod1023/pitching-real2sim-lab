# Next automatic continuation

This worktree starts from latest origin/main and explicitly includes unmerged prerequisite commits50e6aa5(PR97),024acfa(PR98). No PR was merged. Follow original scientific gates; data remains unpassed, no accepted hand–ball/athlete intervention.

Completed (reuse, do not repeat): dense1–3 frames540..612 (146camera-frames); native/source fixed-order fit; training-only per-frame order recovery; all4 cube correspondence phases as unaccepted diagnostics; all-1/0/+1frame shifts on matched sets; 2px planar-grid quality-conditioned comparison; masked SIFT background orientation attempts at0/120/240. All failed original metric acceptance. Details in DENSE_LINK_FINDINGS.md.

Actual outputs: artifacts/dense-camera-link/dense-link-results.json,ordered-link-results.json,cube-phase-diagnostic.json,time-shift-link-results.json,per-view-planarity-diagnostic.json,planar-quality-link-results.json,background-orientation-results.json. Video sources remain read-only in previous worktrees. Static cube/CS-200 annotations unchanged; only6cube edges are observable in1–3. Phase3 is provisional, not adopted just for its lower reprojection.

## Next actual experiment

1. Create a new isolated worktree from latest remote default and bring required predecessor commits explicitly. Read gate progress.json for the latest commit/PR, not an old automation path.
2. Reuse full-coverage per-frame observations in dynamic-camera-recovery-20261029/artifacts/dynamic-recovery/coverage. For camera1 and3, use frames **outside540..612** to inspect/refit the original per-camera intrinsic model classes (zero_square for1&3); preserve current source matrices as baseline. Freeze training/evaluation and parameter validity checks first. No cube lengths or speed choose camera model/parameters. Do not repeat source video downloads.
3. Refit camera1–3 stereo using the already frozen quality-conditioned17frames,14train/3heldout and training-only per-frame association. Compare baseline/refit on identical frames and unchanged static observations. Report all visible6edges, depths, all100mm board adjacencies, pose/coverage diversity, phase1 primary and provisional phase diagnostics. Keep <=2px/<=1% and >=8/5/2; no silent exclusions or threshold relaxation.
4. If unresolved, research source definitions and imperfect-target calibration (Strobl/Hirzinger2011 Workshop) or affine-view matching as separately frozen diagnostics. Do not tune heldout target geometry/size to pass. Annotation identity and physical clock/radar reference remain independent missing checks.

Python: ../.venv/Scripts/python.exe. Existing commands: dense_camera_link.py acquire/fit; recover_board_order.py; time_shift_link_diagnostic.py; view_grid_quality.py; planar_quality_link.py; background_orientation.py. Routine progress is quiet; only full completion/unresolved actionable blockade is notified. Current automation real2sim stays ACTIVE. No new subagents, paid purchases, external contact, athlete recruitment, deployments or PR merges. Release any owned active_execution lease when this run ends and save exact next executable step.
