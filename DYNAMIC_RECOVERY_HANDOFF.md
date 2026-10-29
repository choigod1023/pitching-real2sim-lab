# Automatic continuation and recovery

Heartbeat `real2sim` is active every15 minutes in this chat, authorized by the user on2026-10-29. Normal progress stays quiet. The computer and desktop app must remain running for local work. No agent spawning is authorized by this request. No paid services, external messages, athlete recruitment, deployment or PR merging.

This branch started from latest origin/main, then fast-forward cherry-picked the unmerged PR97 commit50e6aa5 as an explicit prerequisite. PR97 is not merged. Build outputs are isolated under `artifacts/dynamic-recovery`. Python: adjacent `.venv/Scripts/python.exe`.

## Completed; do not recompute unchanged

- Four public dynamic calibration MP4s acquired with observed hashes, 1320 frames each. Large-file download initially returned Google's scan-warning form; following that public form with the same fileID recovered access. No access restriction was bypassed.
- Camera1 and4, frames0..552 stride12, 47frames each, three detectors executed. Source and native detections retained. Original4x arm had no full matched frame set.
- Actual frame0 scale/border probes recorded. Repaired2x/border arm reuses previous baseline/native observations. Matched38frames:30train/8heldout. Methods compared using identical frames and frozen static annotations. None passed metric checks.
- Train-only intrinsic refit retained the original camera1 zero-square / camera4 k1-square classes. Smaller mean cube discrepancy came with a worse maximum edge discrepancy. Not accepted.
- Full coarse four-camera probe completed220/220 frames. Detected by either method: camera1/2/3/4=24/14/16/20. Shared same-index detections:1-4=20,1-3=1(frame576), other pairs0. These are sampled observations, not proof that no other overlap exists.

## Next executable actions

1. **Next first command:** `python probe_dynamic_coverage.py --cameras 1 3 --start 540 --stop 612 --stride 1 --max-new 40`, resuming until its separate progress file says complete. Coarse full probe is already complete; do not repeat it. This dense window is chosen around the observed transition overlap at576, without inspecting cube length or radar speed. Keep the same count/metric acceptance thresholds and report narrow pose/time diversity. More adjacent frames alone do not prove independent calibration accuracy. The script checkpoints observations and bounds actual indices to1319.
2. Inspect representative detected and missed views, grid corner ordering, board side/blue orientation markings and exact file alignment. Use the now-grounded observation overlap to decide which camera pairs can be tested with original minimum counts (>=8shared,>=5train,>=2held). Failed or ambiguous pairs stay failed/unknown. Check zero-index correspondence first; ±1frame diagnostics must be reported in full and never selected to hit radar speed.
3. If multiview geometry remains inconsistent, distinguish wrong image centres/correspondence, target non-rigidity/dimensions and camera parameters by reprojections of **unchanged** target observations across camera pairs. Any new centre annotation requires a separate traceable revision and cannot silently replace a previous heldout target. Seek author source drawings/protocol and compare source code; do not infer a new physical cube size merely to make residuals vanish.
4. Revalidate physical lengths, coordinate/time definitions, release observations and independent same-pitch speed reference. Only then use the original gate protocol to advance towards hand–ball prediction and constrained body intervention. Partial board improvement is not a scientific pass or a demonstrated reduction in km/h error.

For actual failures, record the observed error, tested recovery, numerical result and next action. Reuse successful downloads and completed detections. Do not endlessly repeat failed identical input, hand-edit gate status or relax acceptance limits. Further code changes need a fresh isolated branch/worktree from remote default with explicit prerequisite commits. The gate worktree progress.json is the continuation ledger; preserve its existing entries.

Report a final result or an unresolved actionable obstacle only after reasonable recovery/research attempts. Keep the automation active while these specific executable tasks remain. If no new measurement/decision-free route remains, save state, pause the automation and explain the missing evidence. Past35%/2%stop rules are removed.
