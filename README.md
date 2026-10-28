# Pitching Real2Sim Lab

A personal, self-directed research project by **Jang JunHyeok**, a student in the School of Computer Science and AI at Dongguk University (Seoul, South Korea). It is a learning and portfolio project, not funded research and not affiliated with any team or company.

## Goal

Latest body-input experiment: [control retiming findings](CONTROL_RETIMING_FINDINGS.md).
27 timing conditions at two integration resolutions produced **no accepted speed gain** under the stated pose and work budgets. Pitching-scene overlay videos, including measured joints and baseline/modified simulations, follow [the video standard](PITCHING_VIDEO_STANDARD.md).

Can a small change in an athlete's joint motion raise pitch speed **without raising the athlete's force capacity**? To answer this I try to turn measured pitching motion into a physics simulation (Real2Sim), predict the speed of the released ball, and test whether the prediction survives held-out checks.

The project is deliberately skeptical. Every step has a stated pass criterion, and results that fail it are kept and reported. **No result here shows that any change improves a real athlete's speed, accuracy, or safety, and nothing here is training or medical advice.**

## Pipeline

| Stage | Input | Processing | Output |
|---|---|---|---|
| 1. Data alignment | OpenBiomechanics C3D markers (360 Hz), joint signals, force plates (1080 Hz), metadata | Unit, time and coordinate-frame checks | Verified per-pitch tables |
| 2. Joint reconstruction | Marker positions | Fixed segment lengths, skin-marker frames | Joint trajectories |
| 3. Inverse dynamics and muscle load | Trajectories, body mass | OpenSim arm model (47 muscles), generalized-force residuals | Torque demand, residuals |
| 4. Forward integration | Force inputs | RK4 integration of arm, hand and attached ball | Hand and ball states |
| 5. Hand-to-ball release | Hand frame, assumed grip offset | Rigid-offset release and flight model | Release speed and direction |
| 6. Candidate search | Small joint or timing changes | Force, work and load constraints | Speed change per candidate |
| 7. Validation | Held-out pitches | Comparison against a no-change baseline | Pass or fail |

## What I found so far

These are development results on already-seen data (one athlete, session 3034, five fastballs), not independent validation.

- **Absolute speed depends on one distance.** With an assumed 15 cm wrist-to-ball distance the force-based model gives about 113 km/h against recorded 128 to 131 km/h. With a distance measured from video (about 19 to 21 cm, see `video_track/`) the five pitches come within -3.2 to +2.4 km/h on average (mean bias -1.0 km/h at 20 cm), but one pitch (3034_5) stays 5 to 7 km/h low. The distance comes from one other athlete's throw and was not fitted to the radar speeds.
- **Marker-only release is dominated by assumptions.** Moving the assumed wrist-to-ball distance by 5 cm changes the estimate by about 15 km/h, and the filter cutoff changes it by up to about 20 km/h. A single offset calibrated on four pitches predicts the fifth with a mean error of 2.4 km/h, which is worse than simply predicting the mean speed (1.1 km/h).
- **Speed change between pitches is not predicted.** The recorded difference between a slower and a faster pitch is +2.6 km/h; the model gives +0.16 km/h. A learned model on 89 athletes (203 held-out pitches) has MAE 0.990 km/h versus 0.964 km/h for a per-athlete mean.
- **Muscle-path consistency fails.** For wrapping paths, path-length derivatives and path velocities disagree in all 15 sampled states. The mean force-distribution residual is about 138 N·m, too large to trust muscle-level conclusions.
- **Marker quality check.** The C3D residuals and camera masks are empty, so reconstruction quality cannot be assessed. The wrist-to-finger marker distance grows by about 30 to 50 percent after release in two of five trials; real finger motion and marker movement cannot be separated without video.
- **Candidate speed gains were rejected.** Positive candidates (up to +7.5 km/h in one diagnostic) all violated the force or work constraints.
- **Video tracking of the ball.** On a separate public throw (OBP-CV Prime cameras, 62.7 mph), eight synchronized 360 fps cameras give a ball speed of 97 to 102 km/h (99.1 km/h linear fit) against 100.9 km/h recorded, with a median reprojection error of 1.5 px. The ball-to-wrist distance grows from about 14 cm to 19 to 21 cm over the 20 frames before release.
- **The release frame is the main uncertainty.** Moving the video release frame by one frame (2.8 ms) changes the wrist-to-ball distance by 2 to 3 cm and the estimated speed of the 3034 pitches by 6 to 10 km/h.
- **Differences between pitches are still not explained.** After removing the mean bias the simulation's error is 1.5 to 1.7 km/h, worse than predicting the mean recorded speed (0.89 km/h). A change of about 1 km/h cannot be resolved with this method.

## What is needed next

Direct observation of the ball and fingers at release (synchronized video, release-event definition, independent speed measurement) for the same pitches as the motion data. The public releases do not provide this for the pitching trials, so the wrist-to-ball distance of the 3034 athlete cannot be checked directly.

## Data and licensing

- This project uses the [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics) from Driveline Baseball R&D. **Raw C3D files and videos are not included here.** Download them from the OBP repository and respect its terms.
- OBP data and derived data are licensed **CC BY-NC-SA 4.0**. Anyone who is an employee, contractor, or significant shareholder of a professional sports organization or financial analysis firm may not use that data. See the OBP license.
- Code in this repository is licensed under MIT (see `LICENSE`). Derived data files are under CC BY-NC-SA 4.0 (see `NOTICE.md`).
- Please cite OpenBiomechanics and Driveline Baseball Research and Development when you use anything derived from the data.

## Setup

Python 3.12. Core dependencies are in `requirements.txt` (NumPy, SciPy, pandas, ezc3d, MuJoCo, Pillow). The musculoskeletal modules additionally need the OpenSim 4.6 Python bindings.

```bash
python -m venv .venv
pip install -r requirements.txt
python -m unittest discover
```

Tests status: 275 tests run on the core environment, 263 pass (2 skipped). Ten fail there for two reasons, not because of a logic error: nine test modules import `opensim`, which is not part of `requirements.txt`, and one test reads a locally generated file under `artifacts/` (not tracked). They pass in an environment with the OpenSim 4.6 Python bindings and the locally generated artifacts.

To rerun the marker audit in `marker_audit/`, download the OBP pitching C3D files, then run for example `python offset_cv.py <path to baseball_pitching/data/c3d/000002>` from inside that folder. The scripts write their JSON and figures next to themselves.

## Key modules

| File | Purpose |
|---|---|
| `kinematics.py` | Marker and joint kinematics from C3D |
| `ball_release.py` | Hand frame and rigid-offset release diagnostic |
| `trace_error_audit.py` | Separates observation, kinematics, input and integration error |
| `applied_path_forces.py` | Muscle forces applied through the actual path |
| `anatomical_muscle_screen.py` | Muscle-level force screening |
| `shoulder_load_audit.py`, `wrist_residual_audit.py` | Load-residual audits |
| `research_evidence_audit.py` | Data, unit and consistency audit |
| `pitch_outcomes.py` | Ball flight (gravity, drag, constant lift) |
| `marker_audit/` | Marker quality checks, finger-marker gating, and hand-offset cross-validation (see its findings files) |
| `video_track/` | Eight-camera ball tracking, hand-to-ball distance, error budget and five-pitch simulation (see `video_track/README.md`) |
| `RESEARCH_LOG.md` | Chronological research log and links to every stage |

## Contact

Jang JunHyeok, Dongguk University. GitHub: https://github.com/choigod1023
