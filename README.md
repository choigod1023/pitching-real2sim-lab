# Pitching Real2Sim Lab

A personal, self-directed research project by **Jang JunHyeok**, a student in the School of Computer Science and AI at Dongguk University (Seoul, South Korea). It is a learning and portfolio project, not funded research and not affiliated with any team or company.

## Goal

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

- **Absolute speed is not reproduced.** The force-based model gives about 113 km/h against recorded 128 to 131 km/h.
- **Marker-only release is dominated by assumptions.** Moving the assumed wrist-to-ball distance by 5 cm changes the estimate by about 15 km/h, and the filter cutoff changes it by up to about 20 km/h. A single offset calibrated on four pitches predicts the fifth with a mean error of 2.4 km/h, which is worse than simply predicting the mean speed (1.1 km/h).
- **Speed change between pitches is not predicted.** The recorded difference between a slower and a faster pitch is +2.6 km/h; the model gives +0.16 km/h. A learned model on 89 athletes (203 held-out pitches) has MAE 0.990 km/h versus 0.964 km/h for a per-athlete mean.
- **Muscle-path consistency fails.** For wrapping paths, path-length derivatives and path velocities disagree in all 15 sampled states. The mean force-distribution residual is about 138 N·m, too large to trust muscle-level conclusions.
- **Marker quality check.** The C3D residuals and camera masks are empty, so reconstruction quality cannot be assessed. The wrist-to-finger marker distance grows by about 30 to 50 percent after release in two of five trials; real finger motion and marker movement cannot be separated without video.
- **Candidate speed gains were rejected.** Positive candidates (up to +7.5 km/h in one diagnostic) all violated the force or work constraints.
- **Video route works in principle.** On a separate public throw (OBP-CV, 62.7 mph), ball tracking from synchronized 360 fps cameras gives about 101 to 103 km/h against 100.9 km/h recorded, and is sensitive to a one-frame sync error (about 1 to 2 km/h).

## What is needed next

Direct observation of the ball and fingers at release (synchronized video, release-event definition, independent speed measurement) for the same pitches as the motion data. The public releases do not provide this for the pitching trials.

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

Tests status: 268 tests run on the core environment, 258 run cleanly (2 skipped). Ten fail there for two reasons, not because of a logic error: nine test modules import `opensim`, which is not part of `requirements.txt`, and one test reads a locally generated file under `artifacts/` (not tracked). They pass in an environment with the OpenSim 4.6 Python bindings and the locally generated artifacts.

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
| `RESEARCH_LOG.md` | Chronological research log and links to every stage |

## Contact

Jang JunHyeok, Dongguk University. GitHub: https://github.com/choigod1023
