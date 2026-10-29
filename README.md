# Pitching Real2Sim Lab

고정 15cm 손목–공 모델은 **주 구속 예측·자세 추천 경로에서 제외**했습니다. 기본 GUI 실행기와 이 모델을 사용하는 후보 검색 진입점은 계산을 시작하지 않습니다. 과거 실험과 원본 자료는 삭제하지 않았으며, 명시적인 `--legacy-diagnostic` 옵션으로만 해당 진입점의 진단을 재현할 수 있습니다. 진단 출력은 실제 공 구속이나 선수의 향상 예측이 아닙니다. 대체 예측 모델은 아직 채택하지 않았습니다. 범위와 실행 방법은 [모델 사용 중단 기록](coach_hypotheses/model_retirement/README.md)을 참고하세요.

이후 [설계 확정과 실제 자료 검증](coach_hypotheses/research_reset/README.md)을 진행했습니다. 98선수/399투구의 관측 투구쌍 진단도 0변화 기준을 넘지 못해 추천 모델로 채택하지 않았습니다. 회고적 평균 제거 오차와 투구쌍 차이 오차는 서로 다른 평가값입니다. 다음 작업은 [전신 관측/손–공 연결의 병렬 검증 계획](coach_hypotheses/research_reset/NEXT_STEPS.md)에 있습니다.

A personal, self-directed research project by **Jang JunHyeok**, a student in the School of Computer Science and AI at Dongguk University (Seoul, South Korea). It is a learning and portfolio project, not funded research and not affiliated with any team or company.

## Goal

Can a small change in an athlete's joint motion raise pitch speed **without raising the athlete's force capacity**? To answer this I try to turn measured pitching motion into a physics simulation (Real2Sim), predict the speed of the released ball, and test whether the prediction survives held-out checks.

The project is deliberately skeptical. Every step has a stated pass criterion, and results that fail it are kept and reported. **No result here shows that any change improves a real athlete's speed, accuracy, or safety, and nothing here is training or medical advice.**

## Pipeline

아래는 기존 연구의 구성입니다. 5·6단계의 고정 그립 방출 및 후보 검색은 현재 예측·추천 기능으로 제공하지 않으며, 과거 진단 경로로만 보존합니다. 다른 단계도 해당 모델의 구속 예측 유효성을 입증한 것은 아닙니다.

| Stage | Input | Processing | Output |
|---|---|---|---|
| 1. Data alignment | OpenBiomechanics C3D markers (360 Hz), joint signals, force plates (1080 Hz), metadata | Unit, time and coordinate-frame checks | Verified per-pitch tables |
| 2. Joint reconstruction | Marker positions | Fixed segment lengths, skin-marker frames | Joint trajectories |
| 3. Inverse dynamics and muscle load | Trajectories, body mass | OpenSim arm model (47 muscles), generalized-force residuals | Torque demand, residuals |
| 4. Forward integration | Force inputs | RK4 integration of arm, hand and attached ball | Hand and ball states |
| 5. Historical release diagnostic (retired) | Hand frame, assumed grip offset | Rigid-offset release and conditional flight | Assumed-point diagnostic; unavailable as speed prediction |
| 6. Historical candidate search (retired) | Small joint or timing changes | Force, work and load diagnostics | Archived conditional comparisons; unavailable as recommendations |
| 7. Validation | Held-out pitches | Comparison against a no-change baseline | Pass or fail |

## What I found so far

These are development results on already-seen data (one athlete, session 3034, five fastballs), not independent validation.

이하 수치는 사용 중단 전의 연구 기록입니다. 수치를 보존하되, 거리 가정 변경이나 다른 선수의 관측으로 얻은 근접 결과를 검증된 예측 모델의 성능으로 해석하지 않습니다.

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

사용 중단 변경 후 코어 환경에서 289개를 실행해 277개 통과, 10개 오류, 2개 건너뜀을 확인했습니다. 새 회귀 검사 14개는 모두 통과했습니다. 오류 10개는 기존 기준선과 같은 항목으로, OpenSim 바인딩 부재 9개와 로컬 `artifacts/` 파일 부재 1개입니다. 전체 통과를 뜻하지 않으며, 해당 의존성을 갖춘 환경에서의 재검증은 별도로 필요합니다. 자세한 명령과 판정은 [검증 기록](coach_hypotheses/model_retirement/VALIDATION.md)에 있습니다.

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
