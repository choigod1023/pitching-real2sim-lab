# 시야·상황을 맞춘 앞무릎 조건부 분석

[결과와 의미](REPORT.md), [사전 기준](PROTOCOL.md), [추가 진단 절차](DIAGNOSTIC_PROTOCOL.md), [전체 집계](SUMMARY.json).

기존 PR162의 로컬 결과와 원영상을 읽는다. 새 검수 기능·관절 추론·다운로드를 추가하지 않는다. 선수에 대한 훈련·의료 조언이나 인과적 Real2Sim 개입 모듈이 아니다. 코드와 집계만 Git에 저장한다.

저장소 루트에서 기존 NumPy/OpenCV/Pillow/imageio-ffmpeg 전용 환경으로 실행했다. 입력은 부모 `pitching-lab` 폴더에 있어야 하며 없는 파일을 자동으로 만들거나 다운로드하지 않는다.

```powershell
../.venv/Scripts/python.exe -B -m unittest experiments.matched_knee_movement.test_analysis test_kinematics test_ball_release test_pitch_outcomes
../.venv/Scripts/python.exe -B -m experiments.matched_knee_movement.analyze --lab .. --out artifacts/matched-knee-movement-001
../.venv/Scripts/python.exe -B -m experiments.matched_knee_movement.diagnose artifacts/matched-knee-movement-001
../.venv/Scripts/python.exe -B -m experiments.matched_knee_movement.render --lab .. --out artifacts/matched-knee-movement-001
```

입력:

- `motion-outcome-replication-20261113/artifacts/motion-outcome-replication-001/results.private.json`과 `mapped-tracks.private.npz` — 원본 영상 경로·해시·원본 프레임 대응·고정 동작 특징.
- `skenes-cross-game-motion-20261106/artifacts/skenes-cross-game`의 `selection.private.json`, `skenes-2024.private.csv`와 기존 영상.
- `skenes-ff-motion-cohort-20261110/cohort.json` 및 기존 영상.

`analyze.py`는 원본과 같은 투구 기록을 검사하고 배경 시야·상황·조건부 예측을 계산한다. `diagnose.py`는 사전 추가 절차대로 공변량 수와 시야 군집을 조사한다. `render.py`는 순차 원본 프레임에 관측된 관절과 실제 기록/평가 예측을 겹친다. 기본 글꼴은 Windows `malgun.ttf`이며 다른 환경은 `--font`로 지정한다.

ignored 출력에는 `results.private.json`, `diagnostics.private.json`, `background-sheet-*.jpg`, `matched-knee-original-overlay.mp4`, `video-receipt.json`, `independent-review.json`이 있다. 이 모듈을 실행해도 PR162의 이전 결과·기존 앱·검수 사이트는 수정하지 않는다.

`SUMMARY.json`은 원본 행/영상 없이 보존한 집계 스냅샷이다. MAE는 각 경기 오차의 동일 비중 평균이다. 카메라 정합과 검산 수치는 관절의 실제 오차나 선수 향상으로 해석하지 않는다. 주 분석에서 충분한 경기 수가 없거나 값이 `null`이면 미계산이며 0오차가 아니다. 같은 데이터 재분석이므로 미사용 선수·독립 전향 검증이라고 부르지 않는다.
