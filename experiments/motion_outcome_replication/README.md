# 기존 영상의 동작–공 결과 반복 검증

관찰 분석 결과는 [REPORT.md](REPORT.md), 결과를 보기 전에 고정한 기준은 [PROTOCOL.md](PROTOCOL.md), 전체 집계는 [SUMMARY.json](SUMMARY.json)에 있다. 검수 UI와 기존 앱 코드는 변경하지 않는다. 실제 신체 수정, Real2Sim 동역학 개입, 훈련·의료 조언을 제공하는 모듈이 아니다.

## 로컬 입력과 재현

저장소 worktree의 부모 디렉터리에 이미 확보된 다음 자료가 있어야 한다. 원자료는 이 저장소에 포함하지 않으며 이 모듈은 자동 다운로드하지 않는다.

- `skenes-ff-motion-cohort-20261110`: `cohort.json`, `motion-review.json`, `pose.npz`, `pose-provenance.json`, `clips/*.mp4`
- `skenes-cross-game-motion-20261106/artifacts/skenes-cross-game`: `selection.private.json`, `review.private.json`, `pose.private.npz`, `pose-source.private.json`, `skenes-2024.private.csv`, 기존 `*.mp4`

NumPy·OpenCV·Pillow·imageio-ffmpeg가 있는 기존 전용 환경을 사용했다. 아래 명령은 저장소 루트 기준이다. Windows에서 사용한 Python은 부모 폴더의 `.venv/Scripts/python.exe`다. 새 설치는 수행하지 않았다.

```powershell
../.venv/Scripts/python.exe -B -m experiments.motion_outcome_replication.analyze --lab .. --out artifacts/motion-outcome-replication-001
../.venv/Scripts/python.exe -B -m experiments.motion_outcome_replication.verify artifacts/motion-outcome-replication-001
../.venv/Scripts/python.exe -B -m unittest experiments.motion_outcome_replication.test_analysis test_kinematics test_ball_release test_pitch_outcomes
../.venv/Scripts/python.exe -B -m experiments.motion_outcome_replication.render artifacts/motion-outcome-replication-001
```

렌더러의 기본 글꼴은 Windows의 `malgun.ttf`다. 다른 환경에서는 `--font`로 한국어 글꼴을 지정해야 한다. 없는 입력을 추정값으로 채우지 않는다.

## 파일과 산출물

- `analyze.py`: 입력 무결성, 실제 PTS 대응, 구간 특징, 9시점 민감도, 경기 제외 예측.
- `verify.py`: 별도 scalar 수식으로 각도·학습·예측 검산.
- `render.py`: 실제 원영상 5쌍과 관측 골격 겹침. 영상 전체 디코드 검사 포함.
- `test_analysis.py`: 가림·원형 각도·시간창·경기 분할·대체 예측·합성 영상 시계 회귀 테스트.
- `SUMMARY.json`: 원본 행·영상·개인 경로를 제외한 집계 스냅샷. `absolute_error/zero_error/mean_error`는 경기별 MAE를 동일 비중으로 평균한 값이다. `games_beating_both`는 두 기준을 동시에 이긴 평가 경기 수다.

`artifacts/motion-outcome-replication-001`은 Git에서 제외된다. 전체 투구별 결과는 `results.private.json`, 원본에 대응시킨 추론은 `mapped-tracks.private.npz`, 수치 검산은 `verification.json`, 영상 디코드 기록은 `video-receipt.json`이다. 공 기록은 통계적 예측 목표이며 관절·시점의 보정 목적에 쓰지 않는다.

이 결과는 소수의 과거 선별 영상에 대한 개발 분석이다. `followup_candidate=false`는 현재 기준을 통과하지 못했다는 뜻이고, 생체역학적 관계가 없다는 뜻은 아니다. 관측 부족으로 전부 대체 예측이 된 항목은 실제 모델 평가와 구분해야 한다.
