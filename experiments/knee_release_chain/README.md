# 앞무릎–릴리스–회전 연결 검증

[결과](REPORT.md) · [사전 절차](PROTOCOL.md) · [모델 연결 감사](MODEL_ADAPTER_AUDIT.md) · [집계](SUMMARY.json)

기존 PR162/163의 영상·원본 프레임 연결·투구 상황을 재사용한다. 새 다운로드·관절 추론·검수 UI를 추가하지 않는다. 이 모듈은 관찰 연결의 검증과 신체개입 적격성 기록이며 훈련·의료 조언이나 선수 향상 보장이 아니다.

저장소 루트에서 기존 NumPy/OpenCV/Pillow/imageio-ffmpeg 환경으로 실행했다.

```powershell
../.venv/Scripts/python.exe -B -m experiments.knee_release_chain.analyze --lab .. --out artifacts/knee-release-chain-001
../.venv/Scripts/python.exe -B -m unittest experiments.knee_release_chain.test_analysis test_kinematics test_ball_release test_pitch_outcomes
../.venv/Scripts/python.exe -B -m experiments.knee_release_chain.render --lab .. --out artifacts/knee-release-chain-001
```

필수 로컬 입력은 부모 폴더의 `matched-knee-movement-20261113/artifacts/matched-knee-movement-001/results.private.json`, PR162의 `motion-outcome-replication-20261113/artifacts/motion-outcome-replication-001/mapped-tracks.private.npz`, `skenes-cross-game-motion-20261106/artifacts/skenes-cross-game/skenes-2024.private.csv`와 기존 영상이다. 없는 원자료를 자동 다운로드하거나 임의로 채우지 않는다. 렌더러 기본 글꼴은Windows `malgun.ttf`이며 필요 시`--font`로 지정한다.

출력은 ignored `artifacts/knee-release-chain-001`의 `results.private.json`, `direction-traces.private.npz`, `knee-release-spin-original-overlay.mp4`, 대표 JPG, `video-receipt.json`이다. 실제 원본과 투구별 기록은Git에넣지않는다. `SUMMARY.json`에는 원자료행/개인경로 없이 집계와 근거해시를 보존했다.

통계는 같은 선수의 선택된 과거 영상에 대한 경기 제외 개발 분석이다. `repeat_pass=false`는 사전 조건 미통과이며 관계 부재 증명이 아니다. `real2sim.predicted_gain=null`은 신체개입 미실행으로0효과와구분한다. 손가락NaN은이탐지기의구조적미출력이다. 코드테스트·영상의매끄러움·작은수치검산차이를실제관절정확도나선수효과로해석하지않는다.
