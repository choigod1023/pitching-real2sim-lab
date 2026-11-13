# 손가락 관측과 같은 투구 모델 연결

[보고서](REPORT.md) · [사전 절차](PROTOCOL.md) · [집계](SUMMARY.json)

기존 영상에21점 손 모델을 실행하고, Prime의 같은 투구3D를 제외 시야로 검사한다. 관측 실패를 채워서 물리 모델을 구동하지 않는다. 이번 실행은3D기준 미통과이며 동역학·구속/RPM/무브먼트 예측은 미실행이다. 훈련·의료 조언이나 선수 향상 보장이 아니다.

## 재현

현재 worktree 루트에서 실행한다. `<LAB>`은 기존 자료 worktree들의 부모이고 `<OUT>`은ignored `artifacts/finger-bridge-001`이다. 공유 Python에는 NumPy/OpenCV/Pillow/SciPy/OpenPyXL/MuJoCo가 있고, 손 추론에는 기존 `pose-measurement-pilot-20261103/experiments/pose_measurement/.local/venv`의 MediaPipe Python을 사용한다. 원자료를 자동 다운로드하거나 설치하지 않는다.

```text
python -B -m experiments.observed_finger_bridge.infer prepare --lab <LAB> --out <OUT>
<MEDIAPIPE_PYTHON> -B -m experiments.observed_finger_bridge.infer run --manifest <OUT>/input_manifest.private.json --out <OUT>
python -B -m experiments.observed_finger_bridge.evidence --lab <LAB> --output <OUT>/evidence.private.json
python -B -m experiments.observed_finger_bridge.evaluate --out <OUT> --manual-source <LAB>/non-skenes-paired-source-audit-20261110/branson_projection.py
python -B -m experiments.observed_finger_bridge.bridge --lab <LAB> --input <OUT>/inference.private.json --method raw --output <OUT>/bridge-raw.json
python -B -m experiments.observed_finger_bridge.bridge --lab <LAB> --input <OUT>/inference.private.json --method rotated --output <OUT>/bridge-rotated.json
python -B -m experiments.observed_finger_bridge.render --out <OUT>
python -B -m unittest experiments.observed_finger_bridge.test_infer experiments.observed_finger_bridge.test_bridge experiments.observed_finger_bridge.test_evidence experiments.observed_finger_bridge.test_evaluate experiments.observed_finger_bridge.test_render test_kinematics test_ball_release test_pitch_outcomes -q
```

평가에는 기존 `non-skenes-paired-source-audit-20261110/branson_projection.py`의 `samples` 리터럴을 실행 없이 AST로 읽어 동결한 `manual-existing.private.json`이 필요하다. 신규F8/10/12의 가시성 검토는 `manual-new-exclusions.private.json`에 있다. 12가시 손끝 영역의 정체는 미확정이므로 최적배정 거리는 낙관적인 하한이다. 원방향·정규화의 공통9영역과 누락률을 함께 비교한다.

출력 `inference.private.json`은 모든 후보와 제외 사유를 남긴다. `bridge-raw/rotated.json`과 `.private.npz`는 실패한 재구성과 접촉 입력을 분리한다. `evidence.private.json`은 원본 행·영상 링크·해시와 null을 보존한다. `observed-finger-original-overlay.mp4`는 실제 원본 비교 영상이며 신체 개입 시뮬레이션이 아니다.

렌더링 안내는 Damir Sokolovsky (@heydamir)의 unslop, CC BY4.0을 참고했다. 실제 연구 지표·원본 자료는 스타일 지침으로 수정하지 않았다.
