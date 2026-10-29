# 고정 그립 모델의 예측·추천 사용 중단

고정 15cm 손목–공 모델은 실제 투구 간 구속 차이를 검증하지 못했습니다. 사용자의 제거 요청에 따라 이 모델을 주 구속 예측·자세 추천 경로에서 제외했습니다. 이는 구속 오차를 줄인 물리 모델 수정이 아니라, 검증되지 않은 출력을 예측이나 추천으로 제공하지 않도록 하는 변경입니다. 대체 모델은 아직 채택하지 않았습니다.

## 바꾼 범위

- `start_outcome_simulator.cmd`는 사용 중단 이유를 표시하고 종료 코드 2로 끝납니다. 모델 생성이나 GUI 실행을 하지 않습니다.
- `pitch_outcomes.py`의 CLI와 `build_states`, `pitch_outcomes_gui.launch`는 명시적인 진단 옵션이 없으면 계산·캐시 읽기·창 생성을 시작하지 않습니다. 오래된 `states.json`이나 캐시의 승인 표시로 이 정책을 우회할 수 없습니다.
- `joint_intervention.py`, `multi_joint_timing.py`, `constrained_motion_search.py`, `observed_candidate_simulation.py`, `coupled_body_search.py`, `empirical_budget_search.py`의 후보 검색 `run` 및 CLI도 같은 옵션을 요구합니다.
- 진단 GUI와 CLI 출력에서는 계산값을 `diagnostic_assumed_point_speed_kmh`와 `diagnostic_speed_difference_kmh`로 구분합니다. 예측 구속과 예측 향상은 `null`, 예측 가능 여부와 추천 가능 여부는 `false`입니다. GUI 제목·설명·표 머리말에도 진단 용도를 표시합니다.
- 과거 후보 검색의 수치 필드와 계산 방식은 재현을 위해 보존하되, 보고서에 `model_status: retired`, `diagnostic_only: true`, `prediction_available: false`, `coaching_recommendation_ready: false`를 기록합니다. 과거 계산값을 실제 향상 후보로 승인하는 기능은 제공하지 않습니다.

기존 결과 파일·실패 기록·원본 C3D·영상은 변경하거나 삭제하지 않았습니다. 범용 공 비행 함수 `flight`와 조건부 계산 함수 `evaluate`, 관찰 분석, 모델 생성 보조 함수, 과거 렌더링 코드는 연구 진단을 위해 남깁니다. 이 보조 함수를 직접 호출할 수 있다는 사실이 예측 모델의 채택을 뜻하지 않습니다. 다른 접촉 모델이나 관측 기반 분석까지 일괄 삭제하지 않습니다.

## 실행 방법

기존 기본 명령은 계산하지 않고 사용 중단 오류로 종료합니다.

```powershell
python pitch_outcomes.py --states artifacts/outcome-simulator/states.json
python constrained_motion_search.py
```

과거 진단이 필요한 경우에만 별도의 출력 폴더를 지정합니다. 기존 결과가 있는 폴더를 출력 경로로 지정하지 마세요.

```powershell
python pitch_outcomes.py --legacy-diagnostic --states artifacts/outcome-simulator/states.json --output artifacts/legacy-diagnostic/replay
python pitch_outcomes.py --legacy-diagnostic --states artifacts/outcome-simulator/states.json --output artifacts/legacy-diagnostic/gui --gui
```

새 앱 진단 결과는 `legacy_diagnostic_results.json`, 새 상태는 `legacy_diagnostic_states.json`으로 저장합니다. Python API에서 과거 진단을 호출하려면 `legacy_diagnostic=True`를 명시해야 합니다. 이 옵션은 모델을 예측·추천용으로 복구하지 않습니다.

## 검증 범위

```powershell
python -m unittest coach_hypotheses.model_retirement.test_policy -v
python -m unittest discover
```

기본 실행 차단, 캐시 우회 차단, 명시적 진단의 값 보존과 필드 분리, 여섯 검색 API/CLI, Windows 실행기, 실제 Tk 창의 진단 표기 및 JSON 출력을 검사합니다. 합성 입력은 소프트웨어 정책의 회귀 검사에만 사용하며 선수 데이터의 과학적 검증을 대신하지 않습니다. 전체 검사 결과와 기존 환경 오류의 비교는 이 폴더의 `VALIDATION.md`에 기록합니다.

이번 변경으로 실제 선수의 구속·구위·제구 향상, 훈련 효과, 안전성 또는 편안함을 입증하지 않았습니다. 훈련이나 의료 조언이 아닙니다. 새 추천 기능은 같은 투구의 신체·손·공 관측과 미사용 투구 검증을 확보한 뒤 별도로 판단해야 합니다.

기존 데이터 이용 조건과 [NOTICE.md](../../NOTICE.md)를 그대로 따릅니다. OpenBiomechanics/Driveline Baseball R&D 자료 및 파생 자료는 CC BY-NC-SA 4.0이며, 이번 변경에 원본 C3D나 영상을 추가하지 않습니다.
