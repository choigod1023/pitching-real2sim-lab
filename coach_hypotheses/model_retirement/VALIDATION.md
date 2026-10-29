# 고정 그립 모델 사용 중단 검증

검증일: 2026-10-29. 기준 브랜치: 원격 `main`의 `63fc06c2c37b9af55c48d341a5c5e39d8a6deb2e`. 격리 브랜치: `codex/retire-fixed-grip-predictor-20261029`. 미병합 연구 브랜치의 코드를 가져오지 않았습니다.

## 실행과 결과

아래 명령의 `python`은 `../.venv/Scripts/python.exe`입니다.

```powershell
python -m unittest coach_hypotheses.model_retirement.test_policy -v
python -m unittest discover
git diff --check
```

- 새 정책 회귀 검사: 14개 중 14개 통과. 오류·건너뜀 없음.
- 전체 검사: 289개 실행, 277개 통과, assertion 실패 0개, 환경/산출물 오류 10개, 건너뜀 2개.
- 기존 단계 0 기록: 275개 실행, 263개 통과, 오류 10개, 건너뜀 2개. 오류 식별자 집합을 Python으로 비교했으며 추가·제거된 오류가 없습니다. 전체 통과로 보고하지 않습니다.
- 실제 Tk 창을 숨긴 상태로 생성해 제목·표 머리말·행 생성·진단 JSON 저장을 확인했습니다. 기본 호출은 창 생성 전에 차단됐습니다.
- 7개 기본 CLI(앱 1개, 후보 검색 6개)와 Windows 실행기가 모두 종료 코드 2로 계산을 거부했습니다. 오래된 캐시의 승인 플래그도 예측 기능을 켜지 못했습니다.
- 명시적 진단 재현에서는 합성 입력의 가정점 속력 126km/h를 그대로 보존하면서 예측 구속·향상 필드를 `null`로 남겼습니다. 이 숫자는 소프트웨어 회귀 검사 입력이며 실제 선수의 구속 검증 결과가 아닙니다.
- `git diff --check` 통과. 물리 계산·힘 용량·일 예산·그립·이탈 시점·사전 과학적 합격 기준을 바꾸지 않았습니다.

전체 검사 상세 로그와 구조화 결과는 로컬 `artifacts/full-tests.txt`, `artifacts/test_results.json`에 있습니다. 이 폴더의 `artifacts/`는 Git에서 제외합니다. 기존 기준선의 읽기 전용 비교 자료는 `../coach-hypotheses-stage0-20261029/coach_hypotheses/stage0/stage0_results.json`입니다(프로젝트 루트에서 본 경로).

## 남은 환경 오류

OpenSim 미설치 오류는 다음 9개 테스트 모듈입니다: `test_anatomical_shoulder`, `test_applied_path_forces`, `test_directional_shoulder`, `test_elbow_moment_validation`, `test_elbow_registration_audit`, `test_muscle_equilibrium_mode`, `test_shoulder_load_audit`, `test_wrist_integration`, `test_wrist_residual_audit`.

`test_research_audit_gate.ResearchAuditGateTests.test_actual_wrapped_path_fails_consistency_gate`는 로컬 `artifacts/research-evidence-audit/formula_checks.json`이 없어 오류가 났습니다. `test_pitch_fit.ArmFitTests`의 2개 검사는 고정 버전 모델이 없어 건너뛰었습니다. 통과시키기 위한 가짜 산출물이나 대체 OpenSim 모듈을 넣지 않았습니다.

기존 `test_coupled_body_search`가 작업 폴더의 추적 파일 `artifacts/coupled-body/model.xml`을 재생성했습니다. 이 실행이 만든 사본을 로컬 로그 폴더에 보존하고, 해당 파일만 검사 전 내용으로 복구했습니다. 복구 후 SHA-256은 검사 전과 같은 `abb05e8208c80e33ca31caf37077c05a0642bb82cee3821aab540a09c6782d97`입니다. 다른 작업 폴더나 과거 실험 기록은 수정하지 않았습니다.

## 판정과 한계

사용 중단 정책의 회귀 검사에는 통과했습니다. **구속 예측의 과학적 검증 통과 또는 예측 오차 감소를 뜻하지 않습니다.** 과거 수치와 실패를 삭제하지 않았고, 대체 모델도 승인하지 않았습니다. 관절 변경에 따른 실제 선수 성능, 근육 능력, 안전성, 훈련 효과는 이번 변경의 검증 대상이 아닙니다. 훈련이나 의료 조언이 아닙니다.
