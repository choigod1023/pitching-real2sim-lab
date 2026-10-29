# 단계0: 기준선 확인

## 범위와 판정

최신 원격 기본 브랜치 `origin/main`의 `63fc06c2c37b9af55c48d341a5c5e39d8a6deb2e`를
기준으로 전용 worktree와 `codex/coach-hypotheses-stage0-20261029` 브랜치를 만들었습니다.
미병합 연구 브랜치의 코드나 산출물을 가져오지 않았습니다.

**단계0의 문서 검토·기존 테스트 실행·기준선 기록을 완료했습니다.**
전체 테스트가 통과한 상태는 아닙니다. README의 core 환경 기록을 그대로 재현했습니다.
단계1~3은 수행하지 않았으며 자동 루틴도 PAUSED로 유지했습니다.

이번 단계에 별도의 과학적 성능 합격 기준은 지정되지 않았습니다. README의 예상
집계와 실제 집계가 일치하는지를 대조했으며, 이것을 연구 가설의 합격으로 바꾸지 않았습니다.
단계1의 합격 기준 PROTOCOL 작성·사전 커밋도 아직 하지 않았습니다.

## 읽은 기존 문서와 확인 내용

1. `README.md`(80줄): 목표와 한계, 환경/테스트 기록, 라이선스·인용 조건을 읽었습니다.
   선수1명/세션3034/5구의 개발 결과이며 독립 검증이 아닙니다. 편향 제거 후
   구속 오차1.5~1.7km/h가 평균 예측0.89km/h보다 나쁘다는 기록을 확인했습니다.
   89명/203개 검사 투구의 학습 모델0.990km/h 대 선수 평균0.964km/h도 문서 기록입니다.
   이번 단계에서 이 연구 수치를 다시 계산하지는 않았습니다.
2. `RESEARCH_LOG.md`(88줄): 이전 재생 모델과 연구 단계별 문서 링크, 물리 예측과
   `mj_forward` 좌표 재생의 구분, 결과 해석의 한계를 읽었습니다.
3. `JOINT_INTERVENTION_PROTOCOL.md`(17줄): 원래3034_1 기준 모델, 마지막30ms,
   팔꿈치 dof6와 손목 dof8/9/10의 signed1°/2° 및 무변경을 합친17조건입니다.
   손목 축은 해부학적으로 보정된 굴신/편위 축이 아닙니다. 수정 궤적마다 필요한
   역동역학 힘을 다시 계산하므로, 고정 입력이나 검증된 동일 근력 실험과 다릅니다.
   골반·몸통 시퀀싱은 이 프로토콜의 개입 변수가 아닙니다.
4. `FORCE_POSE_FINDINGS.md`(56줄): 발 접촉에 힘을 분배하는 모델이 아니며,
   보조 외력을 포함한 궤적 재현과 실측 외력만 쓴 재현을 구분한다는 한계를 확인했습니다.

별도 전신 실험 `coupled_body_search.py`도 저장소에 존재합니다. 따라서 위 팔 개입
프로토콜의 범위를 저장소 전체의 유일한 기능으로 일반화하지 않습니다.
후속 확장 전에는 이 별도 실험의 재사용 가능성과 검증 상태를 확인해야 합니다.
이번에는 해당 모델을 새로 확장하거나 성능 실험을 추가하지 않았습니다.

각 문서와 원래 추적 파일의 SHA256은 `tracked_before.sha256.json`, 주요 문서의
보존 확인은 `preservation_check.json`에 기록했습니다. 기존 문서/결과는 수정하지 않았습니다.

## 실행 환경과 명령

- Python3.12.13. 실행기: 상위 `pitching-lab/.venv/Scripts/python.exe`.
- NumPy2.5.4, SciPy1.18.1, pandas3.1.0, MuJoCo3.17.0, ezc3d1.7.2, Pillow13.0.0.
- 이 Python 환경에서는 `importlib.util.find_spec('opensim')`이 `None`입니다.
  PC 전체에 OpenSim이 없다고 판단한 것은 아닙니다.
- 패키지 설치, 환경 변경, fixture 복사 또는 오류/테스트의 임의 스킵은 하지 않았습니다.

저장소 루트에서 실행한 전체 테스트 명령:

```powershell
& ../.venv/Scripts/python.exe -m unittest discover -v
```

실제 절대 실행기/작업 폴더/시간/종료 코드는 `environment.json`, `run_started.json`,
`run_finished.json`에 있습니다. stdout/stderr를 분리해 원본 로그를 보존했습니다.
사전 Git 확인에는 `git status --short`, `git branch --show-current`,
`git worktree list --porcelain`, `git symbolic-ref refs/remotes/origin/HEAD`,
`git fetch origin`을 사용했습니다. 실행 전후586개 추적 파일의 해시를 비교했습니다.

## 실제 테스트 집계

- 실행 집계275개.
- 성공263개.
- assertion 실패0개.
- 오류10개.
- 스킵2개.
- unittest 본체 실행3.289초, 프로세스 전체5.936447초, 테스트 명령 종료 코드1.

README의275/263/10/2와 일치합니다. `errors`와 assertion `failures`를 구분했습니다.
OpenSim 모듈의 import 오류는 unittest가 `_FailedTest`로 집계한 것이므로,
그 안의 실제 검사 메서드들이 실행됐다는 뜻이 아닙니다.

### OpenSim import 오류9건

- `test_anatomical_shoulder`
- `test_applied_path_forces`
- `test_directional_shoulder`
- `test_elbow_moment_validation`
- `test_elbow_registration_audit`
- `test_muscle_equilibrium_mode`
- `test_shoulder_load_audit`
- `test_wrist_integration`
- `test_wrist_residual_audit`

모두 `ModuleNotFoundError: No module named 'opensim'`입니다.
OpenSim4.6 바인딩이 있는 환경에서 이 검사들이 통과하는지는 이번에 재검증하지 않았습니다.

### 로컬 산출물 누락 오류1건

`test_research_audit_gate.ResearchAuditGateTests.test_actual_wrapped_path_fails_consistency_gate`가
`artifacts/research-evidence-audit/formula_checks.json`을 찾지 못했습니다.
검사 코드는 뒤이어 `wrapping_control.json`도 읽지만 첫 파일에서 중단됐으므로
두 번째 파일을 읽는 단계의 정상 동작은 확인하지 못했습니다.
누락 파일을 임의로 만들어 통과시키지 않았습니다.

### 스킵2건

`test_pitch_fit.ArmFitTests`의 `test_synthetic_recovery_and_constraints`,
`test_temporal_solver_preserves_constant_feasible_motion`입니다.
`artifacts/vendor/myo_sim`의 고정 버전 모델이 없다는 기존 skip 조건에 따른 결과입니다.

## 테스트의 파일 쓰기 부작용과 보존

기존 `test_coupled_body_search.TestCoupledBody.setUpClass`가
`coupled_body_search.build()`를 호출하면서 추적 중인
`artifacts/coupled-body/model.xml`을 기본 출력 위치에 다시 썼습니다.
이 부작용은 즉시 사용자에게 알렸습니다.

실행 직전의 전용 체크아웃이 깨끗했으며 이 변경은 이번 테스트 실행에서 발생했습니다.
생성본을 `generated_by_existing_tests/model.xml`에 로컬 보관한 뒤, 해당 파일 하나만
`git restore --source=HEAD --worktree -- artifacts/coupled-body/model.xml`로 복원했습니다.
**복원 후 기존 추적 파일586개 전체가 실행 전과 바이트 단위로 일치합니다.**
생성본은 새 `.gitignore`로 Git에서 제외했고 해시/복원 기록만 남겼습니다.
기존 테스트나 모델 생성 코드는 이번 단계에서 수정하지 않았습니다.

## 산출물과 미확인 사항

추가 파일은 `coach_hypotheses/` 아래의 문서·환경/집계/보존 JSON과 테스트 로그뿐입니다.
새 실행 함수가 없어 새 함수 테스트는 추가하지 않았고 기존 전체 테스트를 실행했습니다.
원본 C3D/영상·새 시뮬레이션 결과는 추가하거나 업로드하지 않았습니다.
상위 README의 훈련·의료 조언이 아니라는 문구와 NOTICE/라이선스는 유지했습니다.

확인하지 못한 것은 다음과 같습니다.

- OpenSim4.6 및 로컬 산출물이 갖춰진 전체 환경에서의 정상 통과 여부.
- README의 구속 오차/불확실성 숫자의 재산출과 실제 선수에 대한 효과.
- 제시된 코칭 가설 논문의 원문과 수치. 원문 확인은 아직 필요합니다.
- 확률분포를 정할 충분한 측정 근거 및 선수 간/선수 내 관계의 일반화.

환경/입력 부재로 오류를 분류했다고 해당 모듈의 로직이 정상이라고 입증한 것은 아닙니다.
새 회귀 오류는 확인되지 않았지만 모든 기존 테스트가 통과했다고 보고하지 않습니다.
단계1의 구현·사전등록, 단계2의 몬테카를로, 단계3의 확장 설계는 시작하지 않았습니다.

추가 파일 목록:

- `coach_hypotheses/.gitignore`: 테스트가 만든 XML의 로컬 보존본 제외.
- `coach_hypotheses/README.md`: 단계0 링크와 해석 한계.
- `coach_hypotheses/stage0/STAGE0_BASELINE.md`: 이 보고서.
- `coach_hypotheses/stage0/environment.json`: 실행기·의존성·기준 커밋.
- `coach_hypotheses/stage0/run_started.json`, `run_finished.json`: 실행 명령·시각·종료 코드.
- `coach_hypotheses/stage0/stage0_results.json`: 테스트 집계·오류/스킵 상세.
- `coach_hypotheses/stage0/test_side_effect.json`: XML 쓰기와 복원 해시.
- `coach_hypotheses/stage0/tracked_before.sha256.json`, `preservation_check.json`: 원본 보존 근거.
- `coach_hypotheses/stage0/unittest.stdout.txt`, `unittest.stderr.txt`: 가공하지 않은 출력.
