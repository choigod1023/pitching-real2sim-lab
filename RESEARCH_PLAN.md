# 구속 향상 예측 연구: 실행 계획과 인계

## 최신: 힘·일 및 파라미터 민감도 완료

작업 위치 `outputs/pitching-lab/dynamics-sensitivity`. `dynamics_sensitivity.py`에서 두 투구 각각 질량·반경 7조건을 실행했다. 아래 이전 다음 단계 1은 기초 스트레스 검사까지 완료했다. 몸통을 대신하는 외부 어깨 힘/토크가 약 120~128J의 순 일을 공급했고, 일반 질량을 ±20% 바꾸면서 원래 힘을 유지하면 손목 RMSE가 72~119mm로 증가했다. ±20%는 실측 생체 범위가 아닌 공학적 스트레스 조건이다.

다음은 손·공 분절과 이탈 조건을 식별 가능한 값/가정/불확실성으로 구분하여 추가하는 것이다. 이때 어깨 공급 일과 구동력 한계를 후보별로 늘려서 구속 개선을 만들지 않는다. 현재 작업을 계승할 때는 최신 격리 브랜치에서 테스트 및 결과 기록을 읽는다. 구속 개선 gate는 아직 닫혀 있다.

재실행: `../.venv/Scripts/python.exe torque_arm.py --pitch 3034_1 --output artifacts/torque-arm/3034_1`, 이어서 `../.venv/Scripts/python.exe dynamics_sensitivity.py --source artifacts/torque-arm/3034_1 --out artifacts/dynamics-sensitivity/3034_1`. 3034_2도 같은 방식이다. 재사용한 기존 rollout 입력은 결과 JSON에 절대경로 및 SHA256으로 기록된다.

## 2026-10-25 후속 단계 갱신

최신 작업 위치는 `outputs/pitching-lab/model-validation`이다. 아래 과거 기록보다 이 갱신을 우선한다. `centered_release.py`로 선수 내 변화 학습을 실행했고, `torque_arm.py`로 관찰 투구를 역동역학 힘으로 실제 적분하는 2분절 팔 기준 모델을 만들었다. 35개 테스트 통과. 2개 투구에서 세 시간 간격의 수렴을 확인했다. 이는 근육 실행 가능성 또는 공 구속 개선의 검증이 아니다.

다음 실행 순서:
1. 현재 팔 모델의 질량·반경 가정과 어깨 외부 힘/일의 민감도를 정량화한다. 몸통이 공급하는 일을 숨기거나 후보마다 늘려서 구속 개선을 만들지 않는다. 현재 모델에는 손·공 질량이 없다.
2. 손·공의 분절 및 결합/이탈 모델을 출처와 함께 추가한다. 공 속도 관측값만으로 그립 위치를 식별했다고 주장하지 않는다. 관찰값으로 식별할 수 없는 파라미터는 범위로 남긴다.
3. 동일 선수 구속 변화에서 무변화 기준을 넘는지, 학습 단계 안에서 전처리·파라미터를 정한 평가로 확인한다. 이미 결과를 본 코호트는 개발 자료로 표시한다.
4. 이 조건들을 통과한 뒤에만 힘 능력과 외부 일의 제약 아래 작은 수정 후보를 탐색한다. 선수의 편안함·학습 가능성은 변경 각도만으로 판정하지 않는다.

재실행: `../.venv/Scripts/python.exe torque_arm.py --pitch 3034_1`, `--pitch 3034_2 --output artifacts/torque-arm-second`. 변화 예측은 `release_calibration.py`로 특성을 생성한 뒤 `centered_release.py`를 실행하거나 `--features`로 기존 특성 파일을 지정한다. `../.venv/Scripts/python.exe -m unittest discover -v`로 검사한다.

최신 진행: [EXPERIMENT_FINDINGS.md](EXPERIMENT_FINDINGS.md). 현재 전용 작업 폴더는 `outputs/pitching-lab/pitch-fitting`이다. 관절 제약 fitting, 순간 근육 힘 검사, 전체 구간 fitting 및 관성 민감도, 손 회전 기반 공 위치 보정, 동일 선수 내 구속 차이 비교까지 연속 실행했다. 동작 적합성과 힘 요구 문제 및 단순 기준 방법보다 못한 변화량 예측 때문에 구속 개선 최적화 gate는 아직 통과하지 않았다. 아래 단계 목록은 최초 계획이며 완료 표시를 과대해석하지 않는다.

목표: 공개 실측 투구로 보정한 물리 모델에서, 힘 능력을 늘리지 않고 작은 동작 수정으로 구속 향상 후보를 예측한다. 후보가 물리 제약과 모델 불확실성을 견디는지 확인한다. 선수 훈련 효과·편안함·부상 예방을 입증했다고 주장하지 않는다.

## 단계와 진행 기준

1. **관측 기준 방법 — 실행 완료.** 기존 손 대리점 속도와 실제 공 속도를 구분한다. `release_baseline.py`로 100명·411개 투구를 선수 단위 leave-one-out으로 평가했다. 학습 평균 예측 MAE 6.163km/h, 손 속도 동일시 48.857km/h, 손 속도 회귀 4.712km/h, 손·손목·어깨 속도+신장·체중 ridge 3.938km/h. 선수별 평균 오차에 같은 가중치를 적용했다. 고정 ridge penalty=1, 특성 표준화는 각 학습 집합에서만 한다. 이 회귀를 수정 동작의 인과 모델로 최적화하지 않는다. release 시점의 동작을 쓰는 오프라인 분석이며 사전 예측이 아니다.
2. **공개 근육 모델 실행 — 기초 확인 완료.** myo-sim 0.2.3의 myoarm_r, 38자유도·63구동기, MuJoCo 3.17.0에서 `mj_step` 실행. 0 및 0.05 균일 excitation으로 각각 1초 동안 유한한 상태를 유지하고 입력에 따른 동작 차이를 확인했다. 이는 투구 타당성 또는 최대 힘 검증이 아니다. 컴파일 단계에 XML 저장 관련 empty-prefix 경고가 있으므로 재저장 모델의 재현성을 가정하지 않는다. 실행 중 경고는 없었다.
3. **다음 작업: 투구 동작과 팔 모델 연결.** 우선 우투수 한 명의 발 착지~release 구간을 사용한다. 팔 모델의 좌표·관절 정의·관절 범위와 OBP 정의를 대응시킨다. 실제 체격에 대한 분절 크기 보정, 몸통/어깨의 이동 경계조건, 구속을 보지 않은 동작 fitting을 구현한다. 상지 모델이면 전신 최적화라고 부르지 않는다. prescribed 몸통 운동이 외부에서 공급하는 힘/일도 명시한다. 단순 qpos replay는 준비 단계로 표시한다.
4. **힘으로 기준 동작 생성.** 역동역학 및 근육/구동기 제어로 기준 동작을 재현한다. 잔여 힘·보조 구동기 기여·관절 범위·접촉·수치 안정성을 기록한다. 데이터의 모멘트는 proximal/distal 좌표 정의를 맞춘 후 비교한다. 보조 힘으로 무리하게 재현한 경우 통과시키지 않는다. 측정 근력이 없으므로 일반 모델 파라미터와 민감도 범위를 사용한다고 명시한다.
5. **공 release와 독립 투구 검증.** 손 대리점을 공 중심으로 취급하지 않는다. 공 위치/손 결합/이탈 시점 가정을 문서화하고 질량·구속 정의·단위를 일치시킨다. 관측된 공 속도를 모든 투구에 맞춰 놓고 예측이라고 주장하지 않는다. 별도 평가 투구의 실제 구속, release 시점 민감도, 실패 사례를 분석한다. 동작 적합도·잔여 힘·구속 예측 허용 기준은 후보 탐색 전에 측정 자료 및 문헌에 근거해 고정한다. 기준을 정하지 못했으면 평가치를 보고하되 검증 통과를 선언하지 않는다.
6. **작은 수정 후보 탐색.** 기준 동작, 운동학/관찰 기반 수정, 동일 능력의 물리 제약 수정 비교. 관절 한 개의 정적 각도보다 일정 구간의 협응/타이밍 변화를 명확히 매개변수화한다. 모델의 근력·외부 구동력·몸통 경계조건을 방법별로 유리하게 바꾸지 않는다. 변경량·요구 모멘트·구속(km/h)·추천 보류율을 보고한다.
7. **불확실성과 보고.** 근력/관성/release/관측 오차를 타당한 범위에서 바꾸고 구속 증가의 부호와 후보 순위가 유지되는지 확인한다. 학습에 쓰지 않은 투구/선수 평가와 조건부 예측의 한계를 보고한다. 전 범위에서 향상이 없으면 음성 결과로 정리한다. 목표 위치가 없으면 제구 유지, 실제 개입이 없으면 훈련 효과를 주장하지 않는다.

검증 실패 시 원인을 좁혀 수정하고 다시 평가한다. 같은 부적합 모델에서 후보 탐색을 강행하지 않는다. 실제 선수 실험이나 유료·비공개 자료가 반드시 필요한 지점에서는 부족한 증거를 구체적으로 보고한다. 공개 자료로 가능한 대체 경로는 먼저 시도한다.

## 재실행

초기 기준 방법 작업 폴더: `outputs/pitching-lab/dynamics-baseline`. 최신 코드는 `outputs/pitching-lab/pitch-fitting`에 있다.
Python: 상위 `pitching-lab/.venv/Scripts/python.exe`.

```powershell
uv pip install --target artifacts/vendor --no-deps myo-sim==0.2.3
../.venv/Scripts/python.exe release_baseline.py
../.venv/Scripts/python.exe muscle_model_check.py --render
../.venv/Scripts/python.exe muscle_model_check.py --gui
../.venv/Scripts/python.exe -m unittest -v
```

원자료는 상위 `openbiomechanics/baseball_pitching/data`에 있다. 결과는 `artifacts/release-baseline` 및 `artifacts/muscle-model`. 원자료·파생 선수별 결과·외부 모델은 Git에 올리지 않는다. 기존 공유 환경을 변경하지 않고 모델 패키지를 작업 폴더 안에 설치했다.

후속 실행은 작업 기록과 PR 상태를 먼저 읽고 이어간다. 코드 변경 시 사용자 Git 격리 지침에 따라 최신 remote default에서 새 브랜치와 별도 worktree를 만들고, 필요 시 기존 검증 커밋을 명시적으로 가져온다. 이전 PR을 자동 병합하지 않는다. 중복 실행·중복 PR을 피하고 단계의 완료/미완료 근거를 갱신한다.

## 출처

- [OpenBiomechanics 좌표·신호 설명](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching)
- [MyoSim 공개 모델](https://github.com/MyoHub/myo_sim), Apache-2.0, 사용 버전 0.2.3
- [MyoArm 설명](https://github.com/MyoHub/myo_sim/blob/main/myo_sim/models/arm/README.md)

2026-10-25: 1~2단계 실행. 다음은 3단계. 아직 투구 근육 제어·공 release·수정 구속 향상 예측은 완료하지 않았다.
