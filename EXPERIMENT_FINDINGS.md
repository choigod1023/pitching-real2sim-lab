# 투구 연결·힘 요구·공 release 검증 기록

2026-10-25. 전용 브랜치 codex/pitch-fitting-20261025. 원자료와 선수별 파생 자료는 artifacts에만 보관한다. 아래는 단계별 실행 결과이며 실제 선수의 구속 향상 결과가 아니다.

## 지금까지 연속 실행한 작업

1. **관절 제약을 지키는 동작 연결:** 선수2의3034_1, 발 착지~release61프레임. 어깨·팔꿈치·손목 중심에서 흉곽 기준 상완/전완 방향을 만들고 MyoArm의4개 독립 관절을 맞췄다. 11개 종속 관절의 등식 및 종속 관절 범위까지 적용했다. 평균 방향 잔차 상완6.824°,전완5.796°; 최대20.399°,19.220°. 34/61프레임에서 어깨 회전 하한에 도달했다. 관절 범위 위반·접촉은0. 같은 선수의 별도3034_2도34/57프레임 경계 도달, 평균6.843°,6.056°로 문제가 반복됐다. 구속은 fitting 입력으로 쓰지 않았다.
2. **관절 범위 원인 분리:** shoulder_rot 범위를 수치상 ±180°로 넓힌 진단에서 방향 잔차가 거의0이 됐다. 이는 원인 분리이며 해부학적으로 검증한 범위 수정이 아니다. 해당 결과는 기본 모델의 힘 검사에 넘기지 않도록 범위 일치 검사도 추가했다.
3. **순간 근육 힘 검사:** 27개 독립 자유도(고정한 손가락 포함)의 역동역학 요구 힘과63개 근육의 activation[0,1] 범위를 비교했다. 프레임별 reduced-torque RMSE 평균409.33Nm, 최대 성분 잔차18823.56Nm. 이것은 선수의 실제 관절 모멘트가 아니다. 고정된 일반 체격 모델과 방향 retargeting의 부적합을 드러내는 진단이다. activation 시간상수·forward tracking·공 질량은 아직 반영되지 않았다.
4. **궤적과 관성 민감도:** 구간 전체 동시 fitting에서 관절 범위를 유지한 채 second-difference weight10을 적용했다. 최대 각가속도12085.88→1553.96rad/s², stock 모델 평균 잔차409.33→197.29Nm. 방향 평균 잔차는7.324°,6.722°로 늘었다. 임의의 phantom 관성1e-6 조건을 적용하면 이 값이15.06Nm까지 바뀌었다. 이 값은 검증된 관성도 통과 결과도 아니다. arbitrary smoothing·관성·근력 조절로 결과를 좋게 만드는 것을 방지하기 위한 민감도 기록이다.
5. **공 release 단일 투구 검사:** RFIN/RWRA/RWRB로 손 marker 좌표계를 복원하고 v_origin+d(Rr)/dt를 계산했다. 첫 lexicographic 우투수1031_2의 실측145.485km/h에 비해 RFIN raw release105.097,20Hz95.448km/h. 설정한 ±0.02m 및±1frame 조건의 속도도 재현에 부족했다. 이 offset은 해부학적 범위가 아닌 수치 설정이다. read-only 진단상 raw 전체 구간 최대108.139km/h로, 시점 이동만으로 차이를 없앨 수 없었다.
6. **다른 선수에서 공 위치 보정 평가:** 우투수328개 중 RFIN 없는3개를 제외한325개·77명. 선수 leave-one-out, 학습 선수 균등 가중 손실, 공유3D offset만 보정했다. ±0.10m 성분별 수치 box와7개 시작점을 사용했다. 20Hz athlete-balanced MAE: rigid offset5.085km/h, 단순 선형 보정4.194km/h, 학습 선수 평균5.687km/h. raw는 각각11.267,5.006,5.687km/h. 현재 기하 모델은 선형 보정보다 못하다. 속도 크기로 보정한 offset은 진짜 grip 위치를 식별한 것이 아니며 필터 오차 등을 흡수할 수 있다.
7. **스포츠 질문에 가까운 변화량 검사:** 각 평가 선수의 첫 lexicographic 투구를 기준으로 나머지248개 투구의 관측 구속 차이를 비교했다. 시간 순서나 개입 전후로 해석하지 않는다. 같은 선수의 모든 투구는 구속 예측 모델 학습에서 제외되어 있다. 20Hz 변화량 MAE: 변화 없음1.128km/h, 선형 보정1.599km/h, rigid offset2.592km/h. 현재 모델은 같은 날 자연스럽게 나타나는 구속 차이에서도 단순 무변화 예측을 이기지 못했다. 따라서 작은 수정에 따른 구속 향상 추천으로 넘어갈 근거가 없다.

## 원본 모델 검토

MyoSim0.2.3의 clavphant_r,scapphant_r,humphant_r,humphant1_r는 각각 mass0.001kg와 diaginertia[1,1,1]kg·m²를 갖는다. 외부 에이전트가 [CEINMS-RT 연구기관의 고정 MoBL 배포본](https://raw.githubusercontent.com/CEINMS-RT/UpperLimbModel/459e2ebbf47acb72b0ecbc59950a2d0a983d28db/MOBL_ARMS_41.osim)을 XML 파싱해 원본 계열에도 inertia[1,1,1,0,0,0] 및 mass0.0001kg를 확인했다. SHA25658bacbc1c61d1d42cff9a48b0a9dced9c648893f44adc7d9817cca56441a9fcf. SimTK 최초 배포본과 동일성까지 확인하지 않았다.

따라서 변환 버그라고 단정하거나1e-6으로 고쳐 정상 결과라고 제시하지 않는다. parent가 frame53에서 reduced angular Jacobian을 직접 계산한 결과 clavphant/scapphant의 독립 회전 기여는 수치상0이지만 humphant/humphant1은0이 아니었다. 모든 phantom 관성을 단순 합산하는 해석도 잘못이다. virtual-work projection과 각 body의 운동에너지 기여를 함께 검토해야 한다.

[모델 배포 README](https://github.com/CEINMS-RT/UpperLimbModel/blob/459e2ebbf47acb72b0ecbc59950a2d0a983d28db/README.md)는 프로그램과 모델 이용 조건을 구분한다. 원본 .osim은 저장소에 재배포하지 않았다.

## 다음 실행 순서

- 첫째, 어깨 회전의 해부학적 대응과 관성 파라미터의 용도를 확인하고, 투구 동역학에 적합한 공개 모델/설정을 선택한다. 일반 모델의 한계를 임의로 완화해 성공을 만들지 않는다. 흉곽 기준 방향 fitting을 실제 길이 보정·움직이는 몸통 조건으로 확대할 근거를 마련한다.
- 둘째, 공 위치·손 자세·release 가정의 식별 가능성을 검토한다. 현재 공유 offset의 실패를 모든 물리 모델의 실패로 확대하지 않는다. 필요한 경우 공개 IMU/근전도 자료 또는 별도 공개 모델로 검증 범위를 분리한다.
- 셋째, 새 모델은 절대 구속 예측과 **선수 내 구속 변화량**을 모두 비교한다. 후자는 새 핵심 gate다. 평가지표 자체의 개선도 실제 동작 개입의 인과 검증과는 다르다.
- 위 검증이 성립한 뒤에 같은 힘 능력 안의 작은 수정 후보와 불확실성 분석을 수행한다. 지금 단계에서는 개선 구속 숫자를 출력하지 않는다.

## 재실행 명령

```powershell
../.venv/Scripts/python.exe pitch_fit.py
../.venv/Scripts/python.exe pitch_fit.py --pitch 3034_2 --output artifacts/pitch-fit-second
../.venv/Scripts/python.exe pitch_fit.py --temporal-weight 10 --output artifacts/pitch-fit-temporal
../.venv/Scripts/python.exe pitch_fit.py --diagnostic-rotation-span-deg 180 --output artifacts/pitch-fit-range-diagnostic
../.venv/Scripts/python.exe muscle_capacity.py
../.venv/Scripts/python.exe diagnose_capacity.py
../.venv/Scripts/python.exe ball_release.py
../.venv/Scripts/python.exe release_calibration.py
../.venv/Scripts/python.exe within_pitcher.py
../.venv/Scripts/python.exe -m unittest -v
```

공식 OBP [가공 신호 설명](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching)과 [Visual3D 모델](https://github.com/drivelineresearch/openbiomechanics/blob/main/baseball_pitching/code/v3d/model/v6_model_hybrid_lm.mdh)을 참조했다. RFIN 및 hand_jc는 공 중심 측정값이 아니다. 데이터의 관절 모멘트와 위 reduced-coordinate 모멘트는 좌표 정의를 맞추기 전 직접 비교하지 않는다.
