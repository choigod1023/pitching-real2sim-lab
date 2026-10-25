# 2026-10-25: 선수 내 학습과 실제 동역학 적분

## 결과와 해석

`torque_arm.py`는 OpenBiomechanics의 우투수 투구 3034_1, 3034_2에서 어깨·팔꿈치·손목 위치를 읽어 고정 길이 2분절 팔에 대응시켰다. 역동역학으로 계산한 외부 어깨 힘/토크와 팔꿈치 토크를 초기 상태에서부터 `mj_step`에 넣었다. 중간 자세 재설정이나 피드백 제어는 없다. 단순한 좌표 애니메이션에서 실제 힘 기반 적분으로 진행한 수치 기준 실험이다.

- 3034_1 손목 재현 RMSE: 시간 간격 약 1/1440, 1/2880, 1/5760초에서 6.235, 3.134, 1.571mm.
- 3034_2: 같은 간격에서 6.358, 3.172, 1.584mm.
- 가장 작은 간격에서 최대 차이: 각각 4.538, 4.583mm. 이는 **고정 길이로 대응시킨 기준 궤적 대비**이며 원시 측정 손목에 대한 정확도가 아니다.
- 3034_1 외부 어깨 힘/토크 최고치 819.1N/304.6Nm, 팔꿈치 토크 90.9Nm. 3034_2는 753.0N/395.0Nm/99.8Nm. 모두 단순 모델의 계산값이며 선수의 실제 힘이나 생리학적 한계로 해석하지 않는다. 최고치는 시간 간격에 아직 민감하다.

팔 길이는 관찰값으로 정하지만 질량은 일반 모델의 상완 2.14154kg, 전완 1.280182kg을 썼다. 관성은 반경 4cm/3cm의 균일 원기둥 가정이다. 개인별 체성분 보정이 아니며 손·공·근육·관절 가동 범위·접촉이 없다. 상완 비틀림은 팔꿈치 평면으로 정하고 독립 측정값이라고 주장하지 않는다. 어깨는 외부 힘이 작용하는 자유 관절이다. 따라서 같은 능력의 구속 최적화를 바로 시작할 수 없다.

`centered_release.py`는 학습 선수 각각의 평균을 제거한 잔차로 rigid offset과 scalar slope를 적합했다. 77명·325개 투구에서 선수 단위 제외 학습, 248개의 동일 선수 차이를 평가했다. 20Hz에서 구속 변화 MAE는 offset 1.592km/h, scalar 1.156km/h, 무변화 기준 1.128km/h였다. 이전 absolute-offset 방식의 2.592km/h보다 개선했으나 기준을 넘지 못했다. 원신호에서는 offset 6.072, scalar 1.135km/h였다. 결과를 이미 살펴본 코호트이므로 독립 외부 검증이 아닌 탐색적 개발 결과다.

20Hz 손 마커 속도의 선수 간 평균 상관은 0.705지만 선수 내 중심화 상관은 0.189였다. 선수 구분에 도움이 되는 특징이 개인의 동작 수정 효과를 잘 예측한다는 뜻은 아니다. 이 자료는 같은 세션의 관찰 차이이며 훈련 개입 효과가 아니다. 학습된 offset은 그립을 식별한 값으로 취급하지 않는다.

## 검증·재현성과 출처

35개 Python 테스트 통과. 알려진 위치 재구성, 중력 평형, 힘을 제거했을 때 0.1초간 49.5405mm 자유낙하, 퇴화 자세 거부, 평가 선수의 목표값을 바꿔도 학습 예측이 변하지 않음, 선수별 상수를 더해도 중심화 적합 불변 등을 검사했다. 기존 MyoSim 적합 검사에서는 empty-prefix 저장 경고가 남아 있으며 새 2분절 모델의 적분 불안정 경고는 없었다.

입력 특성 SHA256: `3699c1dffaf6bb19762378696ccf291d72866e766adab32050f57409176b3ed7`. 결과는 로컬 `artifacts/centered-release/report.json` 및 `artifacts/torque-arm/3034_1/report.json`, `3034_2/report.json`에 있다. 원자료·외부 모델·선수별 파생 파일은 커밋하지 않는다.

- [Rajagopal OpenSense 모델의 질량 출처](https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal_OpenSense/Rajagopal2015_opensense.osim): 질량만 참고했고 원본 관성/관절/모델을 변환한 것이 아니다.
- [MoBL 원본 계열 파일](https://raw.githubusercontent.com/CEINMS-RT/UpperLimbModel/459e2ebbf47acb72b0ecbc59950a2d0a983d28db/MOBL_ARMS_41.osim): phantom 관성 1은 원본 계열에도 있어 단순 변환 오류로 단정할 수 없다. 이를 임의의 작은 값으로 바꾸어 정답으로 쓰지 않았다.
- [투구 전방 동역학 연구의 저자 공개본](https://uwaterloo.ca/motion-research-group/sites/default/files/2026-01/mbsd_pub_baseball_fd_cattias_final.pdf): 직접 연결한 어깨 관절 구성을 검토할 참고 자료이며 본 2분절 모델이 그 논문 모델을 재현한 것은 아니다.

다음은 질량·관성 및 외부 일 민감도, 손·공과 이탈 조건의 식별 가능성 검사다. 현재 결과로 구속 향상량 또는 선수 처방을 제시하지 않는다.

---

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
