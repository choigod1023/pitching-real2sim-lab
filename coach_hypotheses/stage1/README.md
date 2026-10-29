# 코치 가설 관측 분석: 단계1

98명399구에서 지표 추출, 선수 간/내 관계, 선수 단위 교차검증 및 추가 필터
민감도를 구현했습니다. **예측 사전 합격 기준은 FAIL**입니다. 관측 평균을 뺀
투구 간 차이의 MAE는 추가20Hz에서0.639517km/h로, 평균 구속 기준0.629742km/h보다
낮지 않았습니다. 이는 물리 시뮬레이션의 절대 구속 오차가 아닙니다.

- [결과·명령·검증·해석 한계](FINDINGS.md)
- [계산 전 커밋한 원래 프로토콜](PROTOCOL.md)
- [원문 확인 수준과 2차 요약의 정정](LITERATURE_REVIEW.md)
- [첫 실행 후 발견한 무릎 속도 부호의 정의 정정](AMENDMENT_01_KNEE_SIGN.md)
- [최종 집계](results/summary.json), [전체 상관과 구간](results/associations.csv)
- [필터 민감도](results/filter_sensitivity.csv), [제외 목록](results/exclusions.json)
- [최초 결과 보존](results_initial_protocol/summary.json)

연구·관찰 분석용이며 훈련 또는 의료 조언이 아닙니다. 실제 선수의 성능·안전·
편안함 변화, 근력 보존, 신체 개입의 인과 효과를 검증하지 않았습니다.
단계2의 방출 불확실성 전파와 단계3의 개입 설계/구현은 수행하지 않았습니다.

## 재실행

저장소 루트에서 Python3.12 환경을 사용합니다. OBP 원자료를 별도로 받아야 합니다.

```powershell
& ../.venv/Scripts/python.exe -m unittest discover -s coach_hypotheses/stage1 -t . -v
& ../.venv/Scripts/python.exe -m coach_hypotheses.stage1.run --data ../openbiomechanics/baseball_pitching/data
```

기본 출력은 무시되는 stage1/artifacts/results입니다. 추출한 CSV의 단위/시간창은
PROTOCOL.md와 부호 정정 문서를 함께 읽어야 합니다. 원래 프로토콜을 소급 변경하지
않았으며, 사후 정정 결과를 완전한 최초 사전등록 결과라고 표시하지 않습니다.

## 출처·라이선스

자료: [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics),
Driveline Baseball Research and Development, [openbiomechanics.org](https://openbiomechanics.org).
지표 추출·추가 필터·중심화·통계 요약으로 변경했습니다. results/와
results_initial_protocol/의 OBP 파생 자료는 원자료와 같은
[CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 조건입니다.
[원자료 사용 조건](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)과
저장소 NOTICE.md를 따릅니다. 코드의 MIT 라이선스와 파생 자료 라이선스를 구분합니다.
원본 C3D, 영상, 논문 전문은 포함하지 않습니다. Driveline의 보증을 받지 않았습니다.
