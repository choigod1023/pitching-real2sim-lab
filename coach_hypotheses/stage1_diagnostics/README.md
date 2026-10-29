# 단계1 실패 원인 진단

합격 기준을 유지한 후속 비교를 완료했습니다. 개선 선택기는 추가20/30Hz에서
98명 모두 평균 예측(null)을 선택했습니다. 주 MAE는0.639517→0.629742km/h로
줄었지만 평균 기준과 같아서 **FAIL**입니다. 새 동작 정보의 예측 우위를 입증하지 못했습니다.

- [원인별 검증·전체 결과·명령·한계](FINDINGS.md)
- [계산 전 계획](PROTOCOL.md), [정규화 효과 분리 계획](REGULARIZATION_CHECK.md)
- [원문 확인 수준](SOURCES.md), [기계 판독 집계](results/summary.json)
- [고정alpha 대조](results/matched_penalty_ablation.csv)
- [전체 예측](results/predictions.csv), [자료·지표 진단](results/feature_diagnostics.csv)

새 코드는 이 폴더 안에만 있습니다. 원래 단계1의 프로토콜·코드·결과와 기존
시뮬레이션을 바꾸지 않았습니다. 단계2의 조건부 방출 불확실성 전파는 사용자
승인에 따라 별도 worktree/PR에서 병렬 수행했고, 여기의 실패 판정을 통과로
바꾸거나 단계3 개입 구현을 시작하지 않았습니다.

연구용 관측 분석이며 훈련 또는 의료 조언이 아닙니다. 실제 선수 효과·안전·편안함·
근력 보존·인과적 동작 교정을 검증하지 않았습니다. 자료는
[OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics),
Driveline Baseball Research and Development의 자료를 필터링·중심화·요약했습니다.
results/의 파생 자료는 [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
조건이며 저장소 NOTICE.md와 원자료 사용 조건을 따릅니다. 원본 C3D/영상/논문
전문은 포함하지 않습니다. 코드의 MIT 조건과 파생 자료의 라이선스를 구분합니다.
