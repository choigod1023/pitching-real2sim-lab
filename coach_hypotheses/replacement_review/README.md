# 야구 구속 모델 교체 검토

근골격·물리, 관측 학습, 물리–학습 결합을 병렬 조사했습니다. 확인한 범위에서
같은 힘·일 예산의 작은 자세 변경에 따른 실제 공 구속 변화를 검증한 즉시 교체품은
없었습니다. 현재 모델을 자세 추천의 근거로 유지할 근거도 확보하지 못했습니다.

작은 비선형 교체 대조의 주 MAE는0.618655km/h로 평균 기준0.629742보다 작았으나,
차이의95%구간이0을 포함하고10Hz에서 악화해 기존 합격 기준은FAIL입니다.
실제 물리 모델의 약16.8km/h 차이를 줄였다는 결과가 아닙니다.

- [통합 판단·실험·다음 우선순위](FINDINGS.md)
- [물리·근골격 원문/코드 감사](physics/PHYSICS_REVIEW.md)
- [실측 학습·새 TrackMan 자료 감사](learning/LEARNING_REVIEW.md)
- [물리–학습 결합 검토](hybrid/REVIEW.md), [코드·라이선스 근거](hybrid/SOURCE_AUDIT.md)
- [실험 전 프로토콜](BENCHMARK_PROTOCOL.md), [전체 결과](results/summary.json)

후보를 검토하고 비교 실험을 추가했으며 기존 앱의 예측 코어를 교체·배포하지
않았습니다. 기존 프로토콜·결과·과학 gate와 PAUSED 루틴은 변경하지 않았습니다.
원문·제3자 코드·원본 C3D/영상/MAT는 저장소에 추가하지 않았습니다.

이 연구는 훈련 또는 의료 조언이 아닙니다. 선수 효과·안전·편안함·인과를
입증하지 않았습니다. results/는 Driveline Baseball Research and Development의
[OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics)
자료를 추가필터·중심화·모델 평가한 파생 자료입니다. CC BY-NC-SA4.0와 저장소
NOTICE.md/원자료 조건을 따릅니다. 새50명 자료는 라이선스 미확인으로 재배포하지
않았습니다. 작성 코드의 MIT 조건과 외부 자료의 라이선스를 구분합니다.
