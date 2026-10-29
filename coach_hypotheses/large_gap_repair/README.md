# 3034 구속 차이 직접 수정 검사

같은3034_1~5,같은15cm 가정,같은BR,같은20Hz에서 손 축과 손목 전달을 직접
분리했습니다. **의미 있는 구속 오차 감소에 실패했고 새 경로를 채택하지 않았습니다.**
5구 평균 차이는17.049771km/h였고,손목 전달 경로 수정은16.832250km/h로
0.217520km/h만 작아졌습니다. 관절중심 축 경로는24.673683km/h로 악화했습니다.

- [수치·검증·막힌 조건](FINDINGS.md)
- [계산 전 계획](PROTOCOL.md)
- [모든5구×6경로](results/paths.csv), [집계](results/summary.json)

이 값은 가정 공 지점의 기구학이며 실제 공 중심 관측이나 접촉·이탈 검증이
아닙니다. 원래 팔·손·공 적분 영상과 숫자를 섞지 않았습니다. 다른 자료·학습
모델의 작은 오차로 대체하거나실측 구속에 그립·이탈·힘을 맞추지 않았습니다.
훈련 또는 의료 조언이 아니며 선수 향상·안전·편안함을 입증하지 않았습니다.

자료는 Driveline Baseball Research and Development의
[OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics)에서
왔습니다. results/는 이를 기구학 재구성·미분·요약한 파생 자료로 CC BY-NC-SA4.0
및 NOTICE.md 조건을 따릅니다. 코드의MIT와 자료 조건을 구분합니다.
원본C3D/영상/논문전문은 저장소에 넣지 않았습니다.
