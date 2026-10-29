# 단계2: 방출 기구학의 시나리오 불확실성

단계1의 관측 예측과 별도로, 기존3034의5구에 대해 손목–공 거리/상대 이동,
릴리스1프레임, 손 회전 필터의 가정을 확률적으로 전달했습니다. 실제 오차분포를
측정한 결과가 아니라 **사전 고정한 공학적 시나리오의 분위수**입니다.

- [사전 프로토콜](STAGE2_PROTOCOL.md), [후속 수치 정밀도 프로토콜](FOLLOWUP_PRECISION_PROTOCOL.md)
- [결과·검증·한계](FINDINGS.md)
- [최초 MC 전부](results/mc_designs.json), [레이더와 별도 비교](results/recorded_comparison.json)
- [원래 결과를 보존한 고해상도 후속 검사](results/precision_followup.json)

3034_1의 원래15cm 명목 기구학은113.415km/h, 기록은130.196km/h입니다.
주 시나리오의2.5~97.5%분위수는98.050~134.790km/h입니다. 이 넓은 구간에
실제값이 들어간다고 모델의 오차가 줄었거나 정확도가 검증됐다는 뜻은 아닙니다.
힘·일·근육·접촉을 통한 전방 적분이나 실제 선수 효과를 새로 검증하지 않았습니다.

최초 N8192/32768의 엄격한 MC 수렴 기준은FAIL 그대로입니다. 후속 N131072에서
표본 정밀도를 확인했으나 기존 과학적 기준을 소급 통과시키지 않았습니다.
단계1의FAIL도 유지하며 단계3은 수행하지 않았습니다.

연구용이며 훈련 또는 의료 조언이 아닙니다. 실제 선수 성능·안전·편안함을
입증하지 않았습니다. 기존 코드·결과·원자료는 수정하지 않았습니다.

## 재실행

저장소 루트에서 별도 OBP 원자료와 Python3.12 환경이 필요합니다.

```powershell
& ../.venv/Scripts/python.exe -m unittest coach_hypotheses.stage2.test_uncertainty -v
& ../.venv/Scripts/python.exe -m coach_hypotheses.stage2.run --data ../openbiomechanics/baseball_pitching/data
& ../.venv/Scripts/python.exe -m coach_hypotheses.stage2.precision --data ../openbiomechanics/baseball_pitching/data
```

## 출처와 라이선스

자료: [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics),
Driveline Baseball Research and Development, [openbiomechanics.org](https://openbiomechanics.org).
자료의 필터/기구학 재구성/MC요약을 변경·추가했습니다. results/의 파생 자료는
[CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)으로 공유합니다.
[원자료 사용 조건](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)과
저장소NOTICE.md를 따르며 코드의MIT와 파생 자료 라이선스를 구분합니다.
원본 C3D/영상/논문 전문은 포함하지 않습니다. Driveline의 보증을 받지 않았습니다.
