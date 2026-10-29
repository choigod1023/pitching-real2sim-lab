# 관측 비교의 오차 여유·이벤트 검사

선행 PR123의 59개 관측 비교에 대해, 부하 지표의 차이가 얼마나 작은 오차와 이벤트 이동에 취약한지 검사한다. 신체 개입이나 구속 예측 모델이 아니며 훈련·의료 조언을 제공하지 않는다. 시뮬레이션된 오차가 실제 센서/역동역학 오차라는 뜻은 아니다.

- 사전 기준: [PROTOCOL.md](PROTOCOL.md)
- 결과와 해석: [FINDINGS.md](FINDINGS.md)
- 요약/모든 쌍: [results/summary.json](results/summary.json)
- 원본 해시: [results/manifest.json](results/manifest.json)
- 독립 검산/테스트: [results/verification.json](results/verification.json)

## 실행

저장소 루트, Python 3.12의 기존 numpy/pandas/scipy 환경에서 실행한다. 21b2921 해시의 원 지표 생성기와 단계1 로컬 CSV가 필요하다. 나중에 수정된 동명 파일이나 성능 좋은 다른 지표로 대체하지 않는다.

```powershell
& ../.venv/Scripts/python.exe -B -m unittest coach_hypotheses.observed_sensitivity.test_analysis -q
& ../.venv/Scripts/python.exe -B -m coach_hypotheses.observed_sensitivity.analysis --prior-results ../observed-speed-load-tradeoffs-20261029/coach_hypotheses/observed_tradeoffs/results --metrics-dir ../coach-hypotheses-stage1-20261029/coach_hypotheses/stage1 --data-dir ../openbiomechanics/baseball_pitching/data
```

all_event_comparisons.csv(14,337행), all_nominal_margins.csv(177행), knee_direction_audit.csv(288행)는 로컬 결과다. 원본·대량 CSV는 커밋하지 않는다. 프로토콜의 오차 상한과 ±1프레임은 공학적 검사 조건이며 실제 오차 확률/상한으로 검증하지 않았다.

## 자료 이용 조건

출처는 Driveline Baseball의 [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching)다. 원자료에서 검색 구간을 바꾼 극값과 차이·집계를 생성했다. 결과·파생 수치는 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 및 상류 [LICENSE-DATA.md](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)의 이용/인용 조건을 유지한다. 코드 MIT 라이선스는 자료 이용 조건을 대체하지 않는다. 저자의 승인이나 선수 효과를 암시하지 않는다.
