# 관측 투구의 구속·팔 부하 비교

같은 선수의 실제 투구를 기록 구속, 팔꿈치 순 관절 모멘트, 어깨 힘 지표로 비교한다. 동작 변경의 효과를 예측하는 모델이 아니며 훈련·의료 조언을 제공하지 않는다. 요구 부하 지표의 감소는 최대 근력의 동일성이나 부상 위험 감소를 뜻하지 않는다.

- 사전 기준: [PROTOCOL.md](PROTOCOL.md)
- 비교 계산 전 부호 정정: [AMENDMENT_01_SIGNED_TENSION.md](AMENDMENT_01_SIGNED_TENSION.md)
- 결과·제한·다음 단계: [FINDINGS.md](FINDINGS.md)
- 집계/사례: [results/summary.json](results/summary.json)
- 출처/해시: [results/manifest.json](results/manifest.json)
- 독립 검산/테스트: [results/verification.json](results/verification.json)

## 재현

저장소 루트에서 Python 3.12와 기존 numpy/pandas 환경을 사용한다. metrics-dir은 기존 단계1 폴더이며 그 안의 results/pitch_metrics.csv SHA256을 고정했다. 이 분석을 위해 기존 파일을 다시 만들거나 수정할 필요가 없다.

```powershell
& ../.venv/Scripts/python.exe -B -m unittest coach_hypotheses.observed_tradeoffs.test_analysis -q
& ../.venv/Scripts/python.exe -B -m coach_hypotheses.observed_tradeoffs.analysis --metrics-dir ../coach-stage1-diagnostics-20261029/coach_hypotheses/stage1 --data-dir ../openbiomechanics/baseball_pitching/data
```

원본 자료와 기존 추출 결과는 별도로 확보해야 한다. 상류 추출의 제외/필터/축 정의를 보존하며 오류를 0이나 보간으로 채우지 않는다. 로컬 결과 all_pair_filter_comparisons.csv에는 1,998개 비교 행을, all_pair_stability.csv에는 모든 666쌍을 저장한다. 대량 CSV와 원본 C3D/영상은 커밋하지 않는다. 결과는 package 내부에만 출력하도록 CLI를 제한한다.

## 자료 이용 조건

출처: Driveline Baseball, [The OpenBiomechanics Project — baseball_pitching](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching).

기존 OBP 가공 지표에서 선수 내 차이와 집계를 생성했다. 결과 문서·집계·파생 수치는 원자료와 같은 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 조건으로 제공한다. 상류의 [LICENSE-DATA.md](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)의 이용 조건과 인용 요구를 확인해야 한다. 코드의 MIT 라이선스가 자료의 비상업/동일조건변경허락 조건을 대체하지 않는다. 저자가 이 분석이나 선수 처방을 승인했다는 뜻은 아니다.
