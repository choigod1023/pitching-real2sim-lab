# 골반–몸통 피크 안정성 감사

기존 타이밍 특징이 필터/최대점 선택/평탄부/이벤트 검색에 얼마나 민감한지 확인한다. 원시 마커에서 각속도를 재구성하거나 새 구속 예측기를 학습한 것은 아니다. 훈련·의료 조언이 아니다.

- 사전 기준: [PROTOCOL.md](PROTOCOL.md)
- 결과: [FINDINGS.md](FINDINGS.md)
- 상세 요약: [results/summary.json](results/summary.json)
- 출처 해시: [results/manifest.json](results/manifest.json)
- 독립 검산/테스트: [results/verification.json](results/verification.json)
- 병렬 구위 조사: [RESEARCH_REVIEW.md](../pitch_quality/RESEARCH_REVIEW.md), [SOURCE_AUDIT.json](../pitch_quality/SOURCE_AUDIT.json)

## 재현

Python3.12의 기존 numpy/pandas/scipy 환경에서 저장소 루트 기준으로 실행한다. 기존 extractor와 데이터는 별도로 필요하며 해시가 달라지면 실패하도록 했다.

```powershell
& ../.venv/Scripts/python.exe -B -m unittest coach_hypotheses.peak_stability.test_audit -q
& ../.venv/Scripts/python.exe -B -m coach_hypotheses.peak_stability.audit --data-dir ../openbiomechanics/baseball_pitching/data --metrics-dir ../coach-hypotheses-stage1-20261029/coach_hypotheses/stage1 --source-summary ../observed-tradeoff-sensitivity-20261029/coach_hypotheses/observed_sensitivity/results/summary.json
```

all_peak_candidates.csv의21,546행은 로컬 결과로 보존한다. 원시 C3D/촬영 영상/대량CSV를 커밋하지 않는다. 입력자료는 이미20Hz 처리됐으며 추가10/20/30Hz는 원시 재처리가 아니다. 상대 높이0.5/1/2%는 공학 조건이며 측정 오차 분포가 아니다. 시차/가능 범위를 실제 정확도·구속 향상량·선수 교정량으로 부르지 않는다.

파생 결과는 Driveline Baseball의 [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching) 자료에서 새로 집계한 것이다. [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 및 [상류 데이터 이용 조건](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)을 유지한다. 코드의 MIT 라이선스는 데이터 이용 조건을 대체하지 않는다.
