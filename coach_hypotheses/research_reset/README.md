# 연구 재설계와 실제 자료 검증

설계는 `9e70bc8`에서 새 계산 전에 고정했습니다. 설계와 자료 감사 에이전트 2개를 병렬로 사용했고, 구현과 고정 실험은 부모 작업 폴더에서 수행했습니다. 읽기 전용 검토에서는 새 코드의 누출·반대칭·선수 가중·bootstrap 오류를 발견하지 못했습니다. 새 모델이나 관절 수정의 효과를 입증한 작업이 아닙니다.

- [사전 프로토콜](PROTOCOL.md): 자료/목표/선수 분할/판정/중단 조건
- [실제 결과와 한계](FINDINGS.md): 자료 수, 세 필터 비교, 검사 명령과 오류
- [수정된 다음 계획](NEXT_STEPS.md): 병렬 가능한 작업과 금지할 단계
- [계산에서 추출한 요약](SUMMARY.json): 원자료 해시·검증·수치·판정

## 재현

프로젝트 루트에서 실행합니다. `python`은 이 작업에서 사용한 `../.venv/Scripts/python.exe`입니다. `--metrics`와 `--source-manifest`는 기존 단계1의 고정 산출물입니다. 새 데이터로 바꿔 좋은 수치를 선택하지 않도록 해시를 검사합니다. 해당 파일이 없는 환경에서는 기존 단계1을 원래 프로토콜대로 재현하고 같은 해시인지 확인해야 합니다.

```powershell
python -m unittest coach_hypotheses.research_reset.test_validation -v
python -m coach_hypotheses.research_reset.run --data ../openbiomechanics/baseball_pitching/data --metrics ../coach-hypotheses-stage1-20261029/coach_hypotheses/stage1/results/pitch_metrics.csv --source-manifest ../coach-hypotheses-stage1-20261029/coach_hypotheses/stage1/results/manifest.json
python -m unittest discover
```

새 검사 14개는 모두 통과했습니다. 전체 303개 중 291개 통과, 기존 환경/로컬 산출물 오류 10개, 건너뜀 2개이며 새 실패는 없습니다. 소프트웨어 검사 통과는 연구 가설 통과가 아닙니다. 이번 쌍 예측 실험의 판정은 **FAIL**입니다.

`artifacts/`에 투구쌍 예측 CSV, 선수 단위 분할, GRF 적격성 기록, 상세 검사 로그를 저장합니다. Git에는 집계 요약만 넣습니다. 원본 C3D·영상은 추가하지 않습니다. [OpenBiomechanics Project / Driveline Baseball R&D](https://github.com/drivelineresearch/openbiomechanics) 자료 및 파생 결과의 CC BY-NC-SA4.0, [NOTICE.md](../../NOTICE.md)와 인용 요구를 따릅니다. 훈련이나 의료 조언이 아니며 실제 성능·안전·편안함을 검증하지 않았습니다.
