# 같은 투구 자료 확보·적격성 감사

예측 모델을 추가하지 않고, 이미 확보한 원자료의 검사 범위를 늘리고 공개 후보의 접근성과 내용을 확인했습니다. 손–공 방출 검증 준비에 필요한 조건을 모두 확인한 자료는 이번6개 후보군에서0개입니다. 이는 공개자료가 전혀 없다는 뜻이 아니라 공 중심/시계/이탈/구속 정의 등의 미확인 항목을 통과 처리하지 않았다는 뜻입니다.

- [새 계산 전 기록](PROTOCOL.md): cbffe4b
- [실제 원자료 결과](FINDINGS.md)
- [출처별 TRUE/FALSE/UNKNOWN 근거](SOURCE_EVIDENCE.json)
- [계산에서 추출한 집계](SUMMARY.json)
- [실험실 전달용 측정 명세](COLLECTION_SPEC.md), [투구 연결 CSV 양식](trials_template.csv)

## 재현

프로젝트 루트에서 아래 명령을 실행합니다. 이번 python은 `../.venv/Scripts/python.exe`입니다. S1.rar/Trialinfo.xlsx의 기존 다운로드 해시를 검사하며, 원본 파일명을 바꾸지 않습니다. 새로 추출한3개 C3D와 링크450행,MAT250개 해시는 이 패키지의 ignored artifacts에만 저장합니다. `tar.exe`의 RAR 읽기 기능을 사용하며 별도 압축 패키지를 설치하지 않았습니다. XLSX도 표준 ZIP/XML로 읽어 수식을 실행하지 않습니다.

```powershell
python -m unittest coach_hypotheses.data_readiness.test_readiness -v
python -m coach_hypotheses.data_readiness.run --raw ../public-hand-ball-evidence-20261029/coach_hypotheses/measurement_evidence/artifacts --download-manifest ../public-hand-ball-evidence-20261029/coach_hypotheses/measurement_evidence/download_manifest.json --mat-root ../replacement-learning-review-20261029/coach_hypotheses/replacement_review/learning/artifacts/cross_data
python -m unittest discover
```

이 브랜치는 최신 원격 main과 사용 중단 선행 e9abea8만 포함합니다. 다른 미병합 분석 브랜치 전체를 가져오지 않았으므로 그 브랜치의 테스트 개수와 다릅니다. 새 검사13개 통과, 전체302개 중290개 통과/기존오류10개/skip2개이며 새 오류는 없습니다. 소프트웨어 검사나 이 자료 적격성이 모델의 과학적 검증 통과를 승인하지 않습니다.

원본C3D/영상/외부CSV/MAT/논문전문/제3자코드는 커밋하지 않습니다. 코드MIT, OBP 자료/파생결과CC BY-NC-SA4.0 및 [NOTICE.md](../../NOTICE.md), Ozkaya 파일의 배포CC0를 구분합니다. 재사용 조건이 미확인인 Takamido 자료는 공개 파일의 구조/무결성만 검사했으며 학습·재배포하지 않았습니다. 외부 연락·유료구매·선수모집·실제측정·PR병합·자동화 재개는 하지 않았습니다.
