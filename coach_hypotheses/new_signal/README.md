# 새 전신 신호와 손–공 평가 분리

사용자 요청에 따라 이전 7개 지표 튜닝 대신 전신 파워·지면반력의 시계열 정보를 추가하고, 손–공 pilot의 로컬 미분 지지 구간을 학습 표본과 분리했습니다. 새 계산 전 계획은 `c3f0698`로 커밋했습니다. **이번에도 채택할 예측/추천 모델은 얻지 못했습니다.** 기존 실패, 기준, 앱 사용 중단 및 PAUSED 자동화는 유지했습니다.

- [사전 프로토콜](PROTOCOL.md)
- [세 필터·전체 모델·손–공 결과](FINDINGS.md)
- [집계 수치와 원본 해시](SUMMARY.json)
- [새 근거 확보의 다음 순서](NEXT_EVIDENCE.md)

## 재현 명령

프로젝트 루트에서 실행합니다. `python`은 이 환경의 `../.venv/Scripts/python.exe`입니다. 기존 stage1 지표와 pilot의 해시가 고정돼 있으며 다른 파일로 몰래 대체하지 않습니다. 다른 작업 폴더는 입력으로만 읽고 새 출력은 이 패키지의 artifacts에 둡니다.

```powershell
python -m unittest coach_hypotheses.new_signal.test_new_signal -v
python -m coach_hypotheses.new_signal.run --data ../openbiomechanics/baseball_pitching/data --metrics ../coach-hypotheses-stage1-20261029/coach_hypotheses/stage1/results/pitch_metrics.csv --source-manifest ../coach-hypotheses-stage1-20261029/coach_hypotheses/stage1/results/manifest.json
python -m coach_hypotheses.new_signal.guard --data ../public-hand-ball-evidence-20261029/coach_hypotheses/measurement_evidence/artifacts/S1 --pilot-module ../observed-hand-ball-pilot-20261029/coach_hypotheses/observed_release_pilot/pilot.py --pilot-summary ../observed-hand-ball-pilot-20261029/coach_hypotheses/observed_release_pilot/results/summary.json
python -m unittest discover
```

새 검사15/15 통과. 전체318개 중306개 통과, 기존 환경/로컬 산출물 오류10개,skip2개입니다. 새 실패는 없습니다. 소프트웨어 검사 통과와 과학적 예측 검증은 별개입니다.

OBP 파생 수치는 [Driveline Baseball R&D / OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics)의 CC BY-NC-SA4.0 및 [NOTICE.md](../../NOTICE.md)를 따릅니다. Ozkaya 표식 자료는 기존 원문/배포 감사의 CC0 자료로 별도 취급합니다. 새 IMU 자료의 Figshare 메타데이터는 CC BY4.0을 명시하지만 원본 데이터는 다운로드에 성공하지 못해 사용하지 않았습니다. 원본C3D·영상·외부CSV·대량 파생CSV를 커밋하지 않습니다.

실제 신체 수정의 향상, 같은 힘 능력/일 예산, 안전성·편안함·훈련 효과를 검증하지 않았습니다. 훈련이나 의료 조언이 아닙니다.
