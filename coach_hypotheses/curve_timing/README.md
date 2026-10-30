# 곡선 중심시각 비교

기존 최대점 시차와 양의 각속도 성분 곡선의 면적 중심시각을 같은 자료에서 비교한다. 신체 변경의 효과나 근력/일을 검증하는 모델이 아니며 훈련·의료 조언을 제공하지 않는다.

- 사전 기준: [PROTOCOL.md](PROTOCOL.md)
- 결과: [FINDINGS.md](FINDINGS.md)
- 수치: [results/summary.json](results/summary.json)
- 분할/학습통계: [results/split_audit.json](results/split_audit.json)
- 독립 검산: [results/verification.json](results/verification.json)
- 공식 MLB 자료 감사 기준: [DATA_PROTOCOL.md](../pitch_quality/DATA_PROTOCOL.md)

## 실행

저장소 루트에서 기존 Python3.12 numpy/pandas/scipy 환경을 사용한다. 원래 데이터/지표 생성기는 별도로 필요하며 동결 해시를 검사한다.

```powershell
& ../.venv/Scripts/python.exe -B -m unittest coach_hypotheses.curve_timing.test_compare -q
& ../.venv/Scripts/python.exe -B -m coach_hypotheses.curve_timing.compare --data-dir ../openbiomechanics/baseball_pitching/data --metrics-dir ../coach-hypotheses-stage1-20261029/coach_hypotheses/stage1
```

features/event_features/stability_ranges/predictions CSV는 로컬에만 보존한다. 원본 C3D/촬영 영상/대량 CSV는 커밋하지 않는다. 이미20Hz 처리된 원 신호에 추가10/20/30Hz를 적용했다. 중심시각은 양의 제공 z성분의 시간 가중 평균이지 실제 에너지 전달/근육 일의 중심이 아니다. 계산상 subframe값은 실측 subframe정확도와 다르다.

OBP 파생 결과는 Driveline Baseball의 [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching)에 출처를 두고 [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 및 [상류 자료 이용 조건](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)을 유지한다. MLB 자료에 이 라이선스를 자동 적용하지 않는다.
