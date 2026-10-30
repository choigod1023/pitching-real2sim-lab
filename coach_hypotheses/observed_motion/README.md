# 전신 관측 비교와 겹침 영상

35쌍의 전신 지표를 분석하고, 두 선수의 실제 처리 관절 좌표를 겹쳐 재생한다. 개인별 자세 변경의 구속 효과를 예측한 시뮬레이션이나 원 카메라 촬영 영상이 아니다. 훈련·의료 조언을 제공하지 않으며 실제 성능 향상·안전·편안함·같은 근력/일을 입증하지 않았다.

- 사전 기준: [PROTOCOL.md](PROTOCOL.md)
- 통합 결과: [REPORT.md](REPORT.md)
- 모든 필터·선수별 수치: [COMPARISON_FINDINGS.md](COMPARISON_FINDINGS.md)
- 후속 질문: [HYPOTHESES.md](HYPOTHESES.md)
- 생성한 관절 재생 영상: [media/README.md](media/README.md)
- 코드: [compare.py](compare.py), [overlay.py](overlay.py)

## 재현

저장소 루트에서 기존 Python 3.12 numpy/pandas/scipy/ezc3d/Pillow/imageio-ffmpeg 환경을 사용한다. Windows 렌더링 한글 글꼴은 C:/Windows/Fonts/malgun.ttf다. 분석은 기존 지표와 단계2 결과를 재사용하며 렌더링에는 OBP 원본 자료가 별도로 필요하다.

```powershell
& ../.venv/Scripts/python.exe -B -m unittest coach_hypotheses.observed_motion.test_compare coach_hypotheses.observed_motion.test_overlay -q
& ../.venv/Scripts/python.exe -B -m coach_hypotheses.observed_motion.compare --summary ../observed-tradeoff-sensitivity-20261029/coach_hypotheses/observed_sensitivity/results/summary.json --metrics ../coach-stage1-diagnostics-20261029/coach_hypotheses/stage1/results/pitch_metrics.csv
& ../.venv/Scripts/python.exe -B -m coach_hypotheses.observed_motion.overlay --data-dir ../openbiomechanics/baseball_pitching/data --metrics-file ../coach-stage1-diagnostics-20261029/coach_hypotheses/stage1/results/pitch_metrics.csv --sensitivity-file ../observed-tradeoff-sensitivity-20261029/coach_hypotheses/observed_sensitivity/results/summary.json
```

렌더는 artifacts/에 출력하며 media/는 검증한 생성 영상의 공유 사본이다. 매 프레임 골반을 중심화하거나 시간 정규화/회전/스케일 적합을 하지 않는다. 각 투구 FP10의 수평 엉덩관절 중점만 한 번 빼고 높이와 이후 병진을 보존한다. 앞부분은 FP10 정렬, 뒷부분은 제공 BR 이벤트 정렬이다. 화면 자세는 원 처리20Hz이고, 정적 부하 비교는 기존 추가20Hz 지표로 표시해 구분한다.

## 출처와 이용 조건

Driveline Baseball의 [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching) 자료에서 관절 좌표를 재생하고 투구/선수별 차이·집계를 생성했다. 결과 문서·파생 수치·생성 영상은 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 및 상류 [LICENSE-DATA.md](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)의 이용/인용 조건을 유지한다. 원본 C3D/촬영 영상/대량 CSV는 저장소에 올리지 않는다. 코드의 MIT 라이선스가 자료 이용 조건을 대체하지 않으며 저자의 승인이나 선수 효과를 암시하지 않는다.
