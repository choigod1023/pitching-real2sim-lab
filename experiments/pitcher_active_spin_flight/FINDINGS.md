# 투수별 평균 유효 회전율을 적용한 포심 비행 결과

## 직접 측정 자료 조사

공개 Statcast에서 투구별로 확인할 수 있는 값은 총회전수와 광학 회전벡터의 X-Z 평면 투영각이다. 투구별 깊이 성분과 유효 회전 비율은 공개 CSV에 없다. [Nathan의 자료 설명](https://baseball.physics.illinois.edu/HawkeyeAveSpinComponents.pdf) · [Savant 필드 정의](https://baseballsavant.mlb.com/csv-docs)

- Nagami 외는 투수 7명, 포심 21구의 비행과 3D 회전축을 별도로 촬영했다. 공개 본문에 평균과 관계 그림은 있으나 이번 방식으로 비교할 투구별 원시 행과 비행 좌표는 찾지 못했다. [학회 원문](https://isbweb.org/images/conf/2011/ScientificProgram/ISB2011_ScientificProgram_files/1057.pdf)
- Ijiri의 공개 포심 예시는 고속 영상으로 3D 회전을 추정하는 자료다. 공개 샘플에 투수 식별자, 무브먼트, 홈 앞면 높이가 없다. [공개 코드와 예시](https://github.com/TakashiIjiri/SpinTracker_Python)
- Baseball Aerodynamics의 기존 공개 Rapsodo 예시에는 투구별 회전 방향·자이로 각도·회전 효율·판면 높이가 있다. 동일 투수 식별 정보가 비어 있고 장비 설정과 릴리스에서 판면까지의 거리 및 투구별 비행 궤적을 확인하지 못했다. [예시 제공 페이지](https://baseballaero.com/services/)
- Sportradar는 투구 이벤트의 백스핀·사이드스핀을 문서화했다. 실제 이벤트 자료에 접근하려면 Statcast 접근 권한이 있는 API 키가 필요하며 공개 응답 예시에는 완전한 축과 비행 자료가 없다. [필드 정의](https://developer.sportradar.com/baseball/page/mlb-statcast-event-tracking-metrics) · [접근 및 공개 응답 예시](https://developer.sportradar.com/sportradar-updates/changelog/mlb-api-statcast-event-tracking-metrics)

## 기존 10,328쌍에 대한 계산

공식 `active_spin`은 릴리스에서 광학 측정한 3D 회전벡터로 산출한 비율의 투수·구종별 시즌 집계다. 2024년 포심 30구 이상 투수의 평균을 이름 대신 `player_id`로 연결했다. 2024/2025 리더보드 709/715행 중 634/645행이 30구 조건을 만족했다. 양력에 들어가는 유효 RPM만 바꿨고 모든 모델의 릴리스 상태를 기존 100% 조건과 동일하게 고정했다. 2024 평균이 없는 2,629쌍은 기존 100% 값을 유지했다. 2025 시즌 평균은 평가 이후 투구도 들어갈 수 있어 후향 확인으로만 계산했다. [공식 직접 측정 정의](https://baseballsavant.mlb.com/leaderboard/active-spin?year=2024_spin-based) · [2024 포심 CSV](https://baseballsavant.mlb.com/leaderboard/spin-direction-pitches?year=2024&pitch_type=FF&min=10&csv=true) · [2025 포심 CSV](https://baseballsavant.mlb.com/leaderboard/spin-direction-pitches?year=2025&pitch_type=FF&min=10&csv=true)

- 방향 비교는 기록 IVB가 같은 214쌍을 제외한 **10,114쌍**이다. RPM 부호 **59.04%**(5,971쌍), 광학 2D 방향과 100% **57.53%**(5,819쌍), 광학 방향과 2024년 평균 **57.46%**(5,811쌍), 백스핀과 2024년 평균 **58.87%**(5,954쌍), 광학 방향과 2025년 평균 **57.42%**(5,807쌍).
- 광학 2D 방향을 고정하고 100%에서 2024년 평균으로 바꾸면, 고유 투구 2,510구의 **IVB MAE 5.247→4.766 cm**, **홈 앞면 높이 MAE 4.796→4.789 cm**다. 전체 10,328쌍에서 **IVB 변화량 MAE 4.322→4.291 cm**였다. 단위가 없는 RPM 부호 기준에는 cm 오차를 부여하지 않았다.
- 2024년 평균이 실제로 있는 **7,699쌍**만 남겨도, 방향 점수는 RPM **59.61%**, 광학 방향·100% **57.55%**, 광학 방향·2024 평균 **57.45%**였다. 이 공통 분모에서 투구별 IVB MAE는 **5.093→4.447 cm**, 통과 높이 MAE는 **4.684→4.675 cm**였다.
- 같은 투수의 두 공에 동일한 양의 비율을 곱하므로 유효 RPM 대소관계는 바뀌지 않는다. 실제로 유효 RPM의 방향을 단독으로 쓰면 기존 RPM 부호와 같은 5,971쌍을 맞혔다. 물리 모델에서는 공별 릴리스 상태와 광학 방향이 달라 결과가 바뀔 수 있지만, 시즌 평균은 한 투구와 다음 투구 사이의 유효 회전 비율 변화를 알려주지 않는다.

## 제이슨 애덤의 같은 두 포심

2024년 포심 398구의 평균은 **87.54%**였다. [원 경기](https://www.mlb.com/gameday/guardians-vs-padres/2025/04/01/778485/final)

- A의 IVB는 **기록 55.47 / 100% 모델 47.94 / 시즌 평균 모델 44.47 cm**. 통과 높이는 **106.22 / 99.71 / 96.23 cm**.
- B의 IVB는 **기록 49.99 / 100% 모델 46.64 / 시즌 평균 모델 43.25 cm**. 통과 높이는 **95.14 / 91.46 / 88.06 cm**.
- A→B IVB 차이는 **기록 −5.49 / 100% 모델 −1.30 / 시즌 평균 모델 −1.22 cm**였다. 이 예시에서는 평균을 넣자 두 공 모두 IVB와 높이 오차가 커졌다. 그림의 실선은 Statcast 적합 궤적이다.

## 재현

```powershell
python -m experiments.pitcher_active_spin_flight.study savant_2025_04_01.csv savant_2025_09_28.csv savant_2024_ff_optical_efficiency.csv savant_2025_ff_optical_efficiency.csv results.json
$report = Get-Content results.json -Raw | ConvertFrom-Json
$report.april | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 overlay-input.json
python -m experiments.trackman_v3_flight.render overlay-input.json overlay.png --font path-to-korean-font.ttf
python -m unittest -v test_pitcher_active_spin_flight test_optical_axis_flight test_trackman_v3_flight test_pitch_outcomes
```

네 입력의 SHA-256은 실행 결과 JSON에 포함된다. 원자료와 그래프는 저장소 밖에 보관했다.
