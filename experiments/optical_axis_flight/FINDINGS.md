# Hawk-Eye 광학 회전 방향을 사용한 포심 비행 비교

## 자료와 판정

공식 Savant의 2025-04-01과 2025-09-28 정규시즌 포심을 사용했다. 같은 경기·투수 내 구속 차이 1.0 km/h 이하 조합은 각각 4,242쌍과 6,086쌍이다. IVB 기록 동률 93쌍과 121쌍을 방향 평가에서만 제외했다. 한 투구가 여러 쌍에 들어갈 수 있으므로 방향 일치율은 쌍 수를 분모로 한 기술통계다. [4월 1일 원자료](https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=pitcher&game_date_gt=2025-04-01&game_date_lt=2025-04-01&hfGT=R%7C&hfPT=FF%7C) · [9월 28일 원자료](https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&player_type=pitcher&game_date_gt=2025-09-28&game_date_lt=2025-09-28&hfGT=R%7C&hfPT=FF%7C)

공개 `spin_axis`는 Hawk-Eye 회전벡터의 X-Z 평면 투영각이다. 투구별 전체 3D 벡터나 유효 회전 비율은 공개 CSV에 없다. 공개 회전 방향과 `pfx_x/pfx_z`로 계산한 무브먼트 방향의 절대 각 차이 중앙값은 4월 10.43°, 9월 7.93°였다. 두 방향을 동일 값으로 취급하지 않았다. [MLB 광학 회전 설명](https://www.mlb.com/news/new-statcast-tool-measures-pitch-spin-direction) · [Nathan의 공개 항목 설명](https://baseball.physics.illinois.edu/HawkeyeAveSpinComponents.pdf) · [공식 CSV 정의](https://baseballsavant.mlb.com/csv-docs)

### 쌍별 수직 무브먼트 변화 방향

- **4월 1일**, 동률 제외 4,149쌍. RPM 변화 2,478쌍(59.73%), 백스핀 비행 2,477쌍(59.70%), 광학 방향 비행 2,384쌍(57.46%).
- **9월 28일**, 동률 제외 5,965쌍. RPM 변화 3,493쌍(58.56%), 백스핀 비행 3,483쌍(58.39%), 광학 방향 비행 3,435쌍(57.59%).
- **합계**, 동률 제외 10,114쌍. RPM 변화 5,971쌍(59.04%), 백스핀 비행 5,960쌍(58.93%), 광학 방향 비행 5,819쌍(57.53%). 광학 방향 조건은 RPM 기준보다 152쌍 적게 맞혔다. 홈 앞면 전에 지면을 통과한 궤적이 들어간 한 쌍을 빼도 결론은 같았다.

### cm 단위 오차

백스핀 고정 모델과 광학 방향 모델은 같은 초기 상태와 공기력 설정을 썼다. RPM 부호만 보는 기준은 높이나 cm 단위 변화량을 계산하지 않는다.

- 쌍에 들어간 고유 투구 2,510구에서 투구별 IVB MAE **12.71→5.25 cm**, 홈플레이트 앞면 통과 높이 MAE **10.30→4.80 cm**.
- 전체 10,328쌍에서 쌍별 IVB 차이 MAE **3.96→4.32 cm**.
- 유효 회전 비율을 두 조건에서 80%로 고정해도 방향 점수는 백스핀 58.77%, 광학 방향 57.49%였다. 이 조건의 투구별 높이 MAE는 합계 6.34→5.92 cm였지만, 4월만 보면 6.05→6.53 cm였다. 100% 조건의 절대 오차 개선 폭은 유효 회전 비율 가정에 따라 달라진다.
- IVB를 중력 보정 접선 대신 무양력 비행 대비 높이로 계산했을 때도 광학 방향은 4월 57.63%, 9월 57.79%로 각 날짜 RPM 기준보다 낮았다. 거리 구간 수를 256에서 512로 늘렸을 때 명목 모델 IVB 차이의 최대는 4×10⁻¹² cm 미만이었다. [무양력 기준 방법](https://tht.fangraphs.com/pitch-movement-spin-efficiency-and-all-that/)

### 제이슨 애덤의 예시 두 구

사전에 정한 규칙으로 4월 표본의 첫 번째 조건 충족 연속 포심을 골랐다. 2025-04-01 클리블랜드–샌디에이고 경기 778485, A는 타석 54의 2구, B는 타석 55의 1구다. 공식 경기 피드에서 투수와 두 투구의 구속·RPM 및 판면 높이를 확인했다. [경기 기록](https://www.mlb.com/gameday/guardians-vs-padres/2025/04/01/778485/final)

- A는 **152.89 km/h, 2,651 RPM**. IVB 기록 **55.47 cm**, 백스핀 **56.08 cm**, 광학 방향 **47.94 cm**. 통과 높이는 각각 **106.22 cm**, **107.81 cm**, **99.71 cm**.
- B는 **153.21 km/h, 2,691 RPM**. IVB 기록 **49.99 cm**, 백스핀 **56.38 cm**, 광학 방향 **46.64 cm**. 통과 높이는 각각 **95.14 cm**, **101.17 cm**, **91.46 cm**.
- 구속 차이는 0.32 km/h다. A→B의 IVB 차이는 기록 **−5.49 cm**, 백스핀 **+0.30 cm**, 광학 방향 **−1.30 cm**였다. 이 예시에서 광학 모델은 감소 방향을 맞혔고 감소량을 **4.19 cm** 작게 계산했다.

실선은 공개된 50 ft 속도·가속도와 릴리스 위치로 복원한 Statcast 적합 궤적이다. 복원 궤적의 판면 높이와 기록 차이의 MAE는 4월 0.077 cm, 9월 0.074 cm였다. 같은 장비 출력의 내부 검산이다. 복원 궤적에서 계산한 IVB와 공개 `pfx_z`의 MAE는 각각 0.54 cm, 0.53 cm였다. 시뮬레이션에는 공개 가속도·판면 좌표·IVB를 넣지 않았다. 두 모델 모두 50 ft 속도 방향을 공통 물리로 릴리스까지 짧게 역적분했다. 손가락 접촉·관절·공력 중 실밥 방향에 따른 추가 힘은 모델에 없다.

## 재현

CSV SHA-256은 4월 `2254ebde4c0c3b85a208c1cac1394458c7daf3368e9bb235a797a7de8e469436`, 9월 `9090db0bf686f90ae0a0df2e6bf7511dd672c9d338287a40049a05086aee6370`이다. 원자료와 결과 그림은 저장소 밖에 두었다.

```powershell
python -m experiments.optical_axis_flight.study savant_2025_04_01.csv savant_2025_09_28.csv results.json
$report = Get-Content results.json -Raw | ConvertFrom-Json
$report.primary | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 overlay-input.json
python -m experiments.trackman_v3_flight.render overlay-input.json overlay.png --font path-to-korean-font.ttf
python -m unittest -v test_optical_axis_flight test_trackman_v3_flight test_pitch_outcomes
```
