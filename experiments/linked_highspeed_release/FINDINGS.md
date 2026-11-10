# 고속 영상에 연결된 포심의 손끝과 공 회전

## 투구 기록과 영상

[Baseball Aerodynamics 공개 예시](https://baseballaero.com/services/)의 Rapsodo 통합문서 41구 가운데 17구에 영상 하이퍼링크가 있다. 포심 9구 중 영상이 있는 공은 #4와 #8이다. 파일의 행에 포함된 투구 ID와 영상 링크로 연결했다. 두 영상의 촬영 배경과 투수가 일치해 보이지만, 기록의 선수 식별 필드는 비어 있다. 파일의 재생률은 약 60 fps이고, 장비 설명의 500 fps를 개별 원본의 촬영 시계로 확인하지는 못했다. 두 투구의 구속 차이는 2.414 km/h로 사전에 정한 1 km/h 조건을 만족하는 영상 쌍은 0쌍이다. [4번 원본](https://photos.app.goo.gl/jZE7B66ajCAFCnKW9), [8번 원본](https://photos.app.goo.gl/mD4Yd8LxF79HBdBb9)

- **#4** · 127.782 km/h, 1,893 RPM, 스핀 방향 1:20, 수직 IVB 35.306 cm, 수평 HB 16.256 cm.
- **#8** · 125.368 km/h, 1,893 RPM, 스핀 방향 0:12, 수직 IVB 30.734 cm, 수평 HB 9.398 cm.

## 원영상의 관측

- **#4 F3** · 화면 오른쪽 손끝과 공 윤곽 사이 간격은 +10.4 px이며 손끝 주석 범위와 공 윤곽 선택을 바꿔도 +3.4~+17.8 px다. 왼쪽은 −2.9 px이며 −10.4~+4.8 px로 겹침과 틈이 모두 가능하다. F4에는 양쪽에 뚜렷한 틈이 보인다.
- **#8 F4** · 화면 왼쪽 −4.9 px(−12.5~+3.1), 오른쪽 +4.8 px(−3.0~+12.9)로 둘 다 경계가 불확실하다. F5에는 양쪽에 틈이 보인다.

좌우는 화면 좌표다. 각 프레임의 공 외곽 일부를 원으로 적합하고, 손끝이 보이는 작은 영역을 표시했다. 음의 간격은 투영 윤곽 안쪽에 해당한다. 이 시야에서 물리 접촉을 판단할 깊이와 손가락 정체는 미확인이다. 공 반지름 약 32 px에 규격 반지름 36.6 mm를 적용하면 4 mm는 영상에서 약 3.5 px다. 경계 직전의 왼쪽 손끝 판독 폭 약 15~16 px보다 작으므로 4 mm 가정을 관측 3D 좌표로 대체할 수 없다. 이탈 뒤의 틈을 접촉점이나 힘의 모멘트암으로 넣지 않았다.

두 원본에서 공의 반경에 맞춰 정규화한 무늬가 반복되는 주기를 측정했다. 같은 세 가지 원형 마스크와 다섯 구간을 사용했다. 500 fps 및 주기당 한 바퀴를 가정하면 #4는 1,893.13 RPM(설정별 1,889.33~1,896.46), #8은 1,945.29 RPM(1,934.07~1,950.14)이다. 장비 차이는 +0.13, +52.29 RPM이다. 구간·마스크의 범위는 측정 신뢰구간이 아니다. 촬영 시계, 공 표면 모양의 반복과 원근에 따른 편향은 남는다.

## 비행 수치

무브먼트는 기존 `pitch_outcomes.flight`에서 양력을 켠 경우와 끈 경우의 판면 위치 차이로 계산했다. 장비의 구속·릴리스 방향·스핀 방향·유효 회전 비율과 [Sawicki 외의 양력식](https://baseball.physics.illinois.edu/AJP-Nov03.pdf)을 사용했다. [Rapsodo의 스핀 방향 정의](https://note-rapsodojp.rapsodo.com/n/n8deca7a5f587?hl=en)를 따랐다. 실제 릴리스에서 판면까지의 거리는 공개 파일에 없어 16.8 m로 두었다. 장비의 무브먼트 산출 절차와 시뮬레이션 기준이 같은지는 확인하지 못했다.

- **#4** · 장비 RPM으로 계산한 수직/수평 38.48/32.41 cm. 영상 RPM을 넣으면 38.48/32.41 cm로 장비 기록보다 각각 +3.18/+16.16 cm다.
- **#8** · 장비 RPM으로 계산한 수직/수평 50.79/5.33 cm. 영상 RPM을 넣으면 51.57/5.41 cm로 장비 기록보다 각각 +20.83/−3.98 cm다.
- **#4에서 #8을 뺀 수직 차이** · 장비 +4.57 cm, 영상 RPM을 사용한 비행 모델 −13.08 cm. 모델은 이 쌍에서 수직 변화의 방향도 재현하지 못했다. 두 구의 수직/수평 MAE는 장비 RPM에서 11.62/10.11 cm, 영상 RPM에서 12.01/10.07 cm였다.
- 영상이 없는 공을 포함한 9구에서 장비 RPM을 사용한 수직/수평 MAE는 7.94/18.02 cm였다. 두 영상 투구에서 거리·항력 민감도 조건 8개는 계산됐다. 영상이 없는 #2와 #6의 네 조건은 무양력 기준 궤적이 판면 전에 지면에 닿아 제외했다. 시간 간격을 1 ms에서 0.5 ms로 줄였을 때 9구의 가장 큰 수치 차이는 0.000093 cm였다.

이 비행 모델에는 손가락 접촉, 공 회전을 만들어 내는 힘, 실밥 방향별 공력과 바람이 들어 있지 않다. 현재 자료에서 구속과 회전수를 손동작으로 예측하거나, 관절 개입에 따른 효과를 검증할 수 없다.

## 공개 자료를 더 찾은 결과

- [Yamaguchi 외](https://doi.org/10.1038/s41598-025-93632-y)는 6명에서 고속 영상으로 검지와 공의 누적 미끄러짐을 측정했다. 영상 회전과 레이더를 비교한 보충 그림의 벡터 원본에서 표시점 119개를 다시 읽어 RMSE 136.14 RPM을 얻었다. 저자 보고는 136.2 RPM이다. 공개 파일에는 손과 공의 3D 접촉 좌표를 각 투구의 RPM·무브먼트에 결합할 원시 ID와 원본 영상이 없다.
- [Rapsodo의 릴리스 사례](https://rapsodo.com/blogs/baseball/rapsodo-insight-the-complete-pitch-design-package)는 포심 두 구에서 검지의 지연 이탈과 두 손가락의 동시 이탈을 서술했다. 당시 영상 경로에서 원본을 확보하지 못해 이탈 판독을 재현하지 못했다.
- [Driveline의 고속 비교](https://drivelinebaseball.com/blogs/blog/laminar-express-baseball-science-behind-the-two-seam-fastball)는 포심과 투심을 비교한다. 같은 구종 안의 반복 검증에는 제외했다.

## 재현

공개 통합문서 SHA-256은 `ce9bdce863f6c0569d6c08481209f04417dbbc953a298d33864e6e8333ffa420`이다. 원본 영상 두 개와 원자료 통합문서는 저장소 밖에서만 사용했다. 원본의 이름이나 영상은 커밋에 넣지 않았다.

```powershell
python -m experiments.linked_highspeed_release.seam_period --clip pitch4.mp4 --first-frame 4 --initial-circle 832 626 31 --nominal-capture-fps 500 --output seam4.json
python -m experiments.linked_highspeed_release.seam_period --clip pitch8.mp4 --first-frame 5 --initial-circle 864 621 31 --nominal-capture-fps 500 --output seam8.json
python -m experiments.linked_highspeed_release.study example-rapsodo.xlsx conditional-flight.json --optical-rpm optical-rpm.json
python -m unittest -v test_linked_highspeed_release test_seam_period test_pitch_outcomes
```

`optical-rpm.json`은 `{"4": 1893.1299902240685, "8": 1945.2889793547977}`처럼 각 회전 결과의 `quadratic_rpm_diagnostic_min_median_max[1]`을 투구 ID에 연결한 파일이다.
