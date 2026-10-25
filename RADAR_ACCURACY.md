# 야구 구속 장비의 오차와 1km/h 변화 검증

2026-10-25. 장비 사양, 반복측정 산포, 장비 간 차이, 표시 분해능은 서로 다른 값이다. 모든 프로 구장에 통용되는 단일 구속 오차 수치는 확인하지 못했다.

## Stalker

현재 [공식 야구 FAQ](https://stalkersport.com/pages/baseball-radar-gun)와 [Pro II+ 공식 설명서 p31](https://stalkerradar.com/sportsradar/documents/011-0264-00%20ProIIPlus%20operator%20manual.pdf)는 측정값의 ±3%를 명시한다. 140km/h에서는 ±4.2km/h다. 제조사 사양을 실제 매번 발생하는 오차 또는 반복측정 표준편차/95% 신뢰구간으로 해석하지 않는다. 0.1mph 표시 단위와 정확도는 별개다. 구형 자료나 판매처의 ±0.1mph 표현을 모델/판본 확인 없이 모든 Stalker의 최신 정확도로 인용하지 않는다.

## TrackMan

[Nathan 등2014 원문](https://www.baseballprospectus.com/news/article/23202/testing-trackman-just-how-well-does-trackman-work/)과 [저자 설명](https://baseball.physics.illinois.edu/trackman.html)의 구장 실험은 발사기로 홈플레이트에서 띄운 공을 고속영상과 비교했다. 초기 속력 차이의 편향 제거 후 RMS 산포0.5mph는0.805km/h다. 타구를 모사한 발사기와 당시 장비 조건의 결과이며 현세대 투구 측정의 보장 오차가 아니다. 편향 제거 전 전체 오차나 같은 장비의 반복오차와도 다르다.

## MLB Hawk-Eye/Statcast

[Sony2020 공식 발표](https://sony.mediaroom.com/2020-08-20-Hawk-Eye-Innovations-and-MLB-Introduce-Next-Gen-Baseball-Tracking-and-Analytics-Platform)는 위치 ±0.1inch, 즉 ±2.54mm를 제시한다. 구속 ±몇km/h의 뜻은 아니다. 이번 조사에서는 현행 시스템 구속의 독립 검증 오차 수치를 확보하지 못했다. [Statcast 정의](https://baseballsavant.mlb.com/csv-docs)의 release_speed는 손에서 나온 시점의 속력이다. 전광판 표시값, 비행 구간 평균속력과 바로 비교하지 않는다.

## Rapsodo와 레이더

[Morimoto 등2026](https://www.jstage.jst.go.jp/article/sposun/36/3/36_3_263/_article/-char/en)은 대학투수27명 각5구를 두 장비로 동시 측정해 구속 ICC0.995를 보고했다. 초록의 −0.3~0.0m/s 범위는 PDF 본문을 확보하지 못해 평균차의 신뢰구간인지 개별 차이의 한계인지 확인하지 못했다. 따라서 개별 측정 오차 범위로 인용하지 않는다. ICC가 높아도1km/h 변화 판별 정확도가 자동 입증되지는 않는다.

## 우리 연구의 평가 기준

같은 장비, 같은 설치 위치, 같은 속도 정의로 전후 투구를 반복 측정한다. 기기 자체 오차와 실제 투구마다 달라지는 구속을 구분해야 한다. 선수 내 변화량 오차, 반복측정 산포, 편향 및 변화량의 불확실성을 함께 보고한다. 반복 평균으로 무작위 오차를 줄일 수 있지만 고정 편향과 모델의 구조적 오류는 없어지지 않는다.

레이더의 단순 방사방향 측정 v*cos(theta) 가정에서는 실제140km/h를 정면에서5도 벗어나 측정하면0.533km/h,10도면2.127km/h 낮게 나타난다(Python 계산). 이는 각도 보정을 하지 않는 단순 모델의 예이며 현대3D추적 시스템 전체의 오차로 적용하지 않는다. 작은 전후 차이를 비교할 때 설치와 정의를 고정해야 하는 이유다.

우리 영상의 임시 카메라 보정과 밝기 중심 추출에서 나오는 오차를 레이더 사양으로 정당화하지 않는다. 최종 목표는 손가락/관절 변경에 따른 실제 구속 변화 예측이며, 공 좌표 추적 안정화는 그 선행 단계다.
