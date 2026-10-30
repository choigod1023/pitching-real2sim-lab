# 투수–공–타자 연결 및 투구 영상 사전 설계

2026-10-30. 사용자가 타자까지 포함하도록 프로그램 수정과 성공 결과마다 투구 장면 영상을 요청했다. PR129 결과와 기존 Real2Sim gate는 유지한다. 구현/결과 계산 전 고정한다. 연구용이며 훈련·의료 조언이 아니다.

## 두 연결을 분리한다

1. 기존 MLB GB1의 저장된 학습 변환/계수를 복원해 같은 원관측 행의 P(swing), strict S/W 및 공식 미스 조건부 확률을 제공한다. 동일 gamePk/atBatIndex/pitchIndex 교집합에서 기존 예측과 재현 차이를 검산한다. P(take),P(swing and labelled miss),P(swing and other)를 합계1로 결합한다. 미스 정의가 물리 접촉과 같지 않으므로 other를 contact로 바꾸지 않는다. 실제 타자 배트 궤적을 이 통계 모델이 예측한다고 주장하지 않는다.
2. 처리된 실측 투수 자세와 별도 실측 타자 손/스윗스폿 궤적을 동일 장면에 배치해 기하 근접을 계산한다. 같은 실제 대결이 아니며 실제 헛스윙·타구 결과 검증도 아니다. MLB 확률을 OBP 영상에 대입하지 않는다.

## 자료와 고정 선택

- 투수는 기존에 분석한 OBP 선수830, 투구1615_1을 쓴다. processed 20Hz 좌표·제공 FP10/BR·기록 구속을 재사용하고 손 중심을 시각화상 방출 원점의 대리로 둔다. 실제 공 중심이나 방출 속도벡터 관측이라고 하지 않는다. 원 동작은 바꾸지 않는다.
- 타자는 OBP metadata의 우타·양수 bat_length 기록을 user/session_swing 숫자 순으로 고른 첫 사례6_1이다. 성능·타이밍 결과로 고르지 않는다. release dataset-v1의 hitting_landmarks.zip을 확보했으며 GitHub asset SHA256 9cc2bdc94f5f5a6c5d540f4abe459caae2c5164cd53f6a6fd23f4b2b7283bcb3과 일치했다. 원본·C3D·CSV는 ignored artifacts에만 둔다.
- landmarks 헤더의 blast_hand/sweet_spot와 contact_time을 사용한다. 실제 bat 끝점·반경이라고 바꾸지 않는다. hitting C3D의 단위/시간과 문서의 좌표 정의를 대조한 뒤 사용한다. hitting은+x투수쪽,+y우타박스,+z위, pitching은+x홈,+y1루,+z위다. 투수는 수평180도 proper rotation후 고정 병진하며, 좌우 반사나 매프레임 정렬을 하지 않는다. 투수 배치는18.44m 구장 길이를 쓰는 장면 가정이다.
- 새 공은 기록 구속을 초기 속력 가정으로 넣은 중력 포물선이다. 손 중심 대리에서 타자 제공 contact_time의 sweet_spot을 지나도록 출발 방향/도착시각을 기하적으로 구성한다. 따라서 기준 시각의 근접은 설계상 정렬이며 실제 관측 재현 성공이 아니다. 공기저항·스핀/실밥·접촉 반작용·사후 타구 속도/각도는 계산하지 않는다.

## 기하 실험

- sweet_spot 중심에서 손→sweet_spot 축의 ±50mm 구간을 평가하는 capsule을 쓴다. 공 반경36.5mm/배트 반경33mm는 일반적 크기의 공학적 가정이고 개인 실측값이 아니다. 배트 전체의 접촉/헛스윙을 뜻하지 않는 “스윗스폿 구간 근접”이다.
- 타자 궤적은 원자료를 그대로 시간 이동한다. 기준0ms 및+12ms를 영상에 항상 함께 표시한다. 추가 진단은−20~+20ms의1ms 간격이다. 유리한 offset을 골라 선수 개선이라고 부르지 않는다.
- 시간 간격0.2/0.1/0.05ms를 비교한다. 주 계산은0.1ms. 반경/좌표 불확실성은 각각 bat radius28/33/38mm, 공 높이−10/0/+10mm의 사전9조건으로 검사한다. 확률/95%측정오차가 아닌 공학적 민감도다. 처리 마커360Hz보다 작은 적분 간격이 실제 관측 해상도를 늘리지 않는다.
- analytic 정지 segment/이동 sphere 사례, 끝점/길이0, proper rotation/translation 불변, 시간 이동 부호, 외삽·비유한 값 거부, 판정 범위/단위와 저장 모델의 예측 재현을 테스트한다. 기하·수치 테스트 성공을 실제 선수 효과로 승격하지 않는다.

## 영상과 성공 표시

항상 투수 장면을 포함하고 타자/배트/공을 함께 보여준다. 원/비교 동작은 같은 시간·좌표에 겹치며 투수 수정량0, 타자 시간 이동·각도/위치 차이, km/h 초기속력 가정과 힘/일 미평가를 표시한다. 그래프 영상으로 대체하지 않는다. 합성 대결·서로 다른 투구/스윙·기록값 초기화와 미검증 범위를 화면에서 알 수 있게 한다. 전체 디코드와 대표 프레임을 검사한다.

성공한 연구 결과를 반환하는 신규 경로는 검증한 투구 영상 artifact를 필수로 요구한다. prediction/engineering/athlete_candidate를 구분하며, 선수 개선 후보에는 같은 투구 baseline 검증·동일 힘/일·미사용 자료·신체 변경 근거가 필요하다. 현재는 athlete_candidate 성공을 허용하지 않는다. 성공한 통계 예측의 영상은 재생/설명 자료일 수 있으며 실제 신체 향상 영상으로 표시하지 않는다. 앞으로도 성공을 보고할 때 채팅에 로컬 MP4를 직접 띄운다.

자료: [OBP hitting 문서](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_hitting), [자료 라이선스](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md), [서로 다른 투구/스윙 궤적을 결합한2025연구](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1557145/full). 마지막 논문의 실험 결과를 재현한다고 주장하지 않는다. 파생 관절/영상은 CC BY-NC-SA4.0 및 OBP 인용을 유지한다.
