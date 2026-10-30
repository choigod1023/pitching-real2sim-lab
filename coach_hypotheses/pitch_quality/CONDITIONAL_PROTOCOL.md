# 반복 경기의 투수 내 구위 지표–헛스윙 관측 분석: 사전 기준

새 반복 표본의 결과 관계를 계산하기 전에 이 문서를 커밋한다. 기존 Real2Sim gate와 합격 기준은 변경하지 않는다.

## 자료와 정의

- MLB 공식 StatsAPI의 2025-06-01~2025-06-14(양끝 포함) 일별 schedule에서 officialDate가 범위 안이고 R/Final인 모든 경기를 gamePk로 중복 제거한다. 실제 schedule의 feed link만 사용한다. 선수·구속·성적으로 경기를 고르지 않는다. 기존 777677 한 경기는 이미 본 개발 자료다. 이번 2주 표본 전체도 최종 외부 검증이 아니다.
- 공개 JSON만 보통 GET으로 수집한다. 최대 동시 3개, 실패 시 최대 2회 추가 시도, 원본 URL/UTC 시각/바이트/해시/실패를 보존한다. 원본·행별 데이터는 ignored artifacts에만 저장한다. 원자료 재배포 라이선스를 가정하지 않는다.
- 분모: 공식 pitchCodes의 swingStatus=true 및 buntAttemptStatus=false. 주 결과는 연구용 엄격 S/W 미스(해당 callCode면 1)다. 공식 비번트 swingMissStatus(S/W/T/Q)는 별도 민감도 결과다. 파울팁을 임의로 공식 헛스윙에서 빼었다고 표현하지 않는다.
- startSpeed는 MLB 공개 gamefeed 배포 JS의 mph 표시를 확인했다. 정확한 평가 위치/시점은 미확인이다. spinRate, breakVerticalInduced, breakHorizontal, pX/pZ의 StatsAPI 경로별 단위·방향·시점은 미확인으로 유지한다. Savant CSV와 임의 치환하지 않는다.
- 단위가 미확인인 필드도 훈련 표본에서만 중심화·표준화한 무차원 관측 분석에는 쓸 수 있다. 결과는 원필드 1 훈련 내 표준편차의 조건부 관계로만 해석한다. 물리 모델·회전 효율·실제 인치 또는 관절 처방으로 번역하지 않는다. 기존 conditional_study_ready를 소급 통과시키지 않고 이 좁은 분석 가능 여부를 별도로 기록한다.
- 기존 parse_feed의 ID·카운트·공식 lookup 검사를 재사용한다. 불명확 ID/카운트, 미확인 call, 없는 필수 수치는 제외하고 사유와 개수를 보존한다. 결측을 0으로 대체하지 않는다.

## 고정 분석

1. 주분석 구종은 FF(공식 구종 코드)다. 다른 구종은 같은 절차의 보조 분석으로 따로 보고하며 성능 좋은 구종으로 주분석을 교체하지 않는다.
2. 경기 gamePk % 5를 고정 5분할로 쓴다. 같은 경기는 학습과 검사에 섞이지 않는다. 목표는 이미 관측한 투수의 새 경기이며, 같은 투수가 양쪽에 있는 설계다. 선수 제외 검증으로 부르지 않는다. 각 fold 검사에는 그 fold의 학습 스윙 20개 이상·학습 경기 2개 이상인 투수×구종만 포함한다. 이 적격 판정은 결과 y를 쓰지 않는다. 신규/희소 투수 검사 수는 따로 기록한다.
3. 모든 비교 모델은 동일한 complete-case 행을 쓴다. 필수 원필드: startSpeed, pX, pZ, spinRate, breakVerticalInduced, breakHorizontal. pre-count는 0~3 balls/0~2 strikes, batSide/pitchHand는 L/R여야 한다.
4. 로지스틱 모델: 투수×구종 절편, 전체 절편, 학습 표본에서만 투수 내 중심화한 연속 지표. 학습에 없는 투수의 중심은 학습 전체 평균이지만 해당 검사는 주 지표에서 제외한다. 표준편차는 학습 내 편차 전체에서만 계산한다. 보조 구종도 따로 적합한다.
5. 공통 통제는 startSpeed, pX, pZ, pX², pZ², pX*pZ, 이 6개와 타자 좌우의 상호작용, 타자 좌우, 투구 전 12종 카운트(기준 0-0 제외)다. 네 모델은 control, +spinRate, +movement(breakVerticalInduced/breakHorizontal 및 각 필드와 투수 좌우·타자 좌우의 상호작용), +both다. 표준화 및 회귀는 전부 fold 훈련 안에서 수행한다.
6. 고정 L2 페널티: 절편 0, 투수 절편 1, 나머지 계수 1. 합계 음의 로그우도+0.5*penalty*b²를 최소화한다. 튜닝/결과 기반 변수 선택은 하지 않는다. 정규화한 투수 효과이므로 완전 비모수 fixed-effect 또는 인과 추정이라고 부르지 않는다. 수렴 실패는 보고하고 성공으로 집계하지 않는다.
7. 검사 Brier와 log loss를 투수별 평균 후 투수에 동일 가중한다. 단순 예측은 학습 전체 평균 및 학습 투수×구종 비율에 Laplace(1,1)를 적용한 값이다. 주 비교는 +both 대 control이고 추가로 투수 평균 기준과 비교한다. 정확도/단일 상관계수로 대체하지 않는다.
8. 사전 합격: FF의 적격 out-of-game 스윙 ≥2,000 및 투수 ≥50, 모든 5 fold 수렴, +both의 Brier 개선(control−both)의 투수 cluster bootstrap 2,000회(seed 20261030) 95% 백분위 구간 하한>0, +both의 log loss<control, +both의 Brier<투수평균 기준을 모두 만족해야 한다. 하나라도 안 되면 미통과다. bootstrap은 경기/시간 의존을 완전히 제거하는 확률 보장이 아니다. 다른 구종/타깃/개별 계수의 유리한 결과로 교체하지 않는다.
9. 투수 사이 FF 관계는 투수별 평균 spin/movement와 미스 비율의 Pearson 상관으로 따로 기술한다(스윙 ≥20·경기 ≥2). 투수 내 단순 관계는 같은 투수 평균을 뺀 변수·y의 상관으로 따로 보고한다. 조건부 예측과 혼동하지 않는다. 전체 자료 재적합 계수와 fold 부호도 탐색적으로만 보고하고 다중 비교 유의성을 주장하지 않는다.
10. 보조 구종은 전체 비번트 스윙 ≥1,000·투수 ≥20일 때 동일 모델을 실행한다. 공식 swingMissStatus 민감도는 같은 주/보조 행과 분할에서 수행한다. 관측된 2주/스윙에 조건부인 결과이며 타자 능력·배합·이전 공·경기 상태의 잔여 혼재가 있다.

## 산출물·중단 범위

수집 manifest/게임별 해시·제외, split audit, 모델/기준 수치, 사전 판정, 검산·테스트와 한국어 보고서를 저장한다. 원본 단위를 단정하지 않는다. 신체→회전/구위 연결이나 관절 수정은 이번에 구현하지 않는다. 관측 연관성이 유리해도 훈련 효과·안전·편안함 또는 실제 성능 향상을 입증하지 않는다. 실제 새 신체 개입이 없으므로 향상된 투구 영상도 만들지 않는다.

공식 근거: [pitchCodes](https://statsapi.mlb.com/api/v1/pitchCodes), [metrics](https://statsapi.mlb.com/api/v1/metrics), [Savant CSV 문서](https://baseballsavant.mlb.com/csv-docs), [MLB 파울팁 설명](https://www.mlb.com/news/steven-kwan-s-streak-without-swing-and-miss-ends), [공개 gamefeed JS](https://builds.mlbstatic.com/baseballsavant.mlb.com/v1/sections/gamefeed/builds/4a2930fb3794eed37b9aabce1f4ff3d0966f246b/scripts/build/gamefeed-all.js).
