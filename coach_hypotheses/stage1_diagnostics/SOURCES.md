# 진단 방법의 출처와 확인 수준

- OBP [DATASHEET](https://github.com/drivelineresearch/openbiomechanics/blob/main/DATASHEET.md),
  [축/신호 정의](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching).
  full_sig의360Hz와 이미 적용된20Hz 처리, 반복 관측 구조를 확인했습니다.
  이탈 시각/레이더 반복 오차의 진실값이 모두 제공된다는 뜻은 아닙니다.
- Nebel 등, [10.1016/j.smhs.2024.10.001](https://www.sciopen.com/article/10.1016/j.smhs.2024.10.001?issn=2097-6291).
  출판사 저자 초록에서82명 NCAA D1의 fastest/slowest 관측 파형과 FP~BR의
  구간별 차이를 확인했습니다. 현재 PDF 요청은 접근 실패로 **전문 확인 필요**입니다.
  이것은20개 시점 지표가 개인 내 예측을 개선한다는 보장이 아니라 시험할 표현의 근거입니다.
- [Scientific Reports2025, GNN-GRU](https://pmc.ncbi.nlm.nih.gov/articles/PMC11882905/),
  DOI10.1038/s41598-025-88284-x. 공식 EuropePMC fullTextXML을 내려받아 Methods를
  확인했습니다. 53명208샘플의80/10/10 분할을 설명하며 선수 분리 여부를 명시적으로
  확인하지 못했습니다. 종속변수는 실제 연속 공 속도가 아니라 손의3D합성 속도입니다.
  따라서 보고된 성능을 이번 선수 내 실측 구속 예측과 직접 비교하지 않습니다.
  신경망으로 바꾸면 해결된다는 근거로 사용하지 않습니다.

기존 단계1의 원문 검토와 부호 감사도 재사용합니다. 추가 알고리즘은 선수 단위
중첩 검증, 정보 표현 비교, 손실 함수 대조이며 코칭 또는 의료 권고가 아닙니다.
