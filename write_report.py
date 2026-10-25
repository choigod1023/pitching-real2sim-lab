import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import OUT

r=json.loads((OUT/'results.json').read_text())
d=pd.read_csv(OUT/'heldout.csv')
p=d[d.method=='personalized'];modified=p[(p.angle_deg!=0)|(p.delay_ms!=0)]
labels=['No change','Common screened','Common unbounded','Individual unbounded','Individual screened']
keys=['unchanged','common_screened','common','unconstrained','personalized']
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].barh(labels,[r['summary'][k]['athlete_mean_proxy_delta_m_s'] for k in keys],color=['gray','gray','#c8704c','#c8704c','#279a8c'])
axes[0].set_xlabel('Computed hand-point speed change (m/s), NOT ball speed')
axes[1].barh(labels,[r['summary'][k]['heldout_eligible_fraction']*100 for k in keys],color=['gray','gray','#c8704c','#c8704c','#279a8c'])
axes[1].set_xlabel('Pass engineering screens (%) - includes no-change trials')
fig.suptitle('Exploratory kinematic counterfactuals: 100 athletes, 311 held-out pitches\nNo dynamics, comfort, accuracy, or real training effect evaluated',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'summary.png',dpi=150);plt.close(fig)
mean=r['summary']['personalized']['athlete_mean_proxy_delta_m_s']
text=f'''# 투구 동작 변경 예비 실험 — 2026-10-25

## 먼저 읽을 결론

공개된 100명·411개 투구에서 작은 동작 변경 후보를 계산하고 GUI에서 재생했다. **아직 구속 향상, 제구 유지, 편안함, 힘으로 수행 가능한 투구를 입증한 결과가 아니다.** 이 실험은 물리적 최적화 전의 운동학적 후보 탐색이다.

제한을 적용한 선택에서 31명에게 수정이 선택됐고 69명에게 무변경이 선택됐다. 31명 모두 동일한 +2.5도/0ms 후보를 골랐다. 따라서 **개인마다 다른 최적 동작을 찾았다는 기술적 차별성은 현재 결과로 뒷받침되지 않는다.** 현재 효과는 같은 수정의 적용 여부를 나누는 수준이다.

## 실행한 내용

- 자료: OpenBiomechanics dataset-v1, 저장소 기준 44b98da. `landmarks.csv`와 메타데이터를 `session_pitch`로 연결. 원시 C3D와 처리된 landmark 시계열은 혼용하지 않았다.
- 선수 100명의 각 첫 번째 session_pitch를 선택용으로 사용하고, 남은 311개를 평가용으로 사용했다. 같은 날/같은 세션의 반복 투구이며 새로운 날이나 새 선수 일반화 검증은 아니다.
- 후보: 팔꿈치 기준 전완 방향의 국소 회전 -5/-2.5/0/+2.5/+5도 × 방향 시계열 지연 -10/-5/0/+5/+10ms = 투구당 25개, 총 10,275개.
- 변경은 기록된 release 시점을 중심으로 표준편차 100ms인 부드러운 가중치를 사용했다. +각도는 코드에서 정의한 회전이며 임상적 팔꿈치 굴곡각 처방이 아니다.
- 공 release 시점은 원자료의 BR_time으로 고정했다. 목적값은 해당 시점의 계산된 손 관절 대리점 속도 증가이며, release ±1프레임에서도 증가하는 후보를 선택했다.
- 원자료 좌표는 제공기관의 처리된 관절 중심 추정치다. 타임스탬프 반올림을 확인하고 모든 투구에서 360Hz 명목 시간축과 0.051ms 이내로 일치함을 검증한 후 미분했다.
- 제한: release 전 200ms~후 50ms 안에서 전완 방향 차이 최대 5도, 손 위치 차이 최대 5cm, 최대 손 가속도 비율 1.10 이하. 이는 임의의 **공학적 탐색 조건**이며 안전성·편안함·근력 한계가 아니다.
- 개인별 선택: 조건을 통과하고 release ±1프레임의 최소 이득이 0.05m/s를 넘는 후보 중 최대값을 선택. 없으면 무변경. 0.05m/s는 임상적으로 의미 있는 향상 기준이 아니다.
- 공통 제한 후보: 선택용 선수의 95% 이상에서 조건을 통과하는 공통 설정 중 목적값이 최대인 설정. 이번에는 무변경이 선택됐다. 개인별 제한 방식과 선택 규칙이 같지 않아 순수한 개인화 우위로 해석하면 안 된다.

## 결과

- 무변경: 계산된 손 속도 변화 약 0m/s.
- 제한 없는 공통 수정: 평균 +{r['summary']['common']['athlete_mean_proxy_delta_m_s']:.3f}m/s. 그러나 311개 평가 투구 중 조건 통과는 1개뿐.
- 제한 없는 개인별 수정: 평균 +{r['summary']['unconstrained']['athlete_mean_proxy_delta_m_s']:.3f}m/s. 조건 통과는 1개뿐.
- 제한을 둔 개인별 선택: 평균 **+{mean:.3f}m/s**. 100명 각각의 평가 투구 평균을 낸 뒤 선수 간 평균을 계산했으며 무변경 선수도 포함한다.
- 제한 개인별 선택의 전체 조건 통과: 301/311개. 무변경을 포함하므로 이 비율만 보고 성공률로 해석하면 안 된다.
- **실제로 수정한 투구만 보면 86/96개 통과, 10개 실패.** 31명 모두 +2.5도/0ms를 사용했다. 별도 투구에 적용해도 조건을 항상 유지한다는 보장은 없었다.
- 선수별 평균 손 속도 대리값이 0.05m/s 이상 증가한 비율은 31%다. **실제 선수 개선 성공 확률이 아니다.**

손 좌표를 수학적으로 수정한 결과이므로 새로운 힘이나 근육 활성 없이 속도만 바뀔 수 있다. 수치상 이득 자체는 현실적 개선의 근거가 아니다. 가속도 역시 관절 토크나 인대 부하의 대체 지표로 검증하지 않았다.

## 화면과 재현

- `05 - Pilot comparison.lnk`를 실행하면 수정 후보와 원본이 자동으로 반복 재생된다.
- [재생 GIF](comparison.gif), [요약 그래프](summary.png), [계산 결과 JSON](results.json), [전체 후보](candidates.csv), [평가용 투구 결과](heldout.csv).
- GUI 예시는 수정이 선택된 선수 중 ID가 가장 작은 선수의 첫 평가 투구다. 최고 이득 사례로 고르지는 않았지만, 전체 결과를 대표하는 무작위 사례도 아니다.
- 화면 청록색은 원본 처리 좌표, 주황색은 수정 좌표를 단순 관절 모델에 옮긴 결과다. 주황색에는 모델 재현 차이도 포함되므로 색 사이의 차이가 모두 의도한 수정량인 것은 아니다. 정량 결과는 관절 모델이 아니라 동일한 landmark 좌표끼리 비교했다.
- 실행: `python pilot.py`, `python show_pilot.py`, `python -m unittest -v`.

## 탐색 과정에서 변경한 내용

초기에는 release 앞뒤 250ms가 없는 투구를 제외해 126개만 남았다. 평가 구간과 미분에 필요한 여유에 맞춰 포함 기준을 수정했고 최종 411개를 사용했다. 이후 CSV의 시간 반올림으로 미분이 왜곡될 수 있음을 발견해 명목 시간축으로 수정했다. 모두 탐색 도중 결정했으며 사전등록 연구가 아니다. 결과 신뢰구간은 고정된 선택과 모델을 조건으로 한 선수 재표집 결과이고, 모델의 현실성이나 인과 효과의 신뢰구간이 아니다.

## 연구 방향과 다음 관문

연구 질문은 **실제 투구를 기반으로, 기존 폼과 힘 요구를 크게 늘리지 않는 개인별 수정이 실제 성능을 개선하는가**이다.

기술적 차별성 후보는 힘으로 재현 가능한 개인별 제한 최적화 및 추천 보류다. 스포츠적 차별성 후보는 제구·불편감·학습 기간·유지 효과를 함께 보는 실제 훈련 개선이다. 현재 어느 쪽도 입증하지 않았다.

다음 우선순위는 후보를 더 많이 만드는 것이 아니라 (1) 원본 투구의 힘 기반 재현, (2) 같은 몸과 힘 제한으로 +2.5도 후보를 수행 가능한지 확인, (3) 공 release 모델과 제구 지표, (4) 동일한 제한을 적용한 공통 수정과 개인화 비교다. 선수의 편안함은 후속 실험에서 직접 평가해야 한다.

## 출처

- [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics): 데이터와 좌표/이벤트 정의. 데이터 및 유도 결과 이용 시 해당 라이선스와 프로 스포츠 조직 관련 조건 적용.
- [Attias et al. 2026](https://doi.org/10.1007/s11044-026-10143-y): 예측 근골격 투구 시뮬레이션.
- [Templin et al. 2024](https://commons.nmu.edu/isbs/vol42/iss1/94/): 투구 최적제어.

이번 보고서의 수치는 로컬 Python 실행 결과이며 위 논문에서 가져온 성능 수치가 아니다.
'''
(OUT/'연구결과.md').write_text(text,encoding='utf-8')
print('Saved report, chart, and linked results:',OUT)
