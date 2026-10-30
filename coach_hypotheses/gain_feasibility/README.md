# +1.665km/h 후보의 가정·입력·동작 분리 검증

2026-10-30. 기존 검토를 실제 코드와 계산으로 이어갔다. [PROTOCOL.md](PROTOCOL.md)를 `7bb1ea0`에서 사전 고정하고 보수적인 위치·방향 판정을 `d7e8088`에서 결과 실행 전에 추가했다. 첫 계산 뒤의 누적 일 보호 실험은 별도 [SHIELD_PROTOCOL.md](SHIELD_PROTOCOL.md)를 `4c4c01b`에서 고정했다. 실패 뒤 방법을 바꾼 후속 실험임을 구분하며, 첫 실패 결과는 보존한다.

## 이번에 구현·검증한 것

- `profiles.py`: 원래 +2° 수정각을 고정하고 창20/30/40/60ms, cubic-in/smoothstep/smootherstep의12개 동작 수요를 실제 RK4로 계산한다. 이와 별도로 손목3축 입력만 blend·peak 제한한36조건을 계산한다. 같은 초기 상태와 기존 예산·자세 판정을 적용한다.
- `sensitivity.py`: 저장된 실제 기준/수정 상태에 같은 손–공 거리와 같은 시각을 적용하는 짝지은 기구학 민감도를 계산한다. quaternion-aware 보간, 입력·경계·해시 검사를 포함한다.
- `shield.py`: 실제 속도로 매 substep의 성분별 양/음의 일을 검사한다. 가상 적분이 예산을 넘으면 전체 일반화 입력을 반감해 다시 시험하고, 필요하면0입력을 적용한다. 실제 채택 상태·힘·억제 시각을 저장하고 별도 재생과 회계로 검산한다. 중력은 유지한다. 이 제어기는 선수 신경근 제어를 검증한 모델이 아니다.
- 모든 과학적 실행은 기존 일반 관성·15cm 그립·공칭 BR·3034_1을 고정했다. 레이더 속도에 맞춘 조정은 없다. 원래 muscle/whole-body scientific gate는 바꾸거나 advance하지 않았다.

## 결과: 첫48조건, 두 해상도에서96실행

기준은113.466581973km/h다. 아래는 더 세밀한 해상도의 같은 기준 대비 결과다.

- **원래30ms cubic-in:**115.131818941km/h, **+1.665236967**. 양의 성분 일 **+6.235790235J**. 자세 변화2.010759°, 공 중심 최대 이동5.220393mm, 방출 방향차0.734807°. 기존 예산은 FAIL이다.
- **30ms smoothstep:**114.212155859km/h, **+0.745573886**. 양의 성분 일 **+1.488289049J**, 예산 FAIL.
- **30ms smootherstep:**114.206615156km/h, **+0.740033182**. 양의 성분 일 **+1.176606414J**, 예산 FAIL.

뒤 두 파형은 연속식 끝점의 추가 각속도가0이다. 원래 cubic-in은+200°/s다. 따라서 원래 증가량을 끝 각도+2°만의 효과라고 설명할 수 없다. 그러나 서로 다른 전체 시간 궤적의 비교이므로 감소한 약0.92km/h를 '각속도의 인과 기여분'으로 분해하지도 않는다. smoothstep은 끝 추가 각속도가0이어도 경계의 추가 가속도가 불연속이고, smootherstep은 양끝 추가 각속도·각가속도가0이다.

손목 입력 peak를 제한한36조건에서는 성분·관절 노름 peak 위반이0개였다. 하지만 최종 예산 통과는0개다. **이 중33조건은 총 양의 일이 줄었는데도 DOF별 일 또는 어깨 순일에서 탈락**했다.

- 예를 들어20ms cubic-in, α=.25는 **+0.543318870km/h**, 총 양의 일 **−0.003793177J**, 최대 회전2.691167°였다. 그래도 손목Z 양의 일 **+0.178435840J**, 팔꿈치 음의 일 **+0.189493J**, 어깨 병진 순일 **+0.043832J** 등으로 실패했다.
- 원래30ms cubic-in, α=1의 peak 제한은 **+1.417447207km/h**, 총 양의 일 **−0.893666746J**였지만 최대 회전 **13.310168°**로 요청한 작은 동작을 유지하지 못했다. 실제 손목/전완은 입력 수요 경로의+2°와 같은 동작이 아니다.

48조건 중 자세3°·공중심20mm·방향1°를 동시에 통과한 것은12개지만, **기존 힘·일·자세·수치수렴을 모두 통과한 증가 후보는0개**다. 두 해상도 사이 증가량 차이의 최댓값은0.0000090912km/h였다. 원래 후보의 구속뿐 아니라 양/음의 일 배열·peak 배열도 원보고서와1e−12 절대 허용치로 재검산했다.

전체 결과는 [profiles_report.json](results/profiles_report.json), 검산은 [numerical_verification.json](results/numerical_verification.json)에 있다. 각 조건의 실패 목록을 보존한다. 힘 최대값만 같거나 총일이 줄었다는 사실로 동일 능력의 성공을 판정하지 않는다.

## 결과: 누적 일 보호기, 추가3조건×2해상도

첫 경로는 어깨·팔꿈치·전완 입력을 원래대로 유지했다. 이 후속 경로는 필요한 경우 **전체 일반화 입력을 줄일 수 있어** 서로 다른 질문이다. 각 성분의 양/음의 일 보호가 순일·자세까지 보장한다고 가정하지 않는다.

- **30ms cubic-in:**117.063308266km/h, **+3.596726293km/h**. 총 양의 일은 **−14.797641751J**이고 성분별 양/음의 일·peak 제한은 지켰다. 그러나 어깨 병진 순일 **+1.827558494J**, 최대 회전 **12.190325°**, 방출 방향차 **5.902159°**여서 **탈락**했다. 기준보다 흡수한 일이 줄면 양의 일이 줄어도 순일은 증가할 수 있다.
- **smoothstep:**92.557863345km/h, **−20.908718628km/h**, 최대 회전76.201483°, 공 중심 이동136.443512mm, 방향차18.836652°다. 자세와 기존 전체 예산 실패이며 수렴 기준도 미통과다.
- **smootherstep:**94.327587952km/h, **−19.138994022km/h**, 최대 회전66.857192°, 공 중심 이동123.098349mm, 방향차16.709468°다. 마찬가지로 실패다.

보호기가 갑자기 입력을 줄이면 원래 동작을 유지하는 구동까지 잃을 수 있다. 따라서 이 제어기는 성공한 선수 제어 모델로 채택하지 않는다. +3.597을 새로운 유효 증가 후보로 발표하지 않는다. 사전 고정한 이3조건에서도 통과 후보는0개다.

기준 입력에 자기 예산을 적용한 대조군은 두 해상도 모두 억제0회이고 원래 RK4 상태와 bitwise 일치했다. 채택된 힘을0부터 재생한6개 궤적도 저장 상태와 bitwise 일치했다. 상태·실제 입력만으로 독립 계산한 구속/성분별 일/peak 값이 보고서와 정확히 일치했으며 가상 실패 시도의 상태가 실제 결과에 남지 않음을 검증했다. [shield_report.json](results/shield_report.json), [shield_verification.json](results/shield_verification.json) 참조.

## 전후 차이의 민감도: +1.6이 얼마나 유지되는가

원래 +1.665 후보의 실제 적분 상태에서 그립14/15/16cm와 공칭BR/−1/−2 source frame(360Hz)을 짝지어 평가했다. **9조건의 전후 차이는+0.560859513~+1.774779621km/h**로 모두 양수였지만, +1.6이라는 크기는 모든 조건에서 유지되지 않았다. +1/+2 frame은 원래 저장 범위 뒤여서 두 조건 모두 제외하고 외삽하지 않았다.

이 계산은 같은 저장 상태의 공 중심 사이트만 옮기는 기구학 진단이다. 새 payload 관성으로 재적분한 결과, 실제 이탈 시각의 측정 오차 분포, 전신/손가락 구조 불확실성을 모두 포함한 분석이 아니다. 이 범위를95% 신뢰구간·실제 오차·성공 확률로 부르지 않는다. 실제 개입 전후의 관측이 없어 경험적 효과 구간은 여전히UNKNOWN이다. [original_sensitivity.json](results/original_sensitivity.json) 참조.

## 무엇을 해결했고 무엇이 남았는가

코드 수준에서는 수요 처방, 실제 입력 제한, 성분별 실제 일 보호, 전후 짝지은 민감도를 분리해 실행·재생 검증하는 경로를 구현했다. **최대 힘 불증가 / 총 양의 일 감소 / 작은 끝 각도** 가운데 하나만으로 성능 향상을 잘못 채택할 수 있는 구체적인 사례를 확인하고 전체 판정에서 차단했다.

과학적 목표인 같은 능력·작은 동작의 유효 증가 후보는 아직 확보하지 못했다. 이번51조건에서 찾지 못했다는 뜻이며 모든 가능한 동작의 불가능성을 증명한 것은 아니다. 마커 축·일반 관성·외부 어깨 입력이 있는 팔 모델을 실제 근육/전신 능력으로 해석하지 않는다. 손가락 접촉·회전벡터·의도 목표가 검증되지 않았으므로 RPM·구위·제구의 동시 개선 수치도 만들지 않았다.

다음 변경은 창/각도/허용량을 계속 늘리는 탐색이 아니라, **어깨 외부 입력과 전완–손 동작을 함께 유지하는 제어 구조 및 해부학적 축 대응을 검증하는 것**이다. 보호기로 입력을 끊어 자세가 무너지는 방식은 폐기하고, 원래 기준 동작을 제약 아래 재현하는지부터 확인해야 한다. 실제 회전/목표가 연결된 투구가 없으면 회전·제구의 신체 개입 예측은 미검증으로 유지한다. 실제 선수 검증은 [이전 보고서](../gain_transfer_review/REPORT.md)의 무작위 반복·실제 동작 확인·부하/일 동시 측정 설계를 따르되 별도 승인 없이 연락·모집·훈련을 실행하지 않는다.

## 재현과 검증

작업 폴더는`gain-feasibility-20261030`, Python은 상위lab의`.venv/Scripts/python.exe`다. 원본 데이터/입력은 별도기존 worktree에서 읽기만 했고 기존 결과를 덮어쓰지 않았다.

```powershell
python -m unittest coach_hypotheses.gain_feasibility.test_profiles coach_hypotheses.gain_feasibility.test_sensitivity coach_hypotheses.gain_feasibility.test_shield test_joint_intervention test_capacity_budget_gate -v
python -m coach_hypotheses.gain_feasibility.profiles --source ../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz --reference-report ../joint-intervention/artifacts/joint-intervention/report.json --out artifacts/gain-feasibility
python -m coach_hypotheses.gain_feasibility.sensitivity --baseline ../joint-intervention/artifacts/joint-intervention/state-baseline-sub2.npz --candidate ../joint-intervention/artifacts/joint-intervention/state-wrist_z_p2-sub2.npz --out coach_hypotheses/gain_feasibility/results/original_sensitivity.json
python -m coach_hypotheses.gain_feasibility.shield --source ../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz --reference-report artifacts/gain-feasibility/report.json --out artifacts/gain-shield
```

테스트 숫자와 전체 기존 오류 대비 결과는`results/final_suite_verification.json`에 기록한다. 실제 통과 후보가 없어 성공 영상은 export하지 않았다. 본 결과는 실제 구속 향상·훈련 효과·안전·편안함의 검증이 아니다. OBP 및 파생 자료는 [OpenBiomechanics/Driveline Baseball](https://github.com/drivelineresearch/openbiomechanics)과 저장소 NOTICE.md의 CC BY-NC-SA4.0 조건을 따른다. 원본 C3D·영상·원본 CSV를 저장소에 새로 올리지 않는다.
