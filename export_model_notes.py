"""Generate Markdown/Obsidian notes; publish only into a new dedicated folder."""
import argparse
import json
import shutil
from pathlib import Path
from same_pitch_audit import OUT, ROOT


def build_notes(output=ROOT/'obsidian/야구 Real2Sim', results=OUT/'summary.json'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    s=json.loads(Path(results).read_text(encoding='utf-8'));r=next(x for x in s['records'] if x['pitch']=='3034_3')
    b=r['models']['nominal']['resolutions']['1'];a=r['models']['frozen_skeleton']['resolutions']['1'];ag=s['aggregate']
    common='\n\n[[00 야구 연구 지도]] · [[01 모델별 역할]] · [[02 오차와 증거 판단]] · [[03 동일 투구 실험 결과]] · [[04 다음 검증 순서]] · [[05 자료와 코드]] · [[06 투구 영상]]\n'
    notes={
    '00 야구 연구 지도':f'''# 야구 Real2Sim 연구 지도

갱신: 2026-10-28. 목적은 작은 신체 조정이 구속·구위·제구를 개선하는지, 부담과 일의 증가 없이 검증하는 것이다. **현재 실제 선수의 성능 향상은 입증하지 못했다.**

먼저 [[01 모델별 역할]]을 읽고, 숫자가 가까운 모델과 신체 개입 효과를 예측하는 모델을 구분한다. [[03 동일 투구 실험 결과]]에는 같은 투구로 다시 계산한 결과가 있다.

> [!important] 이번에 해결한 것과 못한 것
> 다른 선수/모델의 오차를 보정 전후로 섞는 비교를 코드에서 차단했다. 3034_3의 기록 차이는 {b['record_gap']['value']:.3f}→{a['record_gap']['value']:.3f}km/h로 일부 감소했다. 그러나 투구 간 변화 예측과 손목 자세가 악화해 새 성능 모델로 채택하지 않았다. 17km/h의 차이를 해결했다고 말할 수 없다.

```mermaid
flowchart LR
  A[같은 투구 식별] --> B[관측 신체와 손]
  B --> C[관절·물리 모델]
  C --> D[손에서 공으로 전달]
  D --> E[이탈과 공 비행]
  E --> F[구속·구위·제구]
  G[같은 투구 공 관측] -. 검증에 필요 .-> D
  G -. 검증에 필요 .-> E
```

현재 3034 자료에서는 G가 확인되지 않았다. 손 마커를 G로 대체하면 안 된다. 숫자가 맞는다는 이유만으로 몸→공 연결이 검증되는 것도 아니다.
''',
    '01 모델별 역할':'''# 모델별 역할

모델 이름이나 영상의 정교함보다 **무엇을 입력해 무엇을 예측했는지**를 먼저 본다.

## M0 실측 자료

OBP 3034에는 관절/손 마커, 처리 신호, 구속 스칼라와 힘판 자료가 있다. 마커 위치도 장비/처리 오차를 가진다. RFIN, hand_jc, 손목, 공 중심은 서로 다르다. 같은 투구의 직접 공 궤적은 확인되지 않았다.

## M1 관측 동작 재생·운동학

입력은 측정/추정한 관절 좌표, 출력은 자세와 미분 속도다. 관찰 차이와 시간 관계를 찾는 도구다. 영상이 실측 자세를 잘 따라가도 힘으로 그 동작을 만들었는지, 바꾼 동작의 공 속도를 예측하는지는 증명하지 않는다.

## M2 팔–손–강체 공 물리 모델

MuJoCo, 상완/전완/손과 145g 공을 사용한다. 손목 기준 15cm에 공이 고정됐다는 가정, 일반 질량/관성, 마커 기반 손 회전이 들어간다. 자유 어깨 경계에서 힘 기반 적분을 검사할 수 있다. 손가락 접촉이나 실제 공 이탈의 독립 예측은 미통과다. 손의 마커 회전각은 해부학적 손목각과 같지 않다.

## M3 전신 결합 모델

M2에 골반/몸통/다리/글러브 팔을 결합한다. 입력 타이밍을 바꾸면 전신이 반응하도록 적분할 수 있다. 하지만 발 접촉을 구현한 검증 모델이 아니며 골반 외부 힘/토크가 지지를 대신한다. 견갑을 독립적인 검증된 해부학 모델로 다루지도 않는다. 현재 구속 차이가 커서 실제 최적 동작 예측기로 통과하지 못했다.

이번 실험의 M3-N은 투구별 명목 기하, M3-A는 첫 투구로 어깨 부착점만 동결, M3-F는 첫 투구의 골격/관성/부착점을 동결한 버전이다. 세 버전의 질량과 15cm 그립은 바꾸지 않았다. M3-F의 분절 길이와 그에 따른 관성은 첫 투구 기준이며 능력을 향상시킨 조건이 아니다.

## M4 OpenSim 근육 모델

별도 경로의 47근육 모델에서 길이·모멘트암·힘 배분 잔차를 검사했다. 팔–전신 MuJoCo 결과가 자동으로 이 근육 제약을 통과한 것은 아니다. 기존 기록의 평균 잔차 약137.610N·m가 남았고 경로 일관성도 전체 투구에서 통과하지 못했다. 근육을 그렸다는 이유로 실제 근력 범위라고 판단하면 안 된다.

## M5 손가락 접촉·마찰 실험

접촉 유지/해제와 힘·일을 조사하는 별도 단순화 모델들이다. 실제 선수의 동기화 접촉 자료로 검증된 모델이 아니다. 이번 M3 실험에는 적용하지 않았다. 공 회전과 구위를 이 모델들로 검증했다고 주장하지 않는다.

## M6 관측 공으로 초기화한 자유 공 A/B/C

별도 Prime BaseballThrow_001 투구에서 공 관측으로 초기 위치/속도를 적합해 짧은 미래 비행을 검사했다. A는 고정 손–공 관계, B는 선형 상대운동, C는 공 좌표만 사용한다. 작은 오차는 공 관측의 도움을 받은 결과다. 신체 각도 변경이 그 초기 속도를 만들어 낸다는 예측은 아니다. OBP 3034의 오차가 줄었다고 이전하면 안 된다.

## M7 가상 이탈 조건과 공력

별도 PR87의 가정 속도/회전/회전축/이탈 위치를 넣은 시나리오다. 초기 조건이 실제로 가능하다는 전신/근육 검증과 구분한다. 에너지 고정 시나리오도 선수의 힘·일 제약 검증과 다르다. 이번 계산에는 사용하지 않았다.

## M8 통계적 구속 모델

손 속도·신체 크기 등의 특징으로 ridge 회귀를 수행하는 release_baseline.py 경로다. 선수 단위 분리에서 예측력을 평가할 수 있다. 관측 상관관계가 개입 효과를 보장하지 않으므로 회귀식 숫자를 최대화한 자세를 훈련법으로 추천하면 안 된다.

## M9 타이밍 후보 탐색

M3의 힘 입력 시각을 바꾸는 PR88 실험이다. 기준/후보의 실제 적분 속도에서 일을 계산했다. 자세 제한 내 +0.039137km/h였지만 전신/팔의 일이 늘어 채택 후보는 0개다. M9가 돌았다는 사실이 M3의 정확도나 실제 선수 향상을 검증하지 않는다.
''',
    '02 오차와 증거 판단':'''# 오차와 증거 판단

## 1. 비교할 숫자의 신분부터 확인한다

데이터셋, 선수/투구 ID, 예측 대상, 시점/구간, 단위, 오차의 종류를 함께 적는다. 같은 이름의 km/h라도 하나가 다른 선수의 공 외삽 MAE이고 다른 하나가 전신 가정 공의 기록 차이라면 보정 전후가 아니다.

same_pitch_audit.require_same_evaluation은 dataset/pitch/target/event_or_window/metric/units가 다르면 비교를 거부한다. 관측이 없으면 null로 남기고 0오차로 바꾸지 않는다. 이 검사는 비교 혼동을 막으며 모델 자체를 정확하게 만들어 주지는 않는다.

## 2. 적분 오차

모델의 참고 궤적과 적분 궤적, 또는 시간 간격을 줄였을 때의 차이다. 아주 작아도 잘못된 기하·그립·경계 조건을 정확히 계산했을 수 있다. 이것을 실제 선수 구속 오차라고 쓰면 안 된다.

## 3. 관절 재현 오차

실측/추정 신체 좌표와 시뮬레이션 위치·방향의 차이다. 위치 RMSE는 mm, 방향은 도로 보고하며 어느 관절과 구간인지 적는다. 작은 위치 오차와 작은 속도 오차는 서로 자동으로 보장되지 않는다. 처리 좌표와 피부 마커도 서로 다른 정의를 가질 수 있다.

## 4. 기록 구속과의 차이

현재 3034의 레이더 측정 위치/시점과 실제 BR 정의가 미확정이다. 모델 BR 속력과 기록 스칼라를 비교한 값은 기록 차이라고 부른다. 같은 순간의 실제 이탈 오차, 장비 정확도, ±신뢰구간으로 바꾸지 않는다.

## 5. 투구 간 변화 예측

같은 선수의 빠름−느림 차이를 예측해야 신체 수정 연구로 이어질 수 있다. 절대값이 가까워졌어도 변화 예측이 나빠지면 채택하지 않는다. 한 유리한 쌍뿐 아니라 모든 지정 투구와 무변화 대조군을 본다. 이미 본 자료 재사용은 독립 검증이 아니다.

## 6. 실제 선수 성능 향상

모델 안의 속도 증가, 자유 공 초기 속도 보정, 관찰 상관관계와 구분한다. 선수에게 개입해 반복 측정하고 대조 조건·부담·제구·구위를 함께 검사해야 한다. 현재 달성하지 못했다.

> [!warning] 예전 0.799km/h와 이번 18km/h
> 0.799424는 별도 Prime 투구에서 공 관측으로 초기화한 짧은 미래 구간의 공 속력 MAE다. 18.155697은 3034_3에서 신체/강체 그립으로 예측한 전신 모델과 기록의 차이다. 동일 투구·동일 입력·동일 지표가 아니므로 악화량을 빼서 계산하지 않는다.
''',
    '03 동일 투구 실험 결과':f'''# 동일 투구 실험 결과

대표: OBP user2/session3034/3034_3. 기록 {r['recorded_speed_kmh']:.6f}km/h. 필터 20Hz, 제공 BR, 15cm 그립을 유지했다. 새로운 선수/외부 검증이 아니다.

## 속도가 어디서 달라졌나

- 처리 hand_jc: {r['ladder']['speeds_kmh']['processed_hand_landmark']:.6f}km/h. 공 속도가 아니다.
- RFIN: {r['ladder']['speeds_kmh']['RFIN_marker']:.6f}km/h. 공 속도가 아니다.
- 처리 손목: {r['ladder']['speeds_kmh']['processed_wrist']:.6f}km/h.
- 손 회전+15cm 가정 공의 연속 기구학: {r['ladder']['speeds_kmh']['continuous_assumed_ball']:.6f}km/h.
- 자유 어깨 팔 모델 기준: {r['ladder']['speeds_kmh']['free_shoulder_arm_reference']:.6f}km/h.
- 명목 전신 기준: {r['models']['nominal']['reference_speed_kmh']:.6f}km/h.
- 실제 전신 적분: {b['speed_kmh']:.6f}km/h.
- 같은 투구의 직접 공 관측: **없음/미확인(null)**.

전체 기록 차이 {b['record_gap']['value']:.6f}km/h 중 {abs(r['attribution_kmh']['continuous_assumption_minus_record']):.6f}는 전신 연결 전의 손–공 가정 단계에 이미 있다. 전신 연결 차이 {abs(r['attribution_kmh']['body_reference_minus_free_arm']):.6f}, 마지막 적분 차이 {abs(r['attribution_kmh']['forward_minus_body_reference']):.6f}km/h다. 순서에 따른 속력 차이의 분해이며 독립 원인별 인과 기여율이 아니다. 가정한 공과 실제 공의 차이 안에는 미확인 이탈/기하/시간/처리/기록 정의가 섞인다.

## 보정 전후

어깨 부착점만 동결한 M3-A는 4투구 평균 기록 차이 {ag['nominal']['reused_four_pitch_record_gap_mae_kmh']:.6f}→{ag['frozen_attachment']['reused_four_pitch_record_gap_mae_kmh']:.6f}로 악화했다.

첫 투구로 전신 기하를 동결한 M3-F는 대표 기록 차이를 {b['record_gap']['value']:.6f}→{a['record_gap']['value']:.6f}km/h로 줄였다. 모델 속도는 {b['speed_kmh']:.6f}→{a['speed_kmh']:.6f}다. 실제 투수가 빨라진 것이 아니라 기준 모델 보정의 결과다. 역동역학 입력도 모델별로 다시 계산했으므로 동일 힘/일 개입 실험이 아니다.

4개 재사용 검사 투구의 평균 기록 차이는 {ag['nominal']['reused_four_pitch_record_gap_mae_kmh']:.6f}→{ag['frozen_skeleton']['reused_four_pitch_record_gap_mae_kmh']:.6f}로 줄었지만, 첫 투구 대비 구속 변화 MAE는 {ag['nominal']['reused_four_pitch_change_mae_kmh']:.6f}→{ag['frozen_skeleton']['reused_four_pitch_change_mae_kmh']:.6f}로 악화했다. 무변화 기준은 {ag['no_change_mae_kmh']:.6f}다.

빠른 3034_2−느린3034_3의 실제 차이는 {ag['nominal']['fast_3034_2_minus_slow_3034_3_recorded_kmh']:.6f}, 예측은 {ag['nominal']['fast_3034_2_minus_slow_3034_3_predicted_kmh']:.6f}→{ag['frozen_skeleton']['fast_3034_2_minus_slow_3034_3_predicted_kmh']:.6f}km/h였다. 이 한 쌍은 좋아졌어도 전체 변화 성능 악화를 숨기면 안 된다.

대표 손목 위치 RMSE도 {b['observed_joint_rmse_mm']['wrist']:.3f}→{a['observed_joint_rmse_mm']['wrist']:.3f}mm로 악화했다. **기본 성능 예측기로 채택하지 않았다.**

[[06 투구 영상]]에서 실측/명목/기하 동결 모델을 겹쳐 볼 수 있다.
''',
    '04 다음 검증 순서':'''# 다음 검증 순서

## A. 우선 손에서 공으로 이어지는 실제 관측 연결

3034_3의 18km/h 차이 대부분은 전신 연결 이전부터 남는다. 발 접촉 수정만으로 이 차이가 해결된다고 가정하지 않는다. 동일 투구의 손/손가락/공 3D 위치, 동기화 영상, 이탈 전후 시각을 확보해야 고정 그립 가정과 실제 상대 운동을 구분할 수 있다. 다른 선수 Prime의 상대 운동을 복사하지 않는다.

현재 3034에서는 기록 속력 하나만으로 공의 3D 초기 속도·그립·이탈 시점·접촉 힘을 동시에 식별할 수 없다. AI 모델을 붙여도 사라지지 않는 관측의 제약이다. 공개 자료를 더 찾을 때에는 손과 공이 같은 투구에 있으며 원본 시간/카메라 보정이 있는지를 우선 확인한다. 없으면 해당 항목은 미검증으로 유지한다.

## B. 같은 Prime 투구에서 가능한 단계별 검증

Prime에는 직접 공 후보와 신체/손 좌표가 있다. 기존 공 관측 초기화 대조군과 손 모델의 결과를 같은 투구/구간에서 비교할 수 있다. 다만 잠정 정렬, 보정 공유, 가림/블러와 제공 Theia 처리의 불확실성을 먼저 유지해야 한다. 신체 전체와 힘판/근력까지 검증한 자료로 취급하지 않는다.

## C. 전신 구조와 부하 검증을 별도로 진행

동일 선수의 골격 기하를 동결하면 절대 구속만 일부 가까워지고 변화/손목 위치는 악화했다. 부착점·분절 프레임·길이와 움직임의 일관성을 같이 맞춰야 한다. 그 뒤 실측 지면반력/압력중심/자유모멘트를 확인해 기존 골반 지지 입력과 중복 없이 연결한다. 근육 모멘트암과 힘/일 제약도 따로 통과해야 한다.

## D. 성능 모델 채택 조건

새로운 투구/선수에서 관절 자세, 실제 공 초기 상태, 기록/관측 구속과 선수 내 변화가 함께 맞아야 한다. 오차가 후보의 예상 효과보다 큰데 1km/h 향상을 확정적으로 추천하지 않는다. 적분 수렴만으로 통과시키지 않는다.

## E. 구속 외 성능

구위에는 실제 회전수/3D 회전축/궤적과 공력 검증이 필요하다. 제구에는 의도 목표와 실제 도착점 및 반복 분산이 필요하다. 피로·부담에는 반복 투구와 개인 부하/근력 기준이 필요하다. 실제 선수 개입에서는 코치/선수의 재현성과 편안함을 측정하며 모델에서 임의로 확정하지 않는다.
''',
    '05 자료와 코드':'''# 자료와 코드

## 데이터셋

- [OBP 원자료](https://github.com/drivelineresearch/openbiomechanics): baseball_pitching/data의 3034 투구. 같은 투구 직접 공 궤적 미확인. 결과 JSON source_ledgers에 원본 경로와 SHA256, 단위/이벤트/처리 정의를 보존한다.
- [Prime 공 추적 원자료 설명](https://github.com/drivelineresearch/openbiomechanics/tree/main/computer_vision/4d_reconstruction): BaseballThrow_001. OBP 3034와 다른 선수/투구이며 두 자료의 오차를 보정 전후로 비교하지 않는다.

## 이번 코드와 문서

- same_pitch_audit.py: 5투구×3모델×2해상도, 속도 경로와 모델 기하 보정 비교. require_same_evaluation으로 잘못된 비교 차단.
- SAME_PITCH_PROTOCOL.md: 사전 조건과 1차 실패 이후 추가한 검사 구분.
- same_pitch_results.json: 모든 투구/조건, 직접 공 null, 소스 해시와 보정 사용 특징.
- render_same_pitch.py: 그래프 대신 투구 신체 겹침 영상.
- export_model_notes.py: Obsidian 노트 생성. 기존 보관함의 새 폴더로만 복사하며 기존 폴더는 덮어쓰지 않는다.

## 연결된 기존 코드

- trace_error_audit.py / speed_path.py / coupled_payload.py / coupled_body_search.py: 관측·팔/전신·기구학·실제 적분.
- baseline_calibration.py: 부착점 위치 적합. 이번 실험은 질량 보정 없이 관측 기하만 사용했다.
- physical_release_comparison.py / MULTIVIEW_RELEASE_FINDINGS.md: Prime 공 관측 초기화 실험. 이번의 새 오차 개선 결과로 재보고하지 않는다.
- WRAPPING_CONSISTENCY_FINDINGS.md: 별도 근육 모델의 잔차와 경로 제한.
- [PR87](https://github.com/choigod1023/pitching-real2sim-lab/pull/87): 가상 이탈/회전 시나리오.
- [PR88](https://github.com/choigod1023/pitching-real2sim-lab/pull/88): 실제 전방 적분 타이밍 탐색, 채택 후보0개. 이 노트 생성 시 원격 main은63fc06c였고 PR87/88이 합쳐졌다고 가정하지 않았다.

## 재현

원자료를 프로젝트 형제 openbiomechanics 폴더에 준비하고 requirements.txt의 Python 의존성을 설치한다.

```powershell
python same_pitch_audit.py
python render_same_pitch.py
python export_model_notes.py
python -m unittest test_same_pitch_audit test_trace_error test_speed_path
```

근육/다른 스포츠의 결과를 이번 야구 성능 개선으로 합산하지 않는다. 각 보고서의 데이터셋과 증거 범위를 유지한다.
''',
    '06 투구 영상':'''# 투구 영상

![[attachments/동일투구_골격보정_3034_3.mp4]]

회색: 실측 관절. 파랑: 명목 전신 모델. 주황: 첫 투구로 기하를 동결한 모델. 10초,24fps, FP→BR 약50배 느린 재생 뒤 정지. 원래/보정 모델은 실제 전방 적분 결과이며 관찰 좌표로 중간 상태를 덮어쓰지 않았다. 두 시점은 같은 좌표/시각이다.

> [!warning] 경기력 향상 영상이 아니다
> 선수의 동작을 최적화한 결과가 아니라 같은 투구를 재현하는 모델의 기하 보정 비교다. 모델별 기준 힘을 다시 계산했다. 같은 힘/일의 개선으로 읽으면 안 된다. 실제 구속 차이는 여전히 크고 손목 위치/전체 투구 변화 예측도 통과하지 못했다.

관절 좌표 입력은 같지만 골격 기하와 실제 적분 결과가 달라 신체 위치가 달라진다. 모델은 발 접촉/개인 근육/실제 손가락 이탈이 검증되지 않았고 머리는 표시용이다. 직접 공 궤적이 없는 3034에서 관측 공의 영상으로 오해하면 안 된다.
'''}
    for name,body in notes.items():
        (output/(name+'.md')).write_text('---\ntags: [야구, Real2Sim, 모델검증]\nupdated: 2026-10-28\n---\n\n'+body+common,encoding='utf-8')
    attachment=output/'attachments';attachment.mkdir(exist_ok=True)
    video=OUT/'same_pitch_body_calibration.mp4'
    if video.exists():shutil.copy2(video,attachment/'동일투구_골격보정_3034_3.mp4')
    return output


def publish(notes, vault):
    vault=Path(vault)
    if not (vault/'.obsidian').is_dir():raise ValueError('Existing Obsidian vault required')
    target=vault/'50 야구 Real2Sim/2026-10-28 동일 투구 모델 검증'
    if target.exists():raise FileExistsError('Refusing to overwrite an existing note folder')
    shutil.copytree(notes,target)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--vault',type=Path)
    args=parser.parse_args();notes=build_notes()
    print(publish(notes,args.vault) if args.vault else notes)
