import json
from pathlib import Path
import numpy as np
from trace_error_audit import sha
root=Path('artifacts');a=json.loads((root/'trace-error/report.json').read_text());b=json.loads((root/'trace-error-extension/report.json').read_text())
assert a['protocol_sha256']==b['protocol_sha256']==sha('TRACE_ERROR_PROTOCOL.md')
assert a['code_sha256']==b['code_sha256']==sha('trace_error_audit.py')
runs=a['runs']+b['runs'];ledgers=sorted(a['ledger']+b['ledger'],key=lambda x:x['pitch'])
assert len(runs)==80 and len(ledgers)==5
summary=[]
for ledger in ledgers:
 p=ledger['pitch'];rows=[r for r in runs if r['pitch']==p]
 old=next(r for r in rows if r['source_rate_requested']==11520 and not r['padded_derivatives'] and r['policy']=='zoh' and r['substeps']==2)
 same=next(r for r in rows if r['source_rate_requested']==11520 and r['padded_derivatives'] and r['policy']=='foh' and r['substeps']==2)
 new=next(r for r in rows if r['source_rate_requested']==46080 and r['padded_derivatives'] and r['policy']=='foh')
 assert max(new['fk_to_rollout_joint_rotation_max_deg'].values()) < max(old['fk_to_rollout_joint_rotation_max_deg'].values())
 assert new['fk_to_rollout_position']['rmse_mm'][3]<old['fk_to_rollout_position']['rmse_mm'][3]
 assert all(r['inverse_forward_acceleration_max_abs']<1e-6 for r in rows)
 summary.append(dict(pitch=p,record=ledger['recorded_speed_kmh'],old=old,same_grid_fix=same,new=new))
(root/'trace-error/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
lines=['# 관찰–기구학–전방 적분 오차 분리 결과','', '2026-10-26. 에이전트 없이 같은 선수 ID2의5개 개발 투구에서80조건을 실행했다. 실측 구속을 목표로 필터·그립·시점을 조정하지 않았다. 원 관찰 자료의20Hz 처리와15cm 강체 그립은 유지했다.','', '## 확인한 원인과 수정','', '기존 코드는 힘을 원본 입력 구간 동안 일정하게 유지했다(ZOH). RK4 적분 간격만 줄이면 그 계단형 입력에 더 정확해질 뿐, 관찰 궤적의 연속 힘을 재현하는 문제는 해결되지 않았다. 또한 FP..BR 안의 자료만 미분해 시작과 끝의 속도는 한쪽 차분이었다.','', '새 integrate_inputs(..., policy="foh")는 RK4 내부 단계의 시각에서 힘을 선형 보간한다. 관찰 구간 양끝에2표본을 더 확보한 뒤 중심 차분을 적용하는 경로도 추가했다. 합성 선형 힘의 해석해 일치, 상수 힘의 두 정책 일치, ZOH에서 적분 간격만 줄여도 입력 오차가 남는 현상을 실제 Python 테스트로 확인했다. 기존 스크립트의 결과를 조용히 덮어쓰거나 모든 기존 실험을 재검증했다고 주장하지 않는다.','', '확장 표본은 BR 이후의 관찰도 사용하는 오프라인 수치 진단이다. 실시간 예측·미래 프레임을 보지 않은 예측이 아니다. 1/46080초 격자는360Hz 측정 스플라인의 수치 평가 간격이며 새 관측 정보를 추가한 것이 아니다.','', '## 같은 투구의 전후 수치','', '이전은1/11520초 입력·한쪽 끝점 차분·ZOH와 dt/2 적분, 수정은 양끝 확장 차분·FOH·1/46080초 입력/적분이다. 같은 입력 격자에서 미분/보간만 바꾼 결과도 JSON에 별도로 보존했다. 회전은 해부학적 각도가 아니라 marker-roll 좌표다.','', '| 투구 | 기록 구속 | 이전 모델 | 수정 모델 | 이전 회전 차이 | 수정 회전 차이 |','|---|---:|---:|---:|---:|---:|']
for x in summary:
 lines.append(f"| {x['pitch']} | {x['record']:.3f} | {x['old']['final_rollout_speed_kmh']:.3f} | {x['new']['final_rollout_speed_kmh']:.3f} | {x['old']['fk_to_rollout_joint_rotation_max_deg']['marker_roll']:.6f}° | {x['new']['fk_to_rollout_joint_rotation_max_deg']['marker_roll']:.6f}° |")
s=next(x for x in summary if x['pitch']=='3034_4');f=next(x for x in summary if x['pitch']=='3034_2')
real=f['record']-s['record'];old=f['old']['final_rollout_speed_kmh']-s['old']['final_rollout_speed_kmh'];new=f['new']['final_rollout_speed_kmh']-s['new']['final_rollout_speed_kmh']
lines+=['','속력 단위는 km/h. 이전·수정의 각도/위치 차이는 관찰에서 만든 모델 참고 궤적 대비 적분 오차이며 실제 선수 측정 정확도가 아니다.','',f"3034_4의 같은1/11520초 입력 격자에서 회전 차이는 {s['old']['fk_to_rollout_joint_rotation_max_deg']['marker_roll']:.6f}→{s['same_grid_fix']['fk_to_rollout_joint_rotation_max_deg']['marker_roll']:.6f}도다. 더 촘촘한 입력에서 {s['new']['fk_to_rollout_joint_rotation_max_deg']['marker_roll']:.6f}도로 감소했다. 가정 공 중심의 궤적 재현 RMSE는 {s['old']['fk_to_rollout_position']['rmse_mm'][3]:.6f}→{s['new']['fk_to_rollout_position']['rmse_mm'][3]:.6f}mm다.", '',f"실제 빠름−느림 차이는 **{real:.6f}km/h**, 이전 모델은 **{old:.6f}**, 수정 모델은 **{new:.6f}**다. 실제 변화량과의 차이는 {abs(real-old):.6f}→{abs(real-new):.6f}km/h로 약 {abs(real-old)-abs(real-new):.6f}만 감소했다. 실제1km/h 정확도는 충족하지 못했다.", '',f"기준3034_4의 기록−모델 차이는 수정 후 **{s['new']['record_minus_rollout_kmh']:.6f}km/h**이며 오히려 소폭 증가했다. 수치 추적 개선을 기록 구속 오차 개선이라고 부르지 않는다.", '', '## 관찰 좌표에서 모델로 옮기는 단계','']
for x in (s,f):
 n=x['new'];lines.append(f"- {x['pitch']}: 처리 손목 중심→기구학 손목 위치 RMSE {n['observed_to_fk_position']['rmse_mm'][2]:.3f}mm, BR 속도 벡터 차이 {n['observed_to_fk_final_velocity_vector_difference_kmh'][2]:.3f}km/h. 손 프레임 방향의 대응 오차는 {n['observed_hand_frame_to_fk_max_deg']:.2g}도로 수치 수준이었다.")
lines+=['','관절 중심 사이 거리는 원 관찰에서 시간에 따라 변하지만 모델은 고정 길이 강체다. 따라서 위치 차이를 무조건 전방 적분 버그나 실제 뼈 길이 변화로 해석하지 않는다. 이 단계의 남은 기하·관절 중심 정의 차이는 별도 문제다.','', '손목 속도·RFIN 속도·처리 hand_jc 속도와 모델 공 중심 속도는 서로 다른 점이다. JSON에는 각각을 보존했으며 손 속도를 실제 공 속도로 대체하지 않았다. 손 회전 항의 노름과 손목 속력도 단순 합산하지 않는다.','', '## 다음 결정','', '1. 같은 힘의 적분 수렴만 확인하는 방식에서, 입력 생성 해상도와 경계 미분까지 확인하는 검증으로 바꾼다. 새 구현과 테스트를 후속 실험의 기준으로 사용할 수 있으나 수정 동작의 구간 밖 연장/끝점 미분은 다시 정의해야 한다.','2. 기존 최적화에서 검사한 자세·일 조건은 기존 ZOH 입력에 대한 결과다. 새 입력 방식에서 같은 판정을 받는다고 가정하지 않는다. 실제 변화 예측이 실패한 상태이므로 최적화 재실행은 아직 보류한다.','3. 다음은 동일 Prime 투구의 손/공 좌표와 시각 대응, 상대운동 및 이탈 구간이다. 원시 공 추적이 없는3034 자료로 실제 그립이나 손가락 힘을 식별했다고 주장하지 않는다.','', '검증: 전체169테스트 중167통과, 기존 선택적2개 제외. 5개의 새 합성 회귀 검사를 포함한다. 80조건 모두 저장했으며 소스/프로토콜 해시, 입력과 상태, 각 단계 위치·회전·속도 벡터 차이를 artifacts/trace-error 및 artifacts/trace-error-extension에 보존했다.','', '재실행: ../.venv/Scripts/python.exe trace_error_audit.py (3034_4/2), 추가3투구는 --out artifacts/trace-error-extension --pitches 3034_1 3034_3 3034_5.']
Path('TRACE_ERROR_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
ledger=['# 오차 장부','', '선수2의 같은 세션3034 투구5개. 원자료 식별자와 해시는 각 report.json의 ledger에 보존했다. 레이더 위치·측정 시각과 실제 BR 검출 방식은 미확인이다.','', '원 C3D360Hz와 처리 CSV의 반올림 시간표를 대조했다. 길이 m, 속도 m/s에서 km/h로 환산, 모멘트 Nm. 손/전완 마커는4차20Hz 양방향 필터, 관절 중심은 공급된 처리 좌표를 그대로 사용한다. 일반 팔/손 관성과145g 공, 손목에서 마커 x축15cm 그립 가정은 유지했다.','', '자료별 파일 경로와 길이:']
for l in ledgers:ledger.append(f"- {l['pitch']}: FP10={l['fp10_s']}, BR={l['br_s']}, 관찰 {l['recorded_speed_kmh']:.6f}km/h, 길이={l['lengths_m']}; C3D {l['source_paths']['c3d']}")
ledger+=['', '주요 미확정: 실제 공 중심·그립·손가락 접촉, 해부학적 손목/회내외 보정, 근육 능력, 레이더와 BR속도의 동일성.','', '이 장부는 자료 존재와 처리 정합성을 기록하며 미확정 항목을 추정 정답으로 채우지 않는다.']
Path('ERROR_LEDGER.md').write_text('\n'.join(ledger)+'\n',encoding='utf8')
print('validated',len(runs),'runs; speed change',real,old,new)
