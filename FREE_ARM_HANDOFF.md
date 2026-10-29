# 다음 재개: Prime 카메라 기하와 표면 관측의 원인 분리

FREE_ARM_FINDINGS.md, free_arm_results.json, free_arm_diagnostics.json,
surface_point_map_results.json, free_arm_validation.json을 먼저 읽는다.
이번 원영상 판독8개와12적합/40평가 다프레임free-root IK, 하위시각 진단,
LP940/960, 손목/회전 귀속 대조, 로컬표면맵6파라미터 적합은 완료했다.
재실행하지 않는다. MAE1.324→0.797만 보고 채택하지 않는다. 평가영상2.645→4.226px,
어깨−30mm 경계로 기각했고 표면맵도2.645→3.055px 및 경계로 기각했다.
B에는 학습 공 좌표가 들어가며 독립적인 관절→구속 능력이 검증된 것이 아니다.
기준 B와과학 data5UNKNOWN/상태는 유지한다. 선수효과·힘·일·구위·제구 미검증.

첫 미완료는 **Prime OptiTrack15–22 보정 원영상/캐시 접근과 별도 알려진 길이 검증**이다.
upstream openbiomechanics/computer_vision/calibration/README.md, ROADMAP.md,
ANNOTATION_PLAN.md와results/optitrack_rig_provisional.json을 읽는다. 리그는
provisional_not_bundle_adjusted이고 Prime의 checkerboard 원본 해시/이름은 JSON에 있다.
opti_dynamic_checkerboard_cam15..22.mp4와opti_cube_cam*.mp4를 공개 수집 목록/기존 캐시에서
찾는다. lab 내부 no-ignore 이름 검색에서는 해당 파일을 찾지 못했다. upstream README의
기본 /tmp/obp-calibration-assets는 다른 OS의 예시이므로 로컬 존재를 가정하지 않는다.
기존 보정은 그대로 재계산하지 말고 원본·검출 캐시·기존train/eval을 재사용한다.
Prime 카메라17/20/21의 동시 표식 방향·길이를 구속/관절과 무관하게 검증하고, 가능한 경우
사전 train/eval을 정해 공동 기하 보정/제외표본 실제 길이 검사를 한다.
Edger 카메라1–4의 완성/실패를 Prime의 전후처럼 사용하지 않는다.
구속에 맞춰 원점·이탈·근력·관성을 고르거나 모호한 큐브/표식 정체를 가정하지 않는다.

관련17테스트 통과. 진단 영상은artifacts/free-arm-regions/
baseball_reconstruction_rejected_overlay.mp4,220프레임30fps7.33초, 전체디코드/대표프레임 확인.
영상은 선수 개선이 아니라 기각된 재구성의 차이다. 새 하위 에이전트 금지.
각 단계 scientific validator/research_gates verify/advance 규칙을 유지한다.
동시 실행/lease를 먼저 확인한다. 의미 있는 결과를 한국어로 한번 알리고 반복은 조용히 한다.
