# 같은 투구의 신체 개입 모델 연결 감사

이 감사는 현재 로컬 소스를 읽고 수행했다. 다른 선수·다른 자료의 검증 결과를 스킨스에 대입하지 않는다.

| 구성요소 | 원래 코드·자료 | 입력/출력과 제한 |
|---|---|---|
| 영상 추출기 | `mlb-rtmpose-worker-20261104/experiments/mlb_detector_comparison/detector.py`, `map_to_33` | COCO5~16의12점을33칸 중11~16,23~28에만 채움. 손가락은NaN |
| 현재 스킨스 캐시 | PR162 `mapped-tracks.private.npz` | 2D좌표·점수·시각·프레임. qpos/qvel/GRF/근력·관성/손접촉 없음 |
| 스킨스 비행 | `skenes-ff-real2sim-20261110/experiments/skenes_fastball_scenarios/study.py`, `release_state`, `scenario` | 50ft 속도/가속도로 방향을 역외삽한 뒤 기록release_speed로 크기 정규화. RPM도 기록 입력. 슬롯은0.65m 가정반경 위치변경 |
| 실제 팔–손–공 | 원격main `coupled_payload.py`, `compliant_timing.py`, `curl_pitch.py` | MuJoCo 실제 동역학이지만 OBP3034 및 일반관성·가정그립/접촉. 스킨스 상태로 보정·검증한 모델이 아님 |
| 힘·일 제한 | `capacity_budget_gate.py`, `gate` | 자유도별 양/음의 일·최대힘/토크를 기준과 비교. 현재스킨스 동일투구 기준입력 없음. 생리학적 근력검증과도 구분 |
| 하체 입력 경로 | `foot-driven-real2sim-20261109/experiments/foot_driven_real2sim/study.py` | OBP3034의 별도 발힘재생 연구. 같은 스킨스 신체개입 어댑터로 사용불가 |

기존 모델 자체를 삭제하거나 수치를 덮어쓰지 않았다. 관찰 후보가 통과하더라도 `release_speed`와RPM을 입력한 결과를 그 신체 변화의 예측 구속/RPM이라고 보고하면 안 된다. 관측 초기조건으로부터의 비행 재생과 신체에서 생성되는 방출 상태를 별도 경로로 유지한다.

이번 `simulation_gate`는 기록 보간·통계적연관·2D골격만으로 신체개입을 허용하지 않는다. `predicted_gain`은 미실행일 때null이며0이 아니다. 조건을통과했다는테스트fixture는실제선수근거로사용하지않는다. 사용자에게시뮬레이션이됐다고보고하려면해당선수·해당투구의모델에서실제전방적분및별도부하검증이필요하다.
