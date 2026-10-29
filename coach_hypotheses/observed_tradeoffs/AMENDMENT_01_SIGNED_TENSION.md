# 사전 부호 정의 정정: 견인 극값과 견인 크기 분리

2026-10-29. 최초 실제 실행은 validate_frame의 negative_speed_or_peak_load 오류로 중단됐다. 비교 함수가 실행되기 전이며 유리한 쌍/부하 차이/집계는 아직 계산하지 않았다. 기존 PROTOCOL.md는 그대로 보존하고 이 정정을 별도 커밋한다.

기존 지표 생성기는 shoulder_distraction_peak_n=max(-shoulder_upper_arm_force_z)를 PKH~BR에서 계산한다. 1,197행 중 523행이 음수였다. 원본 OBP baseball_pitching/README.md의 Forces 표는 상완 기준 어깨 z를 Compression(+)/Distraction(-)로 정의한다. 따라서 음수 극값은 그 검색 구간의 z가 모두 양수인 경우이며 음수의 '견인 크기'가 아니다. 값이나 부호를 구속에 맞추지 않는다.

정정 비교에는 shoulder_tension_peak_magnitude_n=max(0, shoulder_distraction_peak_n)를 추가한다. 이는 max_t max(0,-Fz(t))와 정확히 같아, 해당 검색 구간에 음의 Fz가 없을 때 견인 성분 크기를 0으로 표현한다. 원래 열은 수정하거나 삭제하지 않고 모두 보존한다. 신호를 0으로 보간하거나 측정 노이즈를 제거한 것이 아니며, 소스 구간/좌표의 정확도가 새로 입증된 것도 아니다.

원래 세 부호 포함 열을 사용한 판정과 정정된 세 크기 지표를 사용한 판정을 모든 쌍·필터에 병기한다. 원래 판정 통과 개수, 정정 판정 통과 개수, 판정이 달라진 쌍을 함께 보고한다. 정정 비교의 다른 두 지표는 기존 elbow_varus_peak_nm과 shoulder_resultant_peak_n이다. 대표 사례 선택 규칙과 1km/h 부분집합, epsilon, 모든 미검증 플래그는 유지한다. 이 정정으로 기존 구속 예측 gate를 바꾸지 않는다.

추가 테스트는 모든 압축/견인/혼합 신호에서 극값의 양의 부분과 양의 부분 극값이 일치하는지, 원래 부호 열이 보존되는지, 원래/정정 판정 차이가 결과에 나타나는지 확인한다. 상류 지표 파일은 변경하지 않는다. 원본 README의 경로·해시·저장소 커밋도 결과 manifest에 기록한다.
