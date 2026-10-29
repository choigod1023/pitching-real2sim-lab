# 공개 손·공 자료 검사 범위

2026-10-29. 이 문서는 논문·원본의 존재와 계측 정의를 검사한다. Stage2 분포, 과학 gate, PAUSED 상태를 변경하지 않는다.

발견한 Ozkaya 2018의 Subject1 C3D archive와 CSV archive 및 Trialinfo.xlsx를 내려받고 API MD5와 대조한다. 탐색적으로 먼저 읽은 투구는 S1/1w2d/DP05_O, DP06_O, S1/1w1d/NDP07_O 세 개다. 파일명 O는 제공 상태표의 OK이지 이번 정확도 검증 결과가 아니다. 표본은 모집단·오차분포 추정에 사용하지 않는다.

세 파일의 프레임·단위·명시 마커·결측·EVENT를 읽고, 원문 기준 event를 그대로 사용한다. 이탈을 구속에 맞춰 재선택하지 않는다. 원시 C3D header의 첫 프레임을 사용해 event time→raw frame→배열 index를 만들고 CSV ITEM 좌표와 대조한다. 이탈 직후 고정 5표본(event+1…+5)이 모두 유효한 경우에만 순수 선형 위치 적합의 기울기 크기를 기술한다. 손/공 표식 속도이며 공중심이나 독립 레이더 정확도로 해석하지 않는다. 구속표 단위가 확정되지 않으면 숫자의 구속 오차를 계산하지 않는다.

같은 1w2d 정적 두 파일의 wrist styloid/forearm cluster 라벨, 유효 표본 및 비공선성을 검사한다. 해당 검사만으로 피부표식 강체성, 골중심 정확도 또는 손 roll을 입증하지 않는다. 원본·추출좌표는 ignored artifacts에만 저장한다. 새 보조 코드에 의미 있는 단위검사를 추가한다.

## 시간 대응 정정 부록

후속 공식문서 확인으로 초기의 EVENT time→raw frame 단일 대응을 철회한다. C3D 표준(첫 표본 time0)은 raw_frame=round(t*Hz)+1, legacy producer 가정은 raw_frame=round(t*Hz)다. CSV ITEM과 배열의 공간 좌표 대응은 이벤트 시간 원점의 증거가 아니다. 두 후보를 모두 검사하고 실제 source_event_mapping은 UNKNOWN으로 둔다. 더 좋은 구속·잔차·가시 표본 수를 근거로 후보를 선택하지 않는다. 원래 결과·보고서·출처목록·검증로그는 history_initial_mapping_20261029에 보존한다.
