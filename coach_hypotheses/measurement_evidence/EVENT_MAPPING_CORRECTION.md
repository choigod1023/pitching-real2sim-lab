# EVENT 시간 원점 정정

2026-10-29. 초기 커밋1975912279dabfb0393a206f77b1b041a6e1d906에서 CSV ITEM259=C3D 배열258이라는 좌표 대조를 근거로 이벤트1.295s도 배열258이라고 확정했다. **좌표·번호 대응에서 시간 원점까지 추론한 것은 오류였다.** 초기 보고서의 “배열259를 쓰면 한 표본 밀린다”는 주장을 철회한다.

[C3D 공식 EVENT:TIMES](https://www.c3d.org/HTML/Documents/eventtimes1.htm)는 첫3D표본(frame1)이 time0.0이라고 명시한다. 첫 raw frame이1인 이번 파일에서 이 규약대로 해석하면 array_index=round(t*Hz)이다. 따라서 DP05의1.295s는 배열259/source frame260이다. 첫프레임을1/Hz 시각으로 세는 producer legacy를 가정하면 배열258/source frame259이지만, 이번 producer가 그 방식으로 기록했다는 원문·파일 생성기·시계 동기 근거는 확인하지 못했다.

현재 결과는 표준을 규약 참조로 기록하고 source_event_mapping=UNKNOWN을 유지한다. DP05는259/258, DP06은309/308, NDP07은311/310 두 후보를 모두 계산했다. 구속·움직임 오차나 유효 표본이 많은 쪽으로 시계를 고르지 않았다. CSV의 이벤트 값은 첫 행 별도 열에 저장돼 있고 각 좌표 ITEM에 이벤트 시간이 직접 연결된 형식이 아니다. CSV ITEM→배열 대응만으로는 두 시간 관례를 구별할 수 없다.

각 후보 +1…+5의 고정 표식 속력 계산 결과:

- DP05: 표준259는 마지막 표본 결측으로 미산출. legacy258은11.772339m/s.
- DP06: 표준309는11.242174m/s. legacy308은10.884678m/s.
- NDP07: 두 후보 모두 결측으로 미산출.

이는 공 표면 표식과 이벤트 대리 시점의 조건부 계산이며, 공 중심 구속·실제 접촉 이탈·선수 효과의 검증이 아니다. 초기 결과는 history_initial_mapping_20261029에 그대로 보존했다. sample_audit.json과 source_inventory.json의 역사 사본은 초기 verification.json에 기록한 SHA256과 일치함을 검사했다. 새 코드는 명시한 후보 관례가 있어야 index를 반환하고 UNKNOWN을 묵시 선택하지 않는다. 정정 후 단위검사7개와 실제3투구/원본 해시/역사 결과 보존 검사를 통과했다. 기존 Stage2 분포·과학 gate·PAUSED 상태는 바꾸지 않았다.
