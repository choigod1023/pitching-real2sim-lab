# 이벤트 표본 시각의 정의 정정

첫 계획과 실행은on-disk first=1을 시각에도 적용해array258을 후보event로 뒀습니다.
CSV ITEM259와array258의 일치는 공간값과 배열 번호의 대응만 보여주며,
event1.295초가ITEM259라는 시간 대응은 입증하지 못했습니다. 이 추정을 철회합니다.

[C3D 공식 EVENT:TIMES](https://www.c3d.org/HTML/Documents/eventtimes1.htm)는
first3Dsample(frame1)=0.0초로 정의합니다. 이 규약이면array259/sourceframe260입니다.
원래producer가legacy frame/rate 시각을 썼을 가능성은 이 자료만으로 배제하지 못합니다.
실제 접촉 해제를 독립적으로 확인하지 못했으므로source_event_mapping=UNKNOWN입니다.

이 정정을 계산 전에 별도커밋하고 표준 규약의array259를spec nominal,array258을
legacy alternative로 모두 보존합니다. 첫 계획/실행summary와codehash는
initial_mapping/에 둡니다. 규약을 예측 오차로 고르지 않습니다. 기존shift0/1과 새
shift-1/0은 같은 두 물리 표본 창입니다. 수치가 좋은 쪽을 채택하지 않습니다.

이는 시각 정의 정정이며 실제 이탈 시점을 검증한 것이 아닙니다. 기존OBP나
Prime의 시각 대응에 자동 적용하지 않고 이 자료의 출처와 규약만 다룹니다.
