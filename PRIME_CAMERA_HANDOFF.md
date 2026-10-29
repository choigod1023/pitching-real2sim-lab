# 다음 재개: 빨간 영역 중심과 파이프 축 교차점 검증

주 기준은Prime BaseballThrow001이다. PRIME_CAMERA_FINDINGS.md와
prime_camera_validation.json,prime_cube_correction_results.json을 먼저 읽는다.
Prime17/20/21의checkerboard/cube6영상 확보, 원본3해시 일치,9새 시각27쌍 중
23검사/4제외, 큐브 색검출/원본대응 동결, 원래리그 길이 검사, cube-only PnP 수정,
제외중간점/체커보드 재검사, 배경SIFT6조건, 사후크기/형상 분리를 완료했다.
동일 계산/다운로드를 반복하지 않는다. 캐시와 명시 원본 해시를 재사용한다.

원래 큐브17/20 모서리 MAE28.884mm/재투영0.712px. cube-only수정으로
반변MAE18.690→14.361mm지만 체커보드1.371→4.818mm 악화해 기각했다.
17/21,20/21도 모두 체커보드 악화. 카메라21 큐브 대응은 윗점 픽셀만으로
원본 차트A→B,B→C,C→D,D→A를 선택했고 나머지7대안도 보존했다.
이를 저자가 확정한실물ID로 부르지 않는다. 새 카메라를 투구에 적용하지 않는다.
기존B 조건부속력MAE1.324139 유지.5UNKNOWN/data단계/실제선수효과 미검증.
관련9테스트 통과. 배경 cube→board변위중앙값0.295~0.349px지만
camera21 cube→baseball p95=32.042px로 그 시야 전체불변성 단정 금지다.

첫 미완료는 **색 마스크 중심과 물리 파이프 축 교차점의 분리**다.
원본static cube frame0과 cube17/20/21_raw_markers.png를 보고, 기존 색중심/대응을
변경하지 않고 별도 후속 프로토콜을 고정한다. 흰 파이프 선분 중심선과 교차점은
속도/3D길이/PnP 결과를 목적에 넣지 않고2D 원영상으로만 정의한다.
중간 빨간조인트/다른파이프 겹침/부분 가림/그림자/원근에 의한 불확실성을 기록하고,
원래색중심과 좌표차이·선교차잔차·정해둔 감도조건을 보고한다.
명확한 부착/기하 근거가 생기면 별도train/eval을 고정하고cube/board공동 보정을 시험한다.
실물 중심을 색덩이 중심으로 단정하거나 길이에 맞춰 주석/큐브치수를 옮기지 않는다.
제외표본 물리길이까지 좋아지기 전에는구속오차개선으로 해석하지 않는다.

원본과검출캐시: artifacts/prime-camera. board_detections의9시각 좌표는
prime_board_detections.json으로루트에도복사한다. 참조sourceprov원본K/리그JSON은
upstream openbiomechanics/computer_vision/calibration/results. Cube698.5/349.25mm,
checker100mm근거는sourceindexB14/B17이며,캡처clock/BROffset미확인은별개다.
공개폴더초기50개와달리embeddedview128개에staticcube가있다. 그목록/원본은이미확보했다.
새에이전트추가금지. 검증개선이면진짜야구신체겹침영상,기각이면향상주장금지.
