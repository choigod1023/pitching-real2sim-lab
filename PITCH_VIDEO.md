# 야구 투구 관찰 영상

실제 OBP BaseballThrow_001 카메라19/22 영상에 탐색적 공 좌표를 표시했다. 동작 수정 시뮬레이션이나 구속 향상 예측 영상은 아니다. 원본360fps 가정 아래780~1040프레임을30fps로 재생하고, 이탈 후6프레임을 확대했다. 출력13.7초/1280×800/H264/411프레임이다. 프레임 번호는0기준이며 원본에 인쇄된1기준 숫자와1차이난다.

녹색은 밝기 기반 공 위치 후보다. 빨강은 신발 겹침으로 주 분석에서 제외한 위치다. 겹친 위치를 실제 공 중심으로 채택하지 않았다. 영상 끝의 속력 범위는 TRAJECTORY_FINDINGS.md의 조건부 결과이며 실측 정답이나 향상량이 아니다.

## 다음 연구 순서

1. 추가 카메라와 더 긴 자유비행 구간에서 공 외곽·중심 추적을 검증한다. 신발과 겹치는 관측은 분리하거나 제외하고, 검출 방법에 따른 약2km/h 차이를 먼저 줄인다.
2. 프레임 동기화, 실제 시간 간격, 카메라 척도/상대 자세를 검증한다. 화소 잡음만 낮춘 결과와 보정 편향 감소를 구분한다.
3. 측정 장비·시점이 확인된 같은 투구의 구속과 대조한다. 미사용 투구/선수에서 절대 속력과 선수 내 변화량을 평가하고 무변화 기준을 넘어야 한다. 1km/h 효과의 부호를 불확실성 아래 판단할 수 있어야 한다.
4. 검증된 공 상태를 팔·손 모델과 연결하고, 같은 힘/외부 일 예산 아래 작은 관절 변경 후보를 비교한다. 손가락과 이탈 전 관측 부족은 별도로 해결해야 한다. 실제 훈련 효과는 최종적으로 선수 개입 실험이 필요하다.

## 재실행과 확인

`../.venv/Scripts/python.exe render_pitch_review.py`

기존 comparison-video/artifacts/video-vendor의 imageio_ffmpeg와 Windows 맑은고딕을 사용한다. 입력은 이전 작업의 원본MP4/추출PNG/trajectory_fit.json이다. artifacts/pitch-review/manifest.json에 SHA256과 출력프레임 수를 저장했다. 실제 출력411프레임을 전부 다시 디코딩했고100/270/380프레임의 영상·확대·결과 화면을 직접 확인했다. 원본과 출력은 로컬에 보존한다.

원자료: [OpenBiomechanics / Driveline Baseball](https://github.com/drivelineresearch/openbiomechanics/tree/main/computer_vision), [자료 이용 조건](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md). 원영상에 확대·배속·후보표시·한국어 설명을 추가했다.
