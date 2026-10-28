# 실제 야구 영상과 관절 동작 재구성

같은 `OBP_movement_BaseballThrow_001`의 공개 Theia3D 파일을 확보하고, 실제 카메라 19 영상과 분절 원점으로 그린 골격을 나란히 재생했다. 이는 관측 동작의 운동학적 재구성이다. 힘 기반 전방 적분이나 동작 수정에 따른 구속 증가 예측은 아니다.

## 원자료와 대응 근거

- [공개 C3D](https://drive.google.com/file/d/1EgP-OVVpp5wv2pgorCV8Qs4-vcXZ_NjX/view): 1,820,160 bytes; SHA256 `6b9edc5c2195ae717f9aa18a1a34476ca70fa3f2f46860192c0a4eb9440eca72`.
- 360 Hz, 1,406 frames, 19개 4×4 분절 변환. POINT 좌표는 없으며 ROTATION 변환의 병진 성분을 사용했다. 모두 유한값이었다.
- Theia3D 2023.1.0.3161, 모델 10.0.1, 파일에 기록된 필터 30 Hz. 손 분절은 있으나 손가락이나 공 분절은 없다.
- [상류 README](https://github.com/drivelineresearch/openbiomechanics/blob/main/computer_vision/4d_reconstruction/README.md)는 이 파일과 Prime Color 62.7 mph 투구의 대응을 명시한다. Edgertronic 69.1 mph 자료를 합치지 않았다.
- [제공 정렬](https://github.com/drivelineresearch/openbiomechanics/blob/main/computer_vision/4d_reconstruction/theia_alignment.json)의 `C3D index = video index + 1`을 영상 표시용으로 사용했다. 8개 카메라의 관절에 맞춘 투구별 변환이며 원 적합 코드는 공개되지 않았다. 이 결과를 독립 카메라 보정이나 동기화 검증으로 사용하지 않는다.
- 상류 README에 적힌 `release at 950`은 이번 영상의 실제 공 이탈을 확정하는 근거로 사용하지 않았다. 영상에 표시된 1부터 시작하는 프레임 번호와 0부터 시작하는 디코딩 인덱스도 구분했다.

## 영상

`artifacts/joint-video/baseball_joints_side_by_side.mp4`: 1280×800, 30 fps, 301프레임, 10.03초. 원 영상 인덱스 780~1080을 1/12 배속으로 표시한다. 왼쪽은 실제 투구를 확대하고 제공 정렬을 적용한 관절선, 오른쪽은 원 C3D 실험실 좌표에서 본 사선 방향 골격이다. 양팔과 손목을 주황색으로 표시했다.

부가 manifest의 오른쪽 팔꿈치 내각은 어깨–팔꿈치–손목 분절 원점으로 계산한 값이며 해부학적으로 보정한 임상 관절각이 아니다. 영상에는 숫자를 넣지 않고 움직임을 강조했다. 팔·손목 궤적의 회전과 병진을 관찰할 수 있지만 근력, 편안함, 구속 개선 효과를 측정한 것은 아니다. 공 위치를 손에 임의로 붙이지 않았다.

이 자료를 힘 기반 모델에 연결하려면 관절 중심·질량·관성 정의와 손–공 접촉, 좌표계·시간 정렬을 추가로 검증해야 한다. 현재 공 삼각측량의 임시 보정 프레임과 제공 Theia 정렬을 합친다고 실제 공–손 오프셋의 정답이 생기지는 않는다.

### 관절 재구성과 기존 힘 기반 시뮬레이션을 함께 보는 영상

`artifacts/joint-video/baseball_joints_and_dynamics.mp4`는 위 10.03초 영상, 서로 다른 투구임을 알리는 2초 카드, 기존 `3034_1`의 15초 힘 기반 팔·손·공 시뮬레이션을 차례로 담았다. 1280×800, 30 fps, 811프레임, 27.03초이며 전체 프레임을 다시 디코딩해 확인했다. 원래 24 fps인 물리 시뮬레이션 영상은 재생 시간을 보존해 30 fps로 표시 재표본화했다. 새 물리 실험을 수행하거나 원본 결과를 수정한 것은 아니다. 각 원본 SHA256과 프레임 수는 `combined_manifest.json`에 기록했다.

후반 `3034_1`은 앞의 실제 영상과 다른 투구다. 머리말에 별도 투구라고 계속 표시한다. 외부 어깨 힘·일 및 일반 관성에 의존하며, 관측 궤적을 재현하기 위해 계산한 입력을 사용한다. 최종 조건부 공 속도는 약 113.5 km/h, 관측 구속은 약 130.2 km/h였고 이 차이가 이번 영상 작업으로 줄어든 것은 아니다. 화면 중간의 공 속도는 해당 시각의 모델 속도이며 최종 이탈 구속과 구분해야 한다.

## 출처와 이용 조건

OpenBiomechanics / Driveline Baseball, Theia Markerless. 원자료를 잘라 축소하고 관절선을 추가한 파생 영상이다. [OBP 데이터 이용 조건](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md): CC BY-NC-SA 4.0 및 프로 스포츠 조직·금융분석업 관련자에 대한 추가 제한이 적용된다. 원자료와 파생 영상의 조건을 보존하며 새 영상은 같은 조건으로 제공한다. 상류 프로젝트의 승인이나 정확도 보증을 뜻하지 않는다.
