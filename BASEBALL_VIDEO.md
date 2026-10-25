# 야구 투구 비교 영상

최종 연구 목표는 기존 힘 능력을 높이지 않는 신체 동작 수정이 야구 구속 증가로 이어지는지 검증하는 것이다. 다른 스포츠 자료는 분석 방법의 보조 검증에만 사용하며 야구 결과를 대신하지 않는다.

영상: `artifacts/baseball-video/baseball_3034_1.mp4` (1280×720, 24fps, 15초). 전신 관절 비교와 팔 확대 화면을 차례로 보여준다. 원본 촬영 영상이 아니라 측정 관절 좌표와 물리 시뮬레이션을 3D로 렌더링했다.

OpenBiomechanics 투구 3034_1의 발 착지부터 BR까지를 사용했다. 좌측은 측정 관절 중심, 우측의 주황색 팔과 공은 MuJoCo RK4로 힘을 적분한 상태다. 회색 몸통·다리는 두 화면 모두 관측 동작이며 전신 동역학 모델이 아니다. 기존 fine-reference의 시간 간격 약 0.0868ms, 1953개 상태를 사용하고 초기화 이후 상태를 재설정하지 않았다. 재생은 감속·마지막 프레임 정지로 구성하며 BR 이후 공 비행은 만들지 않았다.

실측 구속 130.195930km/h, 조건부 모델 구속 113.466581km/h. 팔꿈치 굽힘은 어깨·팔꿈치·손목 세 점의 기하학적 각도다. 15cm 강체 그립과 일반 관성을 가정했고 근육 능력은 검증하지 않았다. 기존 동작을 재현하는 데 필요한 힘을 역산했으므로 같은 능력의 수정 동작 비교나 구속 증가를 입증하지 않는다. 모델과 실측 손 끝점도 동일한 물리 점이 아니다. 공 실측 궤적은 없다.

## 재현 및 검사

`python render_baseball_video.py`. 기존 `../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz` 및 `../openbiomechanics/baseball_pitching/data`가 필요하다. numpy, pandas, scipy, Pillow, mujoco 및 imageio-ffmpeg 0.6.0을 사용한다. 현재 imageio-ffmpeg는 `../comparison-video/artifacts/video-vendor`에서 읽는다. 글꼴은 Windows Malgun이다.

별도 전방 적분의 마지막 구속을 기존 inverse_and_rollout 결과와 1e-9km/h 허용 오차로 비교하고 시간 정렬을 검사했다. 모든 적분 상태의 유한성 및 MuJoCo 경고를 검사했고 FFmpeg 전체 디코딩 종료 코드 0, 오류 없음이었다. 이미지로 전신·확대 화면을 확인했다. 로컬 생성물은 Git에서 제외한다.

출처: [Driveline OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics). 물리 가정과 기존 검증은 `IDENTIFIABILITY_FINDINGS.md` 및 `coupled_payload.py` 참조.
