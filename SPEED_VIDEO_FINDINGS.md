# 구속을 표시한 야구 관절·동역학 영상

동일 투구 `3034_1`의 실측 관절과 힘 기반 팔·손·공 RK4 시뮬레이션을 다시 실행해 20초 영상으로 만들었다. 전신 10초와 팔 확대 10초로 구성하며 각각 마지막 3초 이상 릴리스 상태를 유지한다. 다른 투구 영상이나 Theia 재구성은 섞지 않았다.

- 영상: `artifacts/speed-video/baseball_speed_journey_3034_1.mp4` (1280×720, 24fps, 480프레임).
- 실측 기록 구속(실제 측정 시점 미확인): 130.1959296km/h.
- 시뮬레이션 BR 시점 조건부 속도: 113.466580635km/h.
- 기록 구속 − 모델 BR 속도: 16.729348965km/h. 측정 시점이 일치하는 릴리스 오차로 검증한 값이 아니며, 구속 향상이나 오차 감소를 뜻하지 않는다.
- 실측 구속은 측정 시점이 미확인된 고정 기록값이라고 표시했다. 모델 순간속도는 각 시점의 실제 적분 상태에서 공 중심 Jacobian과 일반화 속도로 계산한다. 모델 BR 시점 결과도 별도 고정 표시하며 기록 구속과의 조건부 차이는 BR 도달 때만 표시한다.

## 실행과 검증

`../.venv/Scripts/python.exe render_speed_journey.py`로 실행했다. 기존 기준 상태에서 역동역학 힘을 구한 뒤 같은 모델에서 RK4로 전방 적분했다. 초기화 이후 상태 재설정은 없다. source는 `../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz`이며 정확한 시간 간격은 0.00008678278688534213초다. 명목 1/11520초에 근접하지만 양끝 이벤트를 맞춘 실제 저장 간격을 사용했다.

저장한 1,953개 qpos/qvel 상태를 별도로 복원해 공 중심 Jacobian 속도를 다시 계산했으며 저장 속도와 최대 차이는 0km/h였다. ffmpeg 전체 디코딩에서 480프레임, 1280×720, 24fps를 확인했다. 시작·릴리스 PNG를 직접 확인하여 수치·라벨·관절 표현을 검사했다. `rollout.npz`에 qpos, qvel, 적용 힘, 기준 qpos, 시각, 속도를 보존했다. `video_manifest.json`에 원본 SHA256과 프레임별 모델 인덱스를 기록했다.

## 구속 향상을 위한 해석 범위

영상은 기준 동작 재현 단계다. 관찰 궤적에서 필요한 힘을 재계산했으므로 같은 선수 능력 아래 동작을 수정한 비교가 아니다. 일반 손 관성, 손목에서 15cm 고정 그립, 외부 어깨 입력이 존재하며 손가락 접촉·근력 능력 제약은 검증하지 못했다. 몸통과 다리는 관측 맥락이고 팔·손·공만 적분한다. 실제 공 위치 궤적의 관측 검증은 아니다. 기존 구속 부족을 숨기는 보정이나 최적화 후보는 넣지 않았다.

다음 단계에서 실측 검증과 능력 예산 검사를 통과한 뒤에만 작은 관절 동작 수정과 구속 변화를 비교해야 한다. 기록값과 모델 BR 속도에 16.73km/h의 조건부 차이가 남아 있고 측정 시점 정합성이 확인되지 않아 1km/h 향상 후보의 성공을 주장할 수 없다.

출처: [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics)의 `baseball_pitching/data/full_sig/landmarks.csv`, `metadata.csv` 및 이 저장소의 `coupled_payload.py` 물리 모델.

## Applied-load provenance

The legacy inverse-reference summary includes the unused terminal force row. It is not the applied torque peak. The actual applied wrist torque-vector peak is 32.935803 Nm, computed from `loads[:-1, 8:11]`; the manifest records this separately. This model torque is not an anatomically calibrated wrist torque.
