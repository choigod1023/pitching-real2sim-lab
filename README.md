# 야구 투구 재현 연구실

실제 C3D 투구와 고정된 분절 길이를 갖는 MuJoCo 관절 모델을 나란히 재생하는 초기 연구 도구입니다.

**현재는 운동학적 재생입니다. 힘으로 동작을 생성하는 제어기, 동작 최적화, 구속 예측, 실제 선수 개선 효과는 구현하지 않았습니다.**

## 실행

Python 3.12 환경에서 `pip install -r requirements.txt` 후:

```powershell
python preview.py --data-root PATH/TO/openbiomechanics/baseball_pitching/data
python preview.py --trial PATH/TO/TRIAL.c3d
```

현재 PC에서는 상위 `pitching-lab` 폴더의 `04 - Pitching comparison.lnk`를 실행하세요. MuJoCo 화면과 한글 제어창이 열립니다.

- 청록색: 실제 측정 마커 및 마커에서 계산한 연결선
- 주황색: 길이가 고정된 단순 관절 모델
- 제어창: 재생/일시정지, 0.1~1배속, 프레임 탐색
- MuJoCo 화면: 마우스 회전 및 확대

`--check-only`는 검증 JSON과 미리보기 PNG를 `artifacts/`에 저장합니다. `--smoke-seconds 4`는 GUI 재생 후 자동 종료합니다.

## 데이터 처리와 한계

데이터는 [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics)의 외부 자료를 사용하며 이 저장소에 포함하지 않습니다. 해당 데이터의 라이선스와 프로 스포츠 조직 관련 별도 조건을 준수해야 합니다. 의존 패키지 버전은 `requirements-lock.txt`에 기록합니다.

1. C3D 단위를 확인해 미터로 변환합니다. 실험실 Z축은 보존하고 수평 원점만 이동합니다.
2. 팔꿈치·무릎·손목·발목의 쌍 마커 중점을 사용합니다. 어깨와 고관절도 마커 기반 대리점이며 해부학적으로 보정된 관절 중심이 아닙니다.
3. 분절 길이는 해당 투구의 중앙값으로 고정합니다. 원본의 분절 방향을 관절 회전으로 옮깁니다.
4. 매 프레임 관절 상태를 직접 지정하고 `mj_forward`로 위치를 갱신합니다. `mj_step`에 의한 동작 생성은 아닙니다.
5. 관절은 단순 ball joint입니다. 해부학적 관절 제한, 근육, 인대, 힘, 투구 제어기 및 공 release를 모델링하지 않습니다.

누락된 마커 샘플은 선형 보간하며 개수를 보고합니다. 긴 결측이나 끝부분 결측이 있는 파일은 연구 분석 전에 별도 검토해야 합니다. 기본 예제에는 보간된 샘플이 없습니다.

재현 거리 지표는 마커 기반 대리점과 단순 모델의 거리입니다. 동작 추정 모델의 정확도나 실제 관절 중심의 정확도가 아닙니다. 기본 투구로 모델 길이를 맞추고 같은 투구를 비교하므로 독립적인 검증도 아닙니다.

## 연구 기여의 경계

이 재생 도구 자체에 신규성을 주장하지 않습니다. 투구 모방과 예측 최적화는 기존 연구가 있습니다.

- [Attias et al. 2026, 예측 근골격 시뮬레이션](https://doi.org/10.1007/s11044-026-10143-y)
- [Templin et al. 2024, 투구 최적제어](https://commons.nmu.edu/isbs/vol42/iss1/94/)
- [HCI Korea 2026, 강화학습 기반 물리 시뮬레이션에서의 야구 투구 동작 모방](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12745876)

후속 가설은 **기존 폼 변경량과 힘 요구를 제한한 개인별 수정이 보편적 수정이나 무제약 최적화보다 실행 가능한 개선안을 제공하는가**입니다. 비교군은 무변경, 단순 공통 수정, 무제약 최적화, 개인별 제한 최적화입니다. 우선 force-driven 재현과 공 모델을 검증해야 하며, 편안함과 실제 성능 향상은 선수 실험이 필요합니다. 이 가설의 독창성과 효과는 아직 입증되지 않았습니다.

## 검증

```powershell
python -m unittest -v
python preview.py --check-only
python preview.py --smoke-seconds 4
```

테스트는 회전 좌표계, 퇴화한 분절, 기록 데이터의 분절 방향과 고정 길이, 표시용 평행이동을 확인합니다. C3D가 없으면 기록 데이터 검사는 건너뜁니다. 재현 지표와 이미지에는 원자료에서 유도한 정보가 있으므로 `artifacts/`는 Git에서 제외합니다.
