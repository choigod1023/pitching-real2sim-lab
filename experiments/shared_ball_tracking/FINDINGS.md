# 여러 야구 영상의 공 추적 비교

기존 Pitch Replay 추적기, 원형 대비 후보를 쓰는 추적기, 두 후보를 합치는 추적기를 5개 출처의 원영상 8개에서 같은 규칙으로 실행했다. 투구별로 사람이 화면의 넓은 검출 구간을 지정했다. 후보 추출 이후에는 앱의 동일한 시간 연결기를 썼다. 좌표·촬영 장비 수치를 이용해 영상마다 탐색 설정을 조정하지 않았다.

| 출처 | 평가 영상 | 보이는 기준 프레임 | 기존 앱 일치 | 원형 대비 일치 | 후보 결합 일치 |
|---|---:|---:|---:|---:|---:|
| Alfredo Harp Helú Academy 야외 포수 시점 | 1 | 7 | 0 | 7 | 0 |
| TU Delft 실내 측면, 선명한 테이크 | 1 | 12 | 10 | 0 | 10 |
| TU Delft 실내 측면, 초점 흐림 별도 검사 | 1 | 11 | 0 | 6 | 11 |
| MLB 중계 | 2 | 8 | 8 | 3 | 5 |
| OBP Prime 실험실 | 1 | 15 | 13 | 13 | 13 |
| Rapsodo 후방 고속 영상 | 2 | 65 | 2 | 0 | 0 |

일치는 가로 960px로 환산한 거리 8px 이내의 결과다. 보이는 프레임에서 놓친 점과 잘못 찍은 점을 별도 집계했다. 공 부재를 판독한 19프레임에서 기존 5회, 원형 대비 7회, 결합 2회 오검출했다. MLB·Prime·Rapsodo의 미주석 프레임은 부재로 세지 않았다. 양 후보 모두 미리 정한 출처별 개선 기준을 통과하지 못했다. 결합은 동일 입력을 다시 써 탐색했다.

- 야외 아카데미 영상은 공의 실제 위치를 별도 판독한 7개 프레임에서 원형 대비 방식이 모두 맞혔다. 좌표 차이 평균 0.88px였다. 기존 앱은 한 프레임에서 다른 위치를 출력했고 공이 아직 날아가기 전 7프레임 중 4프레임에 경로를 표시했다. 결합은 이 조기 오검출은 줄였지만 뒤의 공 대신 다른 움직임을 이어 붙였다.
- Delft 선명한 영상에서 기존 앱과 결합은 12개 중 10개를 맞혔다. 원형 대비는 공이 없는 8프레임 중 5프레임에 다른 경로를 표시했다. 초점이 흐린 영상에서는 결합이 11개 모두를 맞혔고, 평균 좌표 차이는 3.29px였다. 다만 공 부재 4프레임 중 2프레임에 점을 냈다. 이 흐림 영상의 참조 판독은 대비 보조를 썼고 뒷부분이 정지해 별도 검사로 뒀다.
- Prime은 세 방식 모두 13개를 맞혔다. 같은 13개 프레임에서 기존 2.45px, 원형 대비 0.83px였다. MLB에서는 기존 방식이 8개 모두를 맞혔고, 후보를 추가하면 5개로 줄었다.
- Rapsodo에서는 두 영상 모두 긴 경로를 이어가지 못했다. 기존 앱이 반환한 보이는 프레임 3개 중 2개만 허용 거리 이내였다. 밝은 손이나 공 윤곽의 일부를 잡은 후보를 오래 이어도 정확한 공 위치가 되지 않는다.
- 현재 후보 선택은 연결이 긴 경로에 우선순위를 둔다. 다음 실험은 릴리스 전후의 투수 손과 자유 비행 후보를 분리하고, 서로 다른 경로가 양립할 때 후보를 유지하거나 보류하는 조건을 검증한다. 게임·카메라·출처를 통째로 분리하고, 배경 움직임·가림·슬로모션에서의 오검출을 출처별로 보고한다. 이 비교는 앱의 기존 기본값을 채택하거나 교체하는 절차와 분리했다.

## 자료와 재현

- 새 아카데미 자료는 공개 데이터 카드의 원본 3.57MB, 1920×1080, 재생 약 29.97fps다. 비행 80–86번 프레임을 모델 출력을 보기 전에 판독했고, 손 겹침·뒤쪽 가림과 잔상이 섞인 프레임은 정확도에서 뺐다. 공개 카드에서 배포 라이선스와 장비 기록을 확인하지 못해 원본과 결과 영상은 로컬에 보관했다. [데이터셋](https://huggingface.co/datasets/dyland222/international_amateur_baseball_catcher_video_dataset) · [핀 고정 원본](https://huggingface.co/datasets/dyland222/international_amateur_baseball_catcher_video_dataset/resolve/efbd12069403eb80e42b2a9e23498c5f60954df7/CATCHER_VID_000003.mp4)
- Delft는 같은 투수의 두 테이크다. 준비 자세만 디코드된 별도 불완전 파일은 투구 정확도 계산에서 제외했다. [4TU 자료](https://doi.org/10.4121/17021966)
- OBP·Rapsodo 좌표는 이전 실험 영상에서 검산한 참조이며, MLB는 기존 독립 원영상 판독을 사용했다. 장비 속도·RPM은 두 모델에 입력하지 않았다. [OBP 자료](https://www.openbiomechanics.org/) · [Rapsodo 예시 출처](https://baseballaero.com/services/) · [MLB 검색 자료 정의](https://baseballsavant.mlb.com/csv-docs)
- John Gant의 CC0 영상은 공개 페이지를 확인했으나 파일 전송에서 429가 반환돼 실행 대상에서 제외했다. Open Command는 투구별 연결이 있지만 공 좌표는 YOLO 모델 출력이므로 독립 정답으로 쓰지 않았다. [Wikimedia 원본 페이지](https://commons.wikimedia.org/wiki/File:Gant_Windup.webm) · [Open Command](https://huggingface.co/datasets/tomdoyo/open-command)

    python -m experiments.shared_ball_tracking.study manifest.json comparison-union.json
    python -m experiments.shared_ball_tracking.render manifest.json comparison-union.json overlay.mp4 --font <한글 폰트>
    python -m unittest -q experiments.shared_ball_tracking.test_study
    node --test apps/pitch-capture/core/*.test.mjs

신규 검증 3개와 앱 코어 16개가 통과했다. 원본 파일과 영상별 주석, 모델 결과를 로컬에 보존했다.
