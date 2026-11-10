# 릴리스 직후 공 관측 역추적 결과

앱 적용 기준은 미통과했다. 회귀 영상 21개와 미사용 투구 5개에서 기존 앱이 반환한 모든 좌표와 적합 플래그를 보존했다. Prime 카메라 22의 일치 좌표 24개도 동일했다. 이미 사용한 야외 #2의 F80에서 실제 영상 후보를 찾아 6/7에서 7/7이 됐다. 추가한 점과 독립 영상 기준점의 차이는 가로 960px 환산 0.779px였다.

## 미사용 투구

| 출처 | 릴리스 직후 일치 | 전체 일치 | 전체 누락 | 잘못 찍은 점 | 공이 없는 프레임의 오검출 |
|---|---:|---:|---:|---:|---:|
| MLB 3경기 3구 | 6/6 → 6/6 | 38/38 → 38/38 | 0 → 0 | 0 → 0 | 0 → 0 / 25 |
| 야외 2구 | 1/4 → 1/4 | 6/14 → 6/14 | 7 → 7 | 1 → 1 | 8 → 8 / 39 |

왼쪽은 기존 앱, 오른쪽은 역추적이다. 가로 960px 환산 8px 이내의 좌표를 일치로 센다. MLB 13프레임과 야외 17프레임은 판독이 모호해 정답·오검출 판정에서 제외했다. MLB 777438·777430·777427은 서로 다른 경기다. 야외 #5·#19는 화면 구도와 포수가 다르지만 같은 야구장 자료이며 동일 투수 여부는 미확인이다. 원본과 판독 프레임의 해시를 검증했다. [야외 원본 목록](https://huggingface.co/datasets/dyland222/international_amateur_baseball_catcher_video_dataset)

## 제한이 걸린 위치

- #5에서는 F116이 비었고 F117 이후 표시된 공은 있었다. 전체 앱 적합 검사를 통과한 점이 없어 역추적이 시작하지 못했다. #19에서는 첫 적합점이 릴리스 범위 [68, 69]보다 이른 F61이고, F71의 앱 좌표는 기준점에서 65.309px 떨어졌다. 두 영상의 F116과 F70에서 각 검출기가 모두 기준점 근처의 실제 후보를 냈다.
- 추가 진단에서 야외 기준점 14/14에 대해 기존 앱 검출기에 8px 이내 후보가 이미 있었다. 개별 후보가 있다는 사실만으로 경로 전체의 정체를 판정할 수 있는지는 확인되지 않았다. 이 두 영상에서는 검출 후보를 더 늘리기보다 릴리스 이후 실제 후보로 경로를 선택하는 단계가 검증 대상이다.
- 이전 후속 확인 방식은 미사용 야외의 공 없는 프레임에서 오검출을 8→2로 줄였으나 일치는 6/14로 같았다. 이번 역추적은 기존 반환점을 그대로 유지하는 설계이므로 기존 오검출도 남았다. MLB는 이 표본에서 기존 방식이 모든 판독 기준점을 맞혀 개선 여지를 평가하지 못했다.
- 같은 야외 투구로 제안한 다음 방식은 아직 평가하지 않았다. 이 결과는 2D 원영상에서의 공 중심과 검출 경로에 한정된다. 독립 구속, 3D 릴리스, RPM, 투구 성능 변화는 미검증이다.

## 재현과 검산

- 기준 고정 커밋 8545b61, 실험 코드 고정 커밋 9f45ad9.
- 입력 SHA-256 233234bd55f57f46f0cbf12333a955e8a672ac282c43dfc56201849e8dc4f20f.
- 영상 비교와 후보 진단은 로컬에 보존했다. 원본 영상, 원시 판독 좌표, 겹침 영상은 커밋에 포함하지 않았다. 비교 영상은 25fps에서 150프레임을 모두 디코드했고 복구 프레임과 오검출 프레임이 포함된 대표 화면을 확인했다.
- 명령과 결과

    python -m experiments.reverse_release_ball.study manifest.json comparison.json
    python -m unittest -q experiments.shared_ball_tracking.test_study experiments.release_conditioned_ball.test_study experiments.preserve_ball_path.test_study experiments.reverse_release_ball.test_study
    node --test apps/pitch-capture/core/*.test.mjs experiments/release_conditioned_ball/track.test.mjs experiments/preserve_ball_path/preserve.test.mjs experiments/reverse_release_ball/recover.test.mjs

Python 11개와 Node 31개가 통과했다. 회귀 21개에서 기존 앱 경로와 이전 후속 확인 방식의 점수가 저장된 비교 결과와 동일했다. 앱 소스 변경은 없다.
