# 시간순 구위 관측 모델 비교

`PROTOCOL.md`는 결과 전 고정한 기준이다. `FINDINGS.md`에 실제 결과를 보고한다. 기존 PR128 결과 및 Real2Sim gate는 유지한다. 연구용 예측 비교이며 훈련·의료 조언이 아니다. 실제 선수 능력·안전·편안함의 향상을 검증하지 않았다.

## 비교의 의미

L0/L1은 기존 선형 control/both다. G0/G1은 학습에서 고정한 저차 cubic B-spline 위치·속도와 구위 상호작용을 쓰는 control/both다. GB0/GB1은 G0/G1에 정규화 타자 절편을 더한다. 모든 모델에 투수 절편이 있고 연속 설계열은 학습 투수 안에서 중심화한다. basis 계수 L2 정규화 방식이며 REML 기반 smoothness 선택을 포함한 다른 연구의 GAMM을 그대로 재현한 것은 아니다.

학습에서만 knot·범위·투수 평균·표준편차·타자 열을 정한다. spline 입력은 범위를 벗어나면 clip하며 수를 남긴다. IVB basis×원 pZ 및 HB basis×원 pX의 원 위치는 clip하지 않는다. 각 모델의 기저·계수·훈련 변환은 ignored `artifacts/temporal_quality`의 적합 감사 JSON에 저장하고 결과에 경로/해시를 남긴다. 다른 기저/표준화 사이의 계수를 직접 비교하지 않는다.

6/1~7/15로 학습한 GB0만 7/16~31로 λ={1,10}을 선택한다. 결과와 입력·코드·프로토콜 해시를 `results/choice.json`에 저장한 뒤 8월 최종 명령을 실행한다. 최종 학습은6~7월, 검사는8월이다. 같은8월행으로 L0/GB0와 GB0/GB1을 비교한다. 다른 코호트인 PR128의 Brier와 전후 감소량을 만들지 않는다.

## 재현 명령

저장소 root, 실제 Python은 전용 lab의 `../.venv/Scripts/python.exe`다. 새 라이브러리를 설치하지 않았다.

```powershell
& '../.venv/Scripts/python.exe' -m coach_hypotheses.temporal_quality.collect
& '../.venv/Scripts/python.exe' -m coach_hypotheses.temporal_quality.model choose --train artifacts/temporal_quality_games/dev_train.json --validation artifacts/temporal_quality_games/dev_validation.json
& '../.venv/Scripts/python.exe' -m coach_hypotheses.temporal_quality.model final --train artifacts/temporal_quality_games/dev_train.json --validation artifacts/temporal_quality_games/dev_validation.json --test artifacts/temporal_quality_games/final_test.json --collection-audit coach_hypotheses/temporal_quality/results/collection_audit.json
& '../.venv/Scripts/python.exe' -m unittest coach_hypotheses.temporal_quality.test_model -v
& '../.venv/Scripts/python.exe' -m unittest discover -v
```

실제 수집은 격리된 `../temporal-collection-agent-20261030`에서 수행했다. 부모의 실제 choose/final 입력은 위 artifacts 경로 앞에 이 worktree 경로를 붙인 값이었다. 같은 SHA256 파일이면 경로와 무관하게 검사하며, 최종 collection audit도 agent worktree에서 읽었다. 원문·투구별 JSON·행별 예측·적합 감사는 Git에 포함하지 않는다. 공개 접근을 MLB 원자료의 CC 라이선스로 해석하지 않는다. 기존 OBP 자료의 별도 인용·라이선스는 유지한다.

최종 명령은 선택 코드/개발 입력/프로토콜 해시, 92일 수집 완전성·실패 없음·선택/파싱 경기수 일치, 3분할 원문 해시를 검사한다. 불완전 자료로 통계 조건만 충족해도 통과시키지 않는다. 처음 본 타자는 절편0이며 검사 수를 남긴다. 공식 타깃 민감도와 모든 투구의 스윙 선택은 별도 결과다. 분모가 다른 Brier를 직접 비교하지 않는다.

개발 학습곡선은 같은 5,109개 검사 행에 대한 진단이며 최종 기간이나 모델 선택에 쓰지 않았다. 학습 자료를 늘려도 반드시 개선되는 것은 아니다. 모델별 numerical refinement는 같은 목적함수의 수치 해 안정성 검사이며 원 결과나 선택을 바꾸는 튜닝이 아니다.

신체→공의 물리 연결은 `BODY_BALL_EVIDENCE.md`와 `SOURCES.json`에 확인·가정·미확인을 구분했다. StatsAPI x0/y0/z0는 실제 손 릴리스 좌표로, spinDirection은 3D 회전축으로 바꾸지 않았다. 신체 조정이 없으므로 향상된 야구 투구 영상도 생성하지 않는다.
