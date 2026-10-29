# 실측 학습·시간 파형 모형 교체 검토

조사일: 2026-10-29. 작업 기준: `ff2847f`. 논문 전문과 제3자 코드는 ignored `artifacts/`에만 보관했습니다. 기존 기준·코드·결과는 수정하지 않았습니다.

## 판단

현재 근거로는 DL을 현재 모델의 대체품으로 채택할 수 없습니다. 가장 가까운 신규 자료는 **실제 TrackMan 구속과 3D 동작을 함께 공개한 Takamido의 50명·250구**입니다. 파일 실체를 Python으로 확인했습니다. 그러나 논문의 주 평가는 선수 평균 구속이며, 같은 사람의 작은 동작 변화에 따른 구속 변화를 검증한 결과가 아닙니다. 공개 모델 체크포인트와 명확한 재사용 라이선스도 없습니다.

동작 생성까지 구현한 PMGF는 더 직접적인 대안처럼 보이지만, 최적화 목표가 실측 구속 변화가 아니라 문헌에서 고른 동작 특징 8개입니다. 이를 구속 향상 예측기로 쓰면 현재 과제의 미확인 가정을 그대로 다시 넣게 됩니다.

이 판단은 모든 DL이 불가능하다는 증명이 아닙니다. 이번에 확인한 원문·공개 코드에서 **새 선수의 개인 내 변화 예측과 실제 동작 수정 효과를 함께 입증한 대체 모델을 찾지 못했다**는 뜻입니다.

## 직접 확인한 원문 4편

### 1. Takamido 등, 2026 — 실제 구속·선수 분리 평가·공개 동작 자료

[원문 PDF](https://arxiv.org/pdf/2605.05487), [논문 정보](https://arxiv.org/abs/2605.05487), [고정 버전 코드](https://github.com/takamido/Ball_speed_pred_cross_ind/tree/29aa44eee86eff5c1fc6f5e221cb9b16a0381f16).

`Cross-individual generalizability of machine learning models for ball speed prediction in baseball pitching`는 2026-05-06 등록한 preprint입니다. 고교 4명·대학 10명·실업 20명·독립리그 3명·프로 13명입니다. TrackMan 초기 구속을 정답으로 삼고 선수별 빠른 5구를 고릅니다. 200–500Hz 광학 측정 자료를 필터링한 뒤 손목 최고 속도 시점을 release로 정의하고 -1.0~+0.2초 구간을 101점으로 바꿉니다. release 이후도 포함한 관찰 예측입니다(§2.1).

GNN-GRU·Transformer 각 6설정에 선수 LOSO를 적용합니다. 그러나 **12설정 중 outer test의 선수 평균 예측 R²가 가장 높은 모델을 선택**합니다. 독립적인 최종 검사가 아닙니다. 선택 모델 Transformer `(heads=4,d_model=64,FF=128)`의 R²=.38은 선수 평균 평가입니다. 비교 R²=.91은 각 선수 5구 중 1구를 무작위 holdout하고 나머지 4구를 훈련에 넣은 미중심화 평가입니다. 두 수치를 개인 내 변화 R²로 읽으면 안 됩니다(§2.2.1, 표2).

공개 notebook의 평균·표준편차는 해당 훈련 선수만으로 계산합니다. 이 점은 확인했으나 모델 선택의 test 재사용 문제와 별개입니다. 새 세션·동작 수정·구속 변화·효과 지속성 검증은 없습니다.

### 2. Takamido 등, 2025 — PMGF의 동작 생성과 구속 예측을 구분

[원문 PDF](https://arxiv.org/pdf/2510.10496), [코드·자료](https://github.com/takamido/PMGF/tree/ea8fc81207c9c2a28c616bc98bb65ee74aa597a6), [README](https://github.com/takamido/PMGF/blob/ea8fc81207c9c2a28c616bc98bb65ee74aa597a6/README.md).

`Personalized Motion Guidance Framework for Athlete-Centric Coaching`는 51명·각 5구로 Transformer-VAE를 학습하고 잠재 공간 보간·Evolution Strategy 탐색으로 동작을 생성합니다. fitness는 8개 동작 특징 변화량의 가중 **Nash product**입니다. 실측 구속에 맞춘 counterfactual speed loss가 아닙니다(§2.3.3, 식5–6). 평가는 재구성·DTW·생성 동작의 특징 변화이며, 실제 선수가 수정 동작을 수행한 구속 결과가 없습니다.

notebook `main_program.ipynb`는 255구 전체를 `train_data`와 `test_data` 양쪽에 넣습니다. 재구성 시연을 새 선수 일반화 검증으로 바꾸어 부를 수 없습니다. 원문은 신체 치수 변화, 실행하기 어려운 동작, 근골격·환경 제약 부재와 최고 수준 선수에 대한 OOD 문제를 한계로 인정합니다(§4.2). 손실의 속도 항은 관절 움직임 재구성 오차이며 공의 실측 구속 변화가 아닙니다.

README·notebook·의존성·모델 파일 목록을 직접 확인했습니다. 5개 `.pt` 모델(각 약24MB)이 있으나 구속 변화 예측 checkpoint가 아닙니다. Python/PyTorch·PyG·SciPy·NumPy·pandas·scikit-learn·dtaidistance와 MATLAB 분석이 필요하고 환경은 완전히 고정되지 않았습니다. README는 코드의 학술 연구 사용, 자료의 비상업 연구 사용과 재배포 제한을 명시합니다. 상업 서비스 재사용 허용으로 해석하지 않습니다.

### 3. Gomaz 등, Sensors 2021 — 개인화가 주로 평균 수준을 설명하는 사례

[원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC8622425/), [출판사](https://www.mdpi.com/1424-8220/21/22/7442), [자료 DOI](https://doi.org/10.4121/16691947), [자료 메타데이터](https://api.datacite.org/dois/10.4121/16691947).

`Individualised Ball Speed Prediction in Baseball Pitching Based on IMU Data`, DOI `10.3390/s21227442`는 네덜란드 청소년 25명·유효 224구입니다. 골반·흉골 IMU의 peak 각속도와 키를 쓰며, Stalker Pro 2로 측정한 mph 구속을 정답으로 삼습니다. R 4.0.3·rstanarm의 다층 Bayesian 회귀를 비교합니다. 검증은 **224회 투구 단위 LOO**여서 같은 선수의 다른 투구가 남습니다(§2.2–2.3).

운동학 없이 키·개인 수준을 쓰는 Personal의 R²=.973과 운동학을 추가한 Full의 .975는 거의 같습니다(표3). 이는 선수별 평균 수준을 설명하는 개인화의 이점을 보여 주지만, 현재 개인 평균 oracle보다 개인 내 변화를 잘 맞힌다는 증거가 아닙니다. 표의 RMSE=.014에는 mph 단위가 없으므로 현재 km/h MAE와 직접 비교하지 않습니다. Bayesian 불확실성도 개입 인과 식별을 대신하지 않습니다.

Europe PMC 공식 full-text XML을 내려받아 확인했습니다. DataCite는 자료 형식을 CSV, 라이선스를 CC BY 4.0으로 명시합니다. 4TU landing page 접속은 timeout이라 원본 CSV·코드의 실제 다운로드와 실행은 검증하지 못했습니다. 논문의 공개 선언과 파일 확보를 구분합니다. 반복 세션 자료라고 확인하지 않았습니다.

### 4. Bright 등, CVPR Workshops 2024 — PitcherNet의 속도 산출 원리

[원문](https://arxiv.org/html/2405.07407v1), [PDF](https://arxiv.org/pdf/2405.07407), [논문 정보](https://arxiv.org/abs/2405.07407).

`PitcherNet: Powering the Moneyball Evolution in Baseball Video Analytics`는 방송 영상에서 3D pose를 추정합니다. 속도는 학습한 구속 반응 함수가 아니라 release 근방 손목 각도 차분에 전완 길이를 곱하는 근사식으로 산출합니다(§3.3.4, 식2). MLBPitchDB의 구속 정보와 비교하지만 레이더 기기·라벨 생성 방법은 이 논문에서 확인되지 않습니다.

표4d는 ±1/2/5% 허용오차 내 정확도 43.4/68.6/94.9%를 보고합니다. MAE나 개인 내 작은 변화의 정확도와 같지 않습니다. Human3.6M의 subject split은 명시하지만 야구 구속 평가의 선수 비중복은 해당 본문에 명시하지 않습니다. 자세 복원 네트워크가 좋아졌다고 동작 변경의 구속 효과 예측이 검증되는 것은 아닙니다. 원문·저자 자료를 조사했으나 이 시스템의 공개 실행 코드·checkpoint·자료 사용권을 확인하지 못했습니다. 공식 구현이 존재하지 않는다고 단정하지 않습니다.

## 50명·250구 공개 자료의 실체 확인

고정 commit: `29aa44eee86eff5c1fc6f5e221cb9b16a0381f16`.

- [README](https://github.com/takamido/Ball_speed_pred_cross_ind/blob/29aa44eee86eff5c1fc6f5e221cb9b16a0381f16/README.md), [전체 경로 목록](https://api.github.com/repos/takamido/Ball_speed_pred_cross_ind/git/trees/29aa44eee86eff5c1fc6f5e221cb9b16a0381f16?recursive=1).
- [작은 샘플 직접 다운로드](https://raw.githubusercontent.com/takamido/Ball_speed_pred_cross_ind/29aa44eee86eff5c1fc6f5e221cb9b16a0381f16/ball_speed_pred_colab/data/Sub001/001.mat): 34,670 bytes. `Sub001`~`Sub050`에 `001.mat`~`005.mat`를 제공합니다.
- [선수 분리 notebook](https://github.com/takamido/Ball_speed_pred_cross_ind/blob/29aa44eee86eff5c1fc6f5e221cb9b16a0381f16/ball_speed_pred_colab/program/base_model_inter.ipynb), [같은 선수 비교 notebook](https://github.com/takamido/Ball_speed_pred_cross_ind/blob/29aa44eee86eff5c1fc6f5e221cb9b16a0381f16/ball_speed_pred_colab/program/base_model_intra.ipynb).
- 250개 파일 모두 HTTP 200으로 확보했습니다. 합계 8,683,840 bytes이며 `scipy.io.loadmat` 실행 결과 모두 `X=(101,15,3)` float64, `Y=(1,1)` float64이고 비유한 값이 없습니다. 선수당 5개인지 경로로 확인했습니다. 시간축을 늘린 augmentation을 독립 투구로 세지 않았습니다.
- Y 범위는 68.4792855540837~94.17526287 mph입니다. 공 구속 label이며 손 속도를 정답으로 삼은 Yang 2025와 구분합니다. [관련 원문](https://arxiv.org/pdf/2605.05487).
- 좌표 XYZ의 물리 단위·좌표계와 파일별 원 측정 표본율 메타데이터는 확인되지 않았습니다. 값의 크기만 보고 mm라고 확정하지 않습니다. 일반적 200–500Hz 측정 설명만으로 각 파일의 원 신호를 복원하거나 현재 10/20/30Hz 민감도 검증을 재현할 수 없습니다.
- Python으로 계산한 자료 기술: 선수 평균의 SD 5.377025746mph, 선수 내 SD 평균 0.523751276mph, 선수 내 구속 범위 중앙값 1.015534362mph. 개인 평균을 알고 있다고 가정한 중심화 null MAE는 0.400010914mph = 0.643755165km/h입니다. 이는 새 모델 결과가 아니라 자료 분산·기준선 계산입니다. OBP null 0.629742km/h와 단위를 맞추면 비슷한 규모입니다. 상위5구 선별에 따른 범위 제한이 있으며 OBP보다 쉽거나 더 작은 개인 내 변동이라고 단정하지 않습니다.
- repo API의 license는 `null`이며 tree에 LICENSE 파일이 없고 README에 허용 범위가 없습니다. **공개 다운로드 가능과 재사용 권리 확인은 다릅니다.** 코드·자료를 본 제품에 복사해도 된다고 판단하지 않습니다.
- 모델 `.pt`/`.pth`는 없습니다. 공개 코드에는 GPU가 없을 때 CPU를 쓰는 분기가 있지만 Colab Drive 경로·설치 cell을 손봐야 합니다. PyG wheel 경로는 `torch-2.5.1+cpu`이고 정확한 전체 환경 lock은 없습니다. notebook을 읽고 자료를 파싱했으며 DL 학습은 실행하지 않았습니다.

### 자료의 독립성

2026 원문은 PMGF2025 자료와 동일한 자료를 썼다고 밝힙니다. 인원 구성은 PMGF의 51명에 비해 실업 선수 1명이 적은 50명입니다. 두 공개 자료를 합쳐 101명이나 505개의 독립 투구로 세면 안 됩니다. 원문의 설명을 근거로 한 판단이며 어떤 선수가 제외되었는지와 255개 PMGF 원자료·250개 파일의 전수 대응은 확인하지 않았습니다.

이 코호트는 OBP와 다른 기관·일본 리그 구성을 기술합니다. 하지만 공개 익명 키만으로 OBP 참가자와 단 한 명도 겹치지 않는다고 증명할 수는 없습니다. 별도 세션 반복과 실제 작은 동작 수정을 제공한 자료로도 확인되지 않았습니다. 현재 원래 scientific_gate의 대체 검증 세트로 바로 승격하지 않습니다.

## 작은 모형 교체 실험의 권고

현재 OBP 자료에서는 모델 수를 계속 늘리기보다 한 번의 **고정된 시간 파형 PCA + ridge**를 표현 손실 점검용으로만 고려할 수 있습니다. 이미 확인한 4채널×5점 waveform20의 실패는 전체 곡선의 실패와 같지 않지만, 시간점을 늘리면 새 정보를 만드는 것도 아닙니다. 이는 확인된 성능 향상 권고가 아니라 마지막으로 구분할 수 있는 표현 가설입니다. 진행 중인 동일 waveform20의 비선형 대조와 중복해서 후보를 사후 선별하지 않습니다.

사전에 다음 한 모형을 고정하는 실행안입니다. 이번 검토에서는 구현·실행하지 않았습니다.

1. 기존 98명·399구, 기존 4채널, FP10~BR, 기존 10/20/30Hz 필터를 그대로 씁니다. 채널별 101점 선형 보간으로 전체 곡선을 표현합니다. raw 공 구속·ID·토크·힘은 입력에서 제외합니다. 실제 FP10~BR 소요 시간의 추가는 별도 가설이므로 여기에 끼우지 않습니다.
2. 기존 선수별 입력 중심화 정의를 유지합니다. 훈련 선수만으로 채널별 스케일을 정하고, 훈련 선수의 공분산을 선수균등 가중해 PCA를 학습합니다. **고정 5개 PC**를 쓰고 whiten하지 않습니다. 차원·필터·채널을 outer 성능으로 고르지 않습니다. PC는 분산이 큰 방향이므로 작은 구속 신호를 버릴 수도 있음을 기록합니다.
3. outer LOAO마다 PCA·스케일·회귀를 재학습합니다. alpha는 기존 훈련 선수 5그룹 inner CV만으로 선택하고 inner마다 PCA까지 다시 학습합니다. 훈련 loss는 선수균등, 검증 정답 평균은 회고적 점수 계산에만 사용합니다. 실제 미래 절대 구속 예측이라고 부르지 않습니다.
4. 기존 null/original12와 함께 전부 보고합니다. 현재 scientific_gate의 20Hz MAE 개선·선수 cluster bootstrap 상한<0·10/30Hz 비열등·선수 비중복·target 없는 전처리를 그대로 적용합니다. 기존 null 선택을 보고 gate나 코호트를 바꾸지 않습니다. 이미 본 개발 자료에서 통과해도 새 세션 확인 전 동작 처방 근거가 아닙니다.
5. 회귀계수·PC loading·예측 함수의 미분은 **관찰 예측의 민감도**입니다. `do(동작변경)`의 인과 효과, 동일 근력·동일 일의 반사실 조건, 실제 실행 가능성·안전·편안함을 검증하지 않습니다.

딥러닝 도입의 근거를 만들려면 최소한 선수 분리·개인 내 중심화 점수·train-only 전처리·훈련 내 모형 선택을 동시에 만족해야 합니다. 선수별 2~5구로 개인별 비선형 함수를 새로 학습하거나 신뢰할 만한 random slope를 얻는다고 가정하지 않습니다. 반복 세션과 실제 수정 전후 구속이 없는 한 모델 복잡도 증가만으로 현재 질문이 식별되지는 않습니다.

## 검증 범위

Python으로 원문 PDF/공식 XML·GitHub tree/README/notebook을 확보하고 250개 MAT 파일을 직접 파싱했습니다. 파일별 SHA-256과 수신 크기는 ignored `artifacts/cross_data_receipts.json`, 통계는 `artifacts/cross_data_audit.json`에 보관했습니다. 추적하는 [EVIDENCE.json](EVIDENCE.json)에는 고정 source commit, 원문·코드 경로, 확보 상태·해시·요약 검증을 남겼습니다. 원문·원본 동작 자료·제3자 코드를 커밋하지 않습니다. 전체 앱 테스트는 수행하지 않았으며 새 모델 성능이나 실제 구속 개선을 보고하지 않습니다.
