# 공통 규칙의 논문 근거와 적용 한계

2026-10-30 조사. 원문에서 확인한 사실, 그 사실을 참고한 설계, 아직 검증하지 않은 가정을 구분한다. 논문의 관찰적 연관성을 인과적 동작 처방으로 변환하지 않는다. 이 문서와 코드는 훈련·의료 조언이 아니다.

## 1. 선수별 보정과 부하 검사의 근거

Slowik JS et al. **Fastball Velocity and Elbow-Varus Torque in Professional Baseball Pitchers.** Journal of Athletic Training, 2019. DOI: [10.4085/1062-6050-558-17](https://doi.org/10.4085/1062-6050-558-17). [PMC 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC6485842/).

Methods, Results, Discussion을 확인했다. 프로 투수 64명의 자료에서 선수 간 단순 회귀와 선수 내 변화를 다루는 random-intercept 혼합모형을 구분했다. 구속과 팔꿈치 varus torque 관계의 R²는 각각 0.076과 0.957이었다. 후자는 해당 통계모형의 설명력이며 관절 수정으로 구속을 예측하는 우리 모델의 정확도가 아니다.

**설계 반영:** 선수별 기준을 유지하고 선수 내 변화부터 검증한다. 여러 선수의 평균 동작을 한 사람의 정답으로 쓰지 않는다. 부하 감소나 비증가는 별도 하드 검사로 확인한다. **입증되지 않은 부분:** 동일한 힘·일 아래 구속 변화가 양수인 공통 규칙의 존재, 실제 개입 효과, 현재 모델의 회전 토크와 실측 해부학적 varus torque의 동등성.

## 2. 최고값뿐 아니라 동작 구간을 다룰 근거

Nebel A et al. **Within-subject time series angular velocity differences between in-game high and low velocity fastballs in college baseball pitchers.** Sports Medicine and Health Science 7 (2025), 460–465; 온라인 2024. DOI: [10.1016/j.smhs.2024.10.001](https://doi.org/10.1016/j.smhs.2024.10.001). [저자 기관 자료](https://digitalscholar.lsuhsc.edu/soah_facpubs/53/), [출판사 공개 PDF](https://journal.hep.com.cn/smhs/EN/PDF/10.1016/j.smhs.2024.10.001).

공개 PDF 본문의 Methods/Results와 저자 기관 초록을 확인했다. 82명 대학 투수의 경기 중 가장 빠른/느린 패스트볼을 선수 내에서 비교했다. 골반·몸통·어깨·팔꿈치 각속도의 차이가 나타나는 구간이 달랐다. Methods는 300Hz의 8카메라 markerless 자료와 sidearm 제외를 명시한다. 초기 PDF open은 리다이렉트 오류, PMC open은 reCAPTCHA였으나 이후 출판사 PDF 인덱스에서 실제 6쪽 문서의 Methods(p.2)와 Results를 열람했다. 파일을 직접 내려받거나 원문 파일 해시를 만들지는 않았다.

**설계 반영:** 단계별 상태와 시계열을 고려하는 규칙을 연구한다. **현재 구현의 범위:** 개입 직전 3개 모델 상태만 쓰는 작은 시제품으로, 전신 시계열 정책은 구현하지 않았다. 논문의 유의한 구간은 최적 개입 시점·최적 각도·인과효과가 아니다. 해당 구간을 60ms 개입이나 우리 모델 좌표와 자동 대응시키지 않는다.

## 3. 모든 투구를 한 시퀀스에 맞추지 않을 근거

Scarborough DM et al. **Kinematic sequence patterns in the overhead baseball pitch.** Sports Biomechanics 19(5), 2020, 569–586; 온라인 2018. [DOI 및 출판사](https://doi.org/10.1080/14763141.2018.1503321).

**출판사 초록만 확인했다. 본문 전체 확인 필요.** 초록은 22명/208구에서 14개 시퀀스 유형과 선수 내 변동을 보고하며 이상적 순서와 변동성은 아직 정해지지 않았다고 설명한다. 출판사 직접 open은 403이었고 검색에 제공된 출판사 초록을 확인했다. 비공식 재배포 본문은 근거로 사용하지 않았다.

**설계 반영:** 골반→몸통→팔→전완→손 순서를 모든 투구의 고정 정답으로 강제하지 않는다. **입증되지 않은 부분:** 어떤 역순이나 타이밍 수정이 유리한지, 모든 선수에게 적용할 최적 간격, 부상 위험 변화.

## 4. 상태에 조건화한 공통 제어기의 방법론적 근거

Joos E, Péan F, Goksel O. **Reinforcement Learning of Musculoskeletal Control from Functional Simulations.** arXiv:2007.06669, 2020. [저자 원문](https://arxiv.org/html/2007.06669v1).

Sections 2–3을 확인했다. 모델의 현재 위치/속도와 다음 목표 상태, 현재 근육 활성도를 입력으로 근육 활성 변화량을 출력하는 제어기를 학습한다. 시험은 어깨 외전 단일 축, 최대 4근육과 무작위 목표 궤적이다. 학습과 다른 범위의 목표 궤적으로 평가한다.

**설계 반영:** 투구 ID마다 별도 답을 저장하는 대신 상태→수정량이라는 공유 함수 구조를 쓴다. **현재 구현과 차이:** DRL, 실제 근육 활성, 해당 어깨 모델, 실시간 feedback을 재현하지 않았다. 우리 시제품은 개입 직전 상태에서 전체 마지막 구간의 입력 배율을 한 번 정하는 오프라인 정책이다. 이 논문은 야구 구속 향상이나 사람으로의 전이를 입증하지 않는다.

## 5. 동역학 제약 안에서 동작을 탐색할 근거와 주의점

Attias CE, Uchida TK, Inkol K, McPhee J. **Musculoskeletal Modelling and Predictive Simulation of Baseball Pitching to Improve Performance and Mitigate Injury Using Forward Dynamics and Optimal Control.** [워털루대 연구진 공개 원고, 39쪽](https://uwaterloo.ca/motion-research-group/sites/default/files/2026-01/mbsd_pub_baseball_fd_cattias_final.pdf). 이 조사에서는 저자 공개본으로 취급하며 저널 최종 출판 상태를 확정하지 않았다.

Sections 2.2, 3.1, Discussion/Limitations를 확인했다. OpenSim/Moco 제약 최적제어, 다른 투수/투구의 초기 추정에 대한 민감도 검사, reserve actuator 및 직접 실험 검증의 제한을 설명한다. 이탈은 손의 홈 방향 선속도 최댓값으로 정의한다.

**설계 반영:** 상태를 관측에 강제로 되돌리지 않고 실제 입력으로 적분하며 비용을 검사한다. 초기 추정 민감도와 실제 다른 투구에서의 예측 검증은 구분한다. **채택하지 않은 부분:** 목표 구속을 맞춰 성공시키기, 손 최대속도를 실제 공 이탈로 간주하기, reserve actuation을 실측 근육 능력으로 해석하기. 이 원고의 결과를 우리 정책의 실제 효과 근거로 쓰지 않는다.

## 논문에 없는 이번 설계값

아래는 위 논문에서 검증한 정답이 아니라 이번 시제품의 공학적 선택이다.

- 3개 상태 특징, 3×4 계수, tanh, 학습 최소/최대 범위 밖 보류.
- 기존 후보에서 빌린 팔꿈치/전완/손목 입력 방향. 원래 외부 어깨 입력의 수정은 제외한다.
- 마지막 60ms, 2% 입력 범위, 3도/20mm/1도 자세 검사.
- 3구 적합/2구 제외 평가, 학습 최악 구속 변화 최대화, C=A∩B∩root 힘 peak 제약.

따라서 “논문으로 증명된 보편 구속 공식”이 아니라 **논문에서 참고한 원칙을 실제 코드로 검증할 수 있도록 만든 연구 가설**이다. 전신 시계열, 실제 근육 능력, 공 이탈, 미사용 선수/투구 및 실제 개입 자료가 뒷받침돼야 적용 범위를 주장할 수 있다.
