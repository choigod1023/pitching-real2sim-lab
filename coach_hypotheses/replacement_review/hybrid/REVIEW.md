# 물리·학습 결합 모델 교체 검토

조사일: 2026-10-29. 원문, 공식 저장소 README·라이선스·핵심 구현을 확인했다. 아래 다섯 후보 중 **기존 근력과 일능력을 유지하면서 작은 투구 동작 변화에 따른 실제 공 중심 초기속도 변화를 검증한 모델은 확인하지 못했다.** 곧바로 전체 모델을 바꾸기보다, 기존 해부학 모델의 제약 최적제어와 불확실성 계산을 분리해 도입하는 편이 타당하다. 최신 학습 정책은 운동 재현을 가속할 수 있지만 현재 손–공 관측의 빈틈을 채우는 실측은 아니다.

이 문서는 교체 가능성을 평가한 연구 보고서다. 새 모델을 설치·학습·튜닝하지 않았으며 기존 코드, 데이터, 판정 기준과 결과를 변경하지 않았다. 코드 확인의 고정 커밋과 실행 환경 확인은 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)에 기록했다.

## 1. 현재 실패를 무엇으로 해석해야 하는가

검토에서 고정한 기존 사실은 OBP3034_1의 손 기반 공속도 대리값 113.415 km/h와 관측 구속 130.196 km/h의 차이, raw hand에 고정한 15 cm grip, 불확실한 release다. 이 수치는 이번 조사에서 다시 측정한 결과가 아니다. 98명·399구의 개인 내 중심화 학습도 MAE 0.639517 km/h로 평균 기준 0.629742를 넘지 못했고, stable7·extended14·waveform20·MAE 근사 손실도 해결하지 못했다. [기존 진단](../../stage1_diagnostics/FINDINGS.md)의 실패 판정을 유지한다.

동역학 계산 실패, 관측 부족, 목적변수 불일치는 구분해야 한다. 손 기준점 속도를 정확히 재현하더라도 공 중심 속도는 아직 검증되지 않는다. 손 기준 좌표계에서 공 중심의 상대 위치를 r(t)라 두면, 접촉 중 운동학은 다음 관계를 가진다.

`v_ball = v_hand_origin + omega_hand cross (R_hand @ r_local) + R_hand @ r_dot_local`.

`cross`는 외적, `@`는 회전행렬과 벡터의 곱이며 omega는 세계 좌표계 각속도다. 고정 grip은 상대 이동을 0으로 놓는 가정이며, 실제 손가락 움직임과 마지막 접촉 충격량을 측정한 것은 아니다. 분리 직전·직후의 공 중심 상태와 release 시각을 별도 평가해야 한다. 관측 공 초기속도를 넣어 비행을 맞추는 것은 비행 모델 검사이며, 이 연결부의 예측 검사가 아니다.

물리 계산이나 신경망을 바꾸어도 이 식의 관측되지 않은 항이 자동으로 알려지지 않는다. 반대로 현재 모든 가능한 모형이 수학적으로 비식별이라고 증명한 것도 아니다. 검토 결론은 **현재 자료에서 개인별 미세 개입의 공속도 효과를 식별·검증했다는 근거가 없다**는 것이다.

## 2. 핵심 후보 다섯 가지

### A. MuscleMimic — 학습된 근육 제어 정책

**원문 확인:** Li 등, *Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale* (2026 preprint), [전문](https://arxiv.org/html/2603.25544v1)의 §2.3, §3, §5.1과 보상을 확인했다. 근육 제어로 전신 동작·양손 조작을 모방한다. 고정 뿌리 상체 126근육과 전신 416근육 모델을 제공한다. 야구 투수의 공속도 검증은 제시하지 않는다.

**실제 검증 단위:** 보행·달리기의 관절 운동학, 관절 모멘트, GRF, 보행 주기 평균 EMG다. 초록의 평균 운동학 상관 r=0.90을 투구속도 오차로 환산할 수 없다. EMG 분석도 일부 다리 근육의 주기 평균 패턴이다. 원문은 운동학 모방만으로 생리적 근육 동원을 보장하기 어렵다고 논의한다.

**능력 고정:** 원문 §5.1은 고속 동작을 위해 상·하체의 최대근력 Fmax와 활성화 시상수 조절을 허용한다. 따라서 모방 성공을 동일 능력 실험으로 간주하면 안 된다. 고정 Fmax, 힘–길이–속도 곡선, 시상수, 초기 근육·힘줄 상태와 별도의 일 예산을 강제해야 한다. [현재 보상 코드](https://github.com/amathislab/musclemimic/blob/47b653db2bce25de834360d2ee6b2dd88142bd78/musclemimic/core/reward/trajectory_based.py)의 활성화 제곱 페널티는 기본 계수가 0이며, 활성화 페널티를 켜도 기계적 일의 엄격한 상한과 같지 않다.

**필요 관측·실행성·판정:** 소스 동작, 개인 해부학·능력의 독립 근거, 접촉 모델이 필요하다. [공식 저장소](https://github.com/amathislab/musclemimic/tree/47b653db2bce25de834360d2ee6b2dd88142bd78)는 Apache-2.0이며 모델·데이터·SMPL 자산은 별도 조건을 따른다. README는 학습에 Linux+NVIDIA를 요구하고 추론은 Linux/macOS를 안내한다. Python 3.12 자체는 요구 범위에 들어가지만 현재 Windows 환경에서 지원·실행을 확인한 상태가 아니다. 현 모델 전체 교체보다 **검증된 제약 모델의 excitation 초기값·motion prior** 후보로 제한하는 것이 타당하다. 공개 동작과 checkpoint는 OBP3034_1 손–공 실측을 대체하지 않는다.

### B. DiffMJX / CFD — 접촉이 있는 동역학의 미분 개선

**원문 확인:** Paulus 등, *Differentiable Simulation of Hard Contacts with Soft Gradients for Learning and Control* (ICLR 2026), [전문 v2](https://arxiv.org/html/2506.14186v2)의 §3–5와 부록 E를 확인했다. 적응 시간 적분과 Contacts From Distance(CFD)를 결합한다. CFD의 부드러운 접촉은 역전파에만 사용하고 순방향 물리를 유지한다.

**실제 검증 단위:** 수치 기울기·유한차분 일치, 실제 큐브 던지기 자료의 형상·접촉 파라미터 식별, 시뮬레이션 손 조작·공 받기·bionic tennis다. 테니스는 의수에 고정한 라켓으로 공을 튕긴 뒤 근육 손으로 받는 과제다. 실제 야구의 손가락 이탈이나 근력 고정 구속 예측 검증이 아니다. [공식 예제](https://github.com/martius-lab/diffmjx/blob/b5453eac0b19274754ef2ae86840ce45ad62480a/experiments/02_tossobjects/run.py)는 초기 qvel을 직접 넣어 궤적과 기울기를 계산한다. 이를 투수로부터 release velocity를 예측한 예제로 해석하면 안 된다.

**능력 고정·관측:** 엔진은 Fmax·근육 일 예산을 자체 보장하지 않는다. 식별하려는 파라미터가 바뀔 때 관측 궤적에 미치는 영향을 계산할 뿐이다. 손–공 접촉 형상·마찰·공 중심 궤적과 release 경계를 관측하지 못하면 좋은 미분만으로 올바른 파라미터를 선택할 수 없다. CFD의 유용한 대체 기울기도 최종 개입 효과의 참미분과 동일하다고 가정하지 말아야 한다. 최종 후보는 원래 순방향 엔진과 유한차분으로 다시 확인해야 한다.

**실행성·판정:** [공식 코드](https://github.com/martius-lab/diffmjx/tree/b5453eac0b19274754ef2ae86840ce45ad62480a)는 Apache-2.0, Python >=3.12,<3.14, MuJoCo 3.3.1·MJX fork·JAX CUDA 의존성을 명시한다. README에 private repository 문구가 있으나 조회 시 세 구성요소 저장소는 공개였다. 이를 비공개 접근 장애로 단정하지 않는다. [JAX 공식 지원표](https://docs.jax.dev/en/latest/installation.html)는 Windows native NVIDIA GPU를 지원하지 않고 WSL2는 experimental로 분류한다. **접촉 관측이 확보된 뒤 gradient backend만 비교할 후보**이며 지금 교체의 우선순위는 낮다. 새 설치나 실행은 하지 않았다.

### C. AddBiomechanics — 표식·GRF를 이용한 몸체 시스템 식별

**원문 확인:** Werling 등, *AddBiomechanics: Automating model scaling, inverse kinematics, and inverse dynamics from human motion data through sequential optimization* (PLOS ONE, 2023), [출판 전문](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295152)을 확인했다. 표식 등록·분절 크기·관절 궤적을 함께 맞춘 뒤 관측 GRF로 질량·운동학을 보정한다.

**실제 검증 단위:** 다중 활동 자료의 전문가 처리 관절각·토크와 비교하고 합성 보행을 검사했다. 원문의 표식 RMS <2 cm, 잔여 힘 <외력 peak의 2%는 몸체 처리 성과다. 근육력의 직접 실측 일치나 실제 야구 공속도 검증이 아니다. 잔여 힘을 줄인 결과만으로 손–공 전달이 맞았다고 볼 수 없다.

**능력 고정·관측:** 동기화한 표식·GRF·CoP·외부 모멘트, 움직임 범위가 충분한 보정 동작이 중요하다. 질량과 marker offset을 최적화하는 과정은 운동 개입과 분리하고 이후 고정해야 한다. [동역학 구현](https://github.com/keenon/AddBiomechanics/blob/8fe69cd8be3786a13e0cb90f4451fb14d7e3cb5c/server/engine/src/dynamics_pass/dynamics_pass.py)에는 질량 조정과 외력 처리가 있다. 공이 외부 하중에서 빠지면 몸체 파라미터가 누락 하중을 흡수할 가능성을 따로 검사해야 한다. 이 도구에는 근력·일 예산 고정이나 공 release 식별이 들어 있지 않다.

**실행성·판정:** [현재 LICENSE](https://github.com/keenon/AddBiomechanics/blob/8fe69cd8be3786a13e0cb90f4451fb14d7e3cb5c/LICENSE.txt)는 GPLv3에 처리·생성 데이터 공개 의무를 더한 문서다. 단순 GPL 또는 Apache 도구로 소개하지 않는다. 소스 데이터 실행·업로드는 하지 않았다. 엔진 요구의 numpy 1.24.4·nimblephysics 0.10.52.1은 기존 환경과 분리 검토가 필요하다. **몸체 보정 방식은 유용하지만 현재 자료의 손–공 검증을 대신할 전체 모델 교체는 아니다.** 공개 표식·GRF 자료 역시 해당 투구의 접촉 관측은 아니다.

### D. OpenSim Moco + 제한된 학습 모듈 — 가장 보수적인 교체 경로

**원문 확인:** Dembia 등, *OpenSim Moco: Musculoskeletal optimal control* (PLOS Computational Biology, 2020), [전문](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493)과 공식 예제를 확인했다. 직접 collocation으로 관측 동작의 근육 제어를 추정하거나 제약 안에서 새로운 동작을 예측한다. 원문 검증은 보행·근력 약화·스쿼트 보조장치이며 야구 공속도는 아니다.

**실제 검증 단위:** 모델에 따라 운동학·GRF·EMG와 mesh 수렴을 비교한다. 작은 tracking loss나 최적화 수렴만으로 참 근육력·공속도·인과 효과를 검증한 것은 아니다. 현재 native 해부학 모델을 유지하면서 제어·탐색 부분을 바꿀 수 있어 재구축 부담이 상대적으로 작다.

**능력 고정:** [공식 MocoTrack 예제](https://github.com/opensim-org/opensim-core/blob/dbf701f56caa59bef126938f5c24829f19559d9a/Bindings/Python/examples/Moco/example3DWalking/exampleMocoTrack.py)는 residual 제어 비용을 더 크게 줄 수 있지만 소프트 페널티는 금지와 다르다. Fmax 등 생리 파라미터는 최적화 변수에서 빼고, reserve·residual의 허용 크기와 일을 별도로 제한해야 한다. [MocoOutputGoal](https://github.com/opensim-org/opensim-core/blob/dbf701f56caa59bef126938f5c24829f19559d9a/OpenSim/Moco/MocoGoal/MocoOutputGoal.h)은 출력 적분을 endpoint constraint로 제한할 수 있다. 다만 여러 근육의 양의 일 합계·순간 power 상한을 정확히 표현하려면 적절한 Output이나 custom component/goal이 필요하다. 라이브러리에 해당 이름이 있다는 것만으로 우리의 예산 제약 구현이 끝난 것은 아니다.

**필요 관측·실행성·판정:** 모델·외력·초기 상태·허용 운동 범위와 능력 예산의 독립 근거가 필요하다. Apache-2.0이며 Windows Python 3.12용 OpenSim 4.6 지원 여부와 현재 설치 여부는 별개다. 확인한 LAB/.venv에서는 OpenSim이 없으므로 이번에 Moco 실행·성공을 주장하지 않는다. **추천은 full replacement가 아니라 Moco 제약 최적제어 + 제한된 excitation policy/초기값 또는 physics surrogate다.** 학습은 해석기 속도·초기화에만 먼저 쓰고, 후보 동작의 판정은 원래 동역학에서 수행한다. 관측되지 않은 release를 신경망의 자유 힘으로 채우지 않는다.

### E. Bayesian muscle-force inference — 단일 최적해 대신 가능한 해의 범위

**원문 확인:** Johnson 등, *Using Bayesian inference to estimate plausible muscle forces in musculoskeletal models* (JNER, 2022), [전문](https://pmc.ncbi.nlm.nih.gov/articles/PMC8944069/)의 팔꿈치 예제·사전분포·수렴 한계를 확인했다. 모션과 effort 가정의 불확실성에서 가능한 근육력 조합을 추정한다. 참값은 Moco에서 생성한 참조 시뮬레이션이며 실제 피험자의 근육력을 직접 측정한 검증은 아니다.

**실제 검증 단위:** 합성 팔꿈치 각도·각속도와 참조 근육력의 범위다. 원문은 여러 근육력 조합이 유사 운동학을 만들 수 있음을 보이고 MCMC 사슬의 불완전 수렴도 보고한다. 이 결과는 야구의 식별 불가능성을 직접 증명하지 않지만, 자세 적합도가 좋아도 힘 분배가 유일하다는 주장을 피해야 할 근거다.

**능력 고정·관측:** effort 사전분포는 전기·대사·기계적 일 예산의 동등 조건이 아니다. 고정 해부학·Fmax·초기 상태·실제 능력 상한을 유지하면서 허용 가능한 제어·접촉 가정의 집합을 만들어야 한다. GRF·EMG·공 궤적은 관측 오차와 함께 추가하되 없는 채널을 합성해 실측으로 부르면 안 된다.

**실행성·판정:** [공식 코드](https://github.com/russelljohnson95/MSM_Modeling_and_MCMC/tree/be6500f8cc78d10a838668c6b9a2dbb1e41a86d8)는 MIT, MATLAB·Parallel Computing Toolbox·OpenSim 3.3을 요구하고 README가 OpenSim >=4 비호환을 명시한다. Python 3.12/OpenSim 4.6에 그대로 쓸 구현은 아니다. **개념은 즉시 채택하고 코드는 교체 후보에서 제외**한다. 현재는 불확실한 구속 개선량 하나보다, 능력 조건을 만족하는 여러 해에서 개선 방향이 유지되는지를 보고하는 것이 실용적이다. 수렴하지 않은 posterior를 정밀한 신뢰구간으로 제시해서도 안 된다.

## 3. 최신 결과에서 추가로 확인한 경계

[Beyond Kinematics](https://arxiv.org/html/2609.21909v1)는 2026년 9월 preprint로, 공통 보행 motion/EMG로 HyFyDy와 MuscleMimic을 비교한다. 비슷한 운동학에서도 근활성 일치가 다를 수 있음을 보고한다. 소규모 보행 비교의 결과를 모든 투구·모든 구현의 우열로 확대하지 않는다. 이 논문은 단순 운동학 모방 성공을 같은 생리 능력의 증거로 쓰지 말아야 한다는 추가 근거다.

[Diff-MSM](https://arxiv.org/html/2508.13303v1)은 미분 가능한 팔 모델의 근육·뼈 파라미터 동시 식별을 다룬다. 전문 §III–IV에서 손목·손가락 고정, 8근육·5자유도, 알려진 activation에서 생성한 모의 자료 중심 평가를 확인했다. 근육 파라미터 자체를 추정하므로 현재의 근력 고정 운동 개입과는 단계가 다르고 손–공 release를 풀지 않는다. 공식 구현 링크는 이번 확인에서 찾지 못해 재현 가능한 핵심 후보에 넣지 않았다.

SIGGRAPH 2025의 [MAGNET 저자 페이지](https://sites.google.com/mrl.snu.ac.kr/jungnam/home)와 [공식 프로그램](https://s2025.conference-schedule.org/presentation/?id=papers_277&sess=sess156)은 다종 동작의 근육 활성 생성 연구임을 확인시킨다. 이번에는 연결된 PDF의 본문·공식 코드를 확보하지 못했다. 초록만으로 야구 공속도 성과나 Windows 실행성을 평가하지 않고 보류한다. 검색에서 발견한 논문 수를 늘리기 위해 완독·코드 확인을 한 것처럼 기록하지 않는다.

## 4. 추천 모듈과 중단할 부분

가장 실용적인 구성은 **능력 조건을 강제한 native 동역학·최적제어 + 학습된 초기값/근육 제어 prior + 불확실성 검사**다. 몸체 보정, 동작 변경, 손–공 분리 예측, 비행의 네 부분을 별도 결과로 유지한다.

1. 몸체 보정은 표식·외력으로만 수행하고 질량·관절축·근육 특성을 개입 전후 고정한다. 관측 구속을 맞추려고 grip·release force·Fmax·시상수를 바꾸지 않는다. AddBiomechanics의 방법론을 참고할 수 있으나 현재 라이선스·관측 조건까지 검토한 뒤 별도 구현을 결정해야 한다.
2. 근력 Fmax·힘–길이–속도·activation dynamics와 운동 시간 범위, 시작 근육·힘줄 에너지를 고정한다. 양의 근육 일 `W+ = sum_m integral max(P_m,0)dt`와 순간 power의 상한도 고정한다. 총 signed work만 고정하면 음의 일로 큰 양의 일을 상쇄할 수 있으므로 충분하지 않다. baseline 실측 능력은 현재 독립 식별하지 못했으므로 '동일하게 둔 모델 능력'과 '실제 선수 능력 확인'을 구분한다.
3. 학습 residual을 사용할 경우 가능한 한 근육 excitation의 저차원 보정으로 제한한다. unrestricted joint/body force는 누락 물리를 흡수하면서 가상의 근력원이 될 수 있다. residual 힘이 꼭 필요하면 크기·작용 위치·양의 일·누적 impulse를 명시하고 주 결과와 별도로 보고한다. 외부 힘을 hidden latent로 넣고 effort penalty만 주는 구성은 동일 능력 검증에 충분하지 않다.
4. surrogate는 고정 파라미터·허용 동작 영역에서 생성한 physics transition을 근사하는 가속기로 사용한다. held-out 동작·접촉 전환·도함수·장기 rollout·에너지 잔차를 원 해석기와 비교하고, 최적화가 고른 극값 후보는 모두 원 해석기로 다시 계산한다. 학습된 transition의 빠른 계산과 실제 야구 outcome validation은 별개다.
5. 공속도 예측 단계는 관측 공 초기속도를 입력하지 않고, 손–공 접촉에서 분리 후 공 중심 속도를 출력해야 한다. 해당 투구의 공 중심·손가락·release 관측이 없으면 이 모듈의 실제 정확도를 독립 검증할 수 없다. 현재 정보로 가능한 결과는 hand-speed proxy와 가정별 민감도·가능 범위다.
6. 하나의 해만 골라 향상을 보고하지 않는다. 기존 자료에 비슷하게 맞고 능력 조건을 만족하는 여러 파라미터·제어 해에서 동일한 작은 각도·타이밍 개입을 반복한다. 효과 부호가 바뀌거나 0을 포함하면 '구속 개선 입증' 대신 불확실하다고 보고한다. 이것은 새로운 합격 기준을 만드는 제안이 아니며 기존 stage1 gate는 유지한다.

지금 중단할 부분은 손 기반 15 cm grip 대리값을 실제 공속도로 명명하는 것, 관측 구속을 넣은 비행 재현을 release 예측 성공으로 세는 것, raw hand와 실제 공의 차이를 Fmax·grip·release 힘으로 정답에 맞추는 것, 모방 reward만으로 같은 근력·일능력을 주장하는 것이다. 작은 모델을 더 복잡한 모델로 바꾸는 작업 자체를 구속 예측 개선으로 보고해서도 안 된다.

## 5. 검증 범위와 남은 결정

이번에는 원문·코드의 목표·출력·조건을 대조하고, 공개 상태·의존성·현재 interpreter를 읽기 전용으로 확인했다. Python으로 source receipt와 문서 범위를 검사했으며 새 모델을 학습하거나 야구 구속 개선 실험을 실행하지 않았다. 기존 전체 테스트를 반복할 코드 변경도 없다.

새 실측 없이 먼저 할 수 있는 일은 동일 모델 능력 조건의 명시, proxy 출력 명칭 교정안 작성, 제약 안의 운동 민감도와 비식별 가능성 검사, surrogate의 원 해석기 대비 정확도 확인이다. 여기서 얻는 것은 모델 내부 조건부 결과다. 실제 작은 자세·타이밍 개입이 선수의 공속도를 높이는지는 여전히 같은 선수의 독립된 손–공/구속 관측으로 검증해야 한다. 공개 모델·motion·GRF·합성 접촉 자료를 추가해도 해당 source 투구의 미측정 공 중심 궤적을 실측한 셈이 되지는 않는다.
