# 물리·학습 결합 조사 출처와 확인 범위

확인일: 2026-10-29. 외부 소스는 읽기만 했다. 원문 HTML·다운로드한 코드·메타데이터는 git에서 제외되는 `artifacts/hybrid_sources/`에 두었다. 외부 코드를 설치하거나 데이터·동영상을 외부에 업로드하지 않았다.

## 저장소별 확인

- MuscleMimic: [고정 커밋 47b653db2bce25de834360d2ee6b2dd88142bd78](https://github.com/amathislab/musclemimic/tree/47b653db2bce25de834360d2ee6b2dd88142bd78). `README.md`, `LICENSE`, `pyproject.toml`, `musclemimic/core/reward/trajectory_based.py`, `musclemimic/core/mujoco_mjx.py`, `musclemimic/algorithms/ppo/loss.py`를 내려받았다. README·요구 패키지·활성화 비용 부분을 직접 확인했다. Python >=3.11, NumPy 2.2.6, MuJoCo/MJX 3.4.0, JAX/JAXlib 0.7.2, Warp 1.10.0 pin. Apache-2.0은 repository 코드의 조건이며 SMPL·데이터·checkpoint의 별도 라이선스를 대체하지 않는다.
- DiffMJX: [고정 커밋 b5453eac0b19274754ef2ae86840ce45ad62480a](https://github.com/martius-lab/diffmjx/tree/b5453eac0b19274754ef2ae86840ce45ad62480a). `README.md`, `LICENSE`, `pyproject.toml`, `setup.sh`, `experiments/02_tossobjects/run.py`, `experiments/04_time-toss/run.py`를 확인했다. Apache-2.0. Python >=3.12,<3.14, MuJoCo/MJX 3.3.1, jax[cuda12]>=0.8.0, local fork 경로 의존. setup.sh는 SSH clone·bash·Python 3.13 환경을 선택한다. 스크립트를 실행하지 않았다.
- DiffMJX 구성요소: API로 `martius-lab/mujoco`, `martius-lab/mjx_diffrax`, `a-paulus/softjax`가 공개임을 확인했다. README의 private 접근 문구와 현재 공개 상태가 다르므로 비공개 접근을 확정 장애로 보고하지 않는다. [mjx_diffrax 고정 커밋 a7756386337cdebaba409936a37b34662303b66c](https://github.com/martius-lab/mjx_diffrax/tree/a7756386337cdebaba409936a37b34662303b66c)의 README, pyproject, `mjx_diffrax/config.py`, `mjx_diffrax/integrate.py`를 확인했다. CPU 설치 예제도 있으나 접촉 reverse-mode에는 수정 solver가 필요하다. 별도 LICENSE 파일은 tree에서 찾지 못했고 pyproject는 Apache-2.0을 선언한다. 우산 저장소의 LICENSE만으로 모든 dependency 상태가 점검됐다고 하지 않는다.
- AddBiomechanics: [고정 커밋 8fe69cd8be3786a13e0cb90f4451fb14d7e3cb5c](https://github.com/keenon/AddBiomechanics/tree/8fe69cd8be3786a13e0cb90f4451fb14d7e3cb5c). `README.md`, `LICENSE.txt`, `server/engine/README.md`, `server/engine/requirements.txt`, `server/engine/src/engine.py`, `server/engine/src/dynamics_pass/dynamics_pass.py`, `server/engine/src/kinematics_pass/subject.py`, `cli/setup.py`를 확보했다. 파이프라인·질량 변경·GRF 경로와 라이선스를 확인했다. 현재 LICENSE는 GPLv3에 처리·생성 모션 데이터 공개 조건을 더한다. GitHub API 분류는 NOASSERTION이다. 엔진 요구는 NumPy 1.24.4, nimblephysics 0.10.52.1 등이다. 현재 Python 3.12 Windows용 실제 설치 성공 여부를 시험하지 않았다.
- OpenSim core/Moco: [고정 커밋 dbf701f56caa59bef126938f5c24829f19559d9a](https://github.com/opensim-org/opensim-core/tree/dbf701f56caa59bef126938f5c24829f19559d9a). `README.md`, `LICENSE.txt`, `Bindings/Python/README.md`, `Bindings/Python/examples/Moco/example3DWalking/exampleMocoTrack.py`, `OpenSim/Moco/MocoOutputConstraint.h`, `OpenSim/Moco/MocoGoal/MocoOutputGoal.h`, `OpenSim/Moco/MocoGoal/MocoControlGoal.h`를 확인했다. Apache-2.0; CasADi 등 별도 dependency 조건이 있다. 여기서 조회한 main의 API가 설치된 OpenSim 4.6 전체 API와 같다고 가정하지 않는다. MocoOutputGoal의 적분 endpoint constraint는 source에서 확인했으며 실제 능력 예산용 Output 구성과 설치된 버전의 지원 여부는 별도 구현·실행 검증 사항이다.
- Bayesian muscle inference: [고정 커밋 be6500f8cc78d10a838668c6b9a2dbb1e41a86d8](https://github.com/russelljohnson95/MSM_Modeling_and_MCMC/tree/be6500f8cc78d10a838668c6b9a2dbb1e41a86d8). `README.md`, `LICENSE`, `Arm_16_Model/Arm16_CRBF_6musc_mcmc_parallel.m`, `Arm_16_Model/Arm16_CRBF_6musc_parallel.m`, `Arm_16_Model/Arm16_Figure_FromChains.m`를 확인했다. MIT. likelihood/prior와 mcmcrun 연결을 확인했다. README는 MATLAB Parallel Computing Toolbox·OpenSim 3.3을 요구하며 OpenSim >=4에서 scripting 변경 때문에 작동하지 않는다고 명시한다.

위 SHA는 조회 당시 각 저장소 default branch의 head다. 전체 repository의 모든 파일을 감사하거나 예제를 실행한 것은 아니다. `확인`은 위에서 특정한 부분의 정적 원문 대조를 뜻한다.

## 논문 확인 수준

1. [MuscleMimic 전문 v1](https://arxiv.org/html/2603.25544v1): 본문, 생체역학 검증, 모델·근육 특성, 보상, 한계 확인. preprint이며 conference acceptance를 주장하지 않는다.
2. [DiffMJX 전문 v2](https://arxiv.org/html/2506.14186v2): gradient 문제·CFD·실제 cube 식별·MPC·추가 실험 확인. [저자 프로젝트](https://sites.google.com/view/diffmjx)와 공식 README의 ICLR 2026 표기를 확인했다. OpenReview 본문 접근은 challenge에 막혀 arXiv 저자 원문으로 확인했다.
3. [AddBiomechanics 출판 전문](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295152): 본문·표식/GRF 식별·실험·한계 확인. 초기 검색에 나온 PMC10312696은 preprint이며 최종 판단·숫자는 출판본을 사용했다.
4. [Moco 출판 전문](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493): 문제 구성·검증·한계 확인. 2020 논문의 당시 제한과 2026 main 코드의 기능은 구분했다.
5. [Bayesian inference 전문](https://pmc.ncbi.nlm.nih.gov/articles/PMC8944069/): 생성 참조 데이터·팔꿈치 likelihood/prior·수렴 제한 확인. 실제 근육력 관측 검증으로 해석하지 않았다.
6. [Beyond Kinematics v1](https://arxiv.org/html/2609.21909v1): 2026-09 preprint의 보행/EMG 비교·제한·결론을 추가 대조했다. 새로운 야구 성과나 일반적인 엔진 순위로 해석하지 않았다.
7. [Diff-MSM v1](https://arxiv.org/html/2508.13303v1): §III–IV의 팔 모델·관측 입력·simulation 평가 확인. 이번 조사에서 공식 코드 링크는 찾지 못했다. 공개 코드가 존재하지 않는다는 증명은 아니다.
8. MAGNET: [저자 페이지](https://sites.google.com/mrl.snu.ac.kr/jungnam/home), [연구실 publication 목록](https://sites.google.com/view/snuimo/publications), [SIGGRAPH 공식 프로그램](https://s2025.conference-schedule.org/presentation/?id=papers_277&sess=sess156)과 PDF 연결만 확인했다. PDF 전문·공식 구현·라이선스 미확인. 핵심 후보 다섯 개의 근거로 세지 않았다.

## 플랫폼과 현재 설치 상태

읽기 전용 명령으로 확인한 LAB/.venv는 Python 3.12.13, MuJoCo 모듈 존재, OpenSim/JAX/nimblephysics 모듈 부재였다. OpenSim import는 ModuleNotFoundError를 반환했다. 시스템 기본 Python은 3.14.3이다. 라이브러리를 설치하거나 공용 환경을 바꾸지 않았다.

플랫폼 지원, Python 버전 범위, 현재 설치, 예제 실행 성공, 야구 검증은 서로 다른 판단이다. Moco 원문은 Windows 지원을 설명하지만 현재 환경에서 실행한 것은 아니다. MuscleMimic 공식 README의 학습 경로는 Linux+NVIDIA다. [JAX 설치 문서](https://docs.jax.dev/en/latest/installation.html)는 Windows native CPU와 NVIDIA GPU의 지원을 구분하며 native NVIDIA는 미지원, WSL2는 experimental이다.

[MuJoCo 공식 MJX 문서](https://mujoco.readthedocs.io/en/stable/mjx.html)는 MJX-JAX와 MJX-Warp를 구분한다. Warp의 병렬 속도가 빠르다는 이유로 automatic differentiation도 지원한다고 추정하면 안 된다. 확인한 공식 문서는 MJX-Warp가 autodiff를 지원하지 않는다고 명시한다. 학습 정책의 PPO 학습과 dynamics를 통한 기울기 계산도 구별해야 한다.

## 문서 검증

문서만 추가했다. 기존 프로토콜·stage1 결과·소스 함수는 수정하지 않았다. 외부 소스 receipt, 링크 형식, 파일 범위, 기존 파일 diff를 Python과 Git으로 확인하고 `git diff --check`를 실행했다. 검증 출력은 ignored artifacts에 보관했다. 모델 학습·Moco solve·실제 야구 성능 시험을 수행했다는 주장은 없다.

Python 검사에서 문서 2개·Markdown 링크 40개·출처 캐시 파일 64개(메타데이터·텍스트 변환본 포함)를 확인했다. 회전하는 손 좌표계와 움직이는 상대점으로 만든 임의의 예제에서 본문의 속도 식을 중앙 유한차분과 비교했으며 최대 차이는 3.91e-11 m/s였다. 두 동일 moment arm 근육의 힘 분배 (0.8, 0.2)와 (0.2, 0.8)은 같은 순토크와 같은 제곱 effort를 갖는다는 산술도 실행 확인했다. 두 검사는 수식·논리 설명을 확인한 예제이며 특정 선수의 비식별성을 증명하거나 실제 공속도를 검증한 실험이 아니다. `verification.json`에 해당 제한과 source SHA-256을 함께 저장했다.
