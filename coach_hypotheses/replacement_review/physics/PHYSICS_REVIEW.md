# 근골격·물리 모델 교체 검토

검토일: 2026-10-29. 범위: 논문 원문, 저자·공식 저장소, 코드·라이선스·설치 메타데이터를 읽은 문헌 조사. 새 투구 시뮬레이션이나 선수 개입 실험을 수행하지 않았다.

## 판단

**현재 모델을 바로 대체할 최우선 추천은 없다.** 검토한 네 경로 가운데 실제 공 이탈, 선수별 힘·일 능력 고정, 작은 동작 변경에 따른 실제 구속 차이의 검증을 모두 갖춘 공개 모델은 없었다. 엔진 변경으로 현재 실패 판정을 뒤집을 근거도 없다. OBP3034_1의 113.415 대 130.196 km/h, 오차 16.781 km/h는 작업 의뢰의 기존 결과이며, 이번 조사에서 다시 계산하지 않았다. 98명·399구의 within-pitcher mean baseline MAE 0.6297415987 km/h는 기존 `stage1/results/summary.json`을 Python으로 직접 읽어 확인했다. 두 오차는 서로 다른 평가 대상이다.

추가 연구를 진행한다면 **OpenSim 4.6/Moco를 유지하고 어깨 개인화와 공·손가락 관측부터 보완하는 경로**가 가장 이행 비용이 낮다. 이는 검증된 대체품 추천이 아니라, 아래 검증 관문을 통과할 때만 이어 갈 개발 경로다. MyoSuite는 별도 손가락 접촉 실험용 후보로 남길 수 있다.

## 1. Attias 등, 2026: 직접 관련성은 가장 높지만 교체 불가

[논문 DOI](https://doi.org/10.1007/s11044-026-10143-y), [Waterloo 저자 원문 39쪽](https://uwaterloo.ca/motion-research-group/sites/default/files/2026-01/mbsd_pub_baseball_fd_cattias_final.pdf). DOI·제목·2026-01-22 출판일을 Crossref에서도 확인했다. 원문 §§2.1–2.2, 3.1, Data Availability를 직접 읽었다.

- **예측 대상:** 공을 손에 강체로 연결하고, 홈 방향 손 선속도 최대 프레임을 release로 정한다. 이후 비물리 운동은 제외한다. 실제 분리·비행 구속 검증이 아니다.
- **능력 제약:** 근육 Fmax 30% 증가, reserve 토크 30% 증가, 다른 주동 액추에이터 토크는 측정 평균 peak의 2배. 기본 모델에서 같은 힘·일을 유지한 비교가 아니며, 선수별 기계적 일 상한도 확인되지 않았다.
- **수치 해석:** 93/110 mph는 최적화 목표다. 93 mph 부하 최소화가 실패하고 75 mph에서 수렴한 사례가 있다. 목표속도 달성을 독립 구속 오차로 해석할 수 없다. 초기값 변경 시 관절 궤적 RMSE <5°는 해의 일관성 지표다.
- **검증·공개:** 직접 실험 검증 부족을 인정한다. 운동을 실제 바꾼 선수의 Δ구속 검증은 제시되지 않았다. NDA 때문에 원시 데이터와 **OpenSim 모델 자체도 비공개**라고 명시한다. 재사용 가능한 연구 모델·라이선스는 확보하지 못했다.

OpenSim 4.5 기반이라는 정보만으로 4.6 이식을 보장할 수 없다. 현재 공·손 위치와 이탈 불확실성을 해결하지 못하고 비공개 모델도 필요하므로 제외한다. 비공개 데이터를 얻기 위한 연락은 하지 않았다.

## 2. Templin 등, 2024: 작은 동작 변경 최적화에 가깝지만 손속도 대리 지표

[ISBS 공식 초록·서지](https://commons.nmu.edu/isbs/vol42/iss1/94/). Vol.42, Issue 1, Article 94. 공식 페이지의 DOI는 확인하지 못했다. 공식 PDF 링크는 403으로 본문 접근에 실패했으므로 **초록 확인 수준**이다.

초록의 27.1% 감소와 1.7% 유지는 각각 어깨 내회전 관련 지표와 **peak hand velocity**에 관한 최적화 결과다. 1.7%는 실제 공속도 예측 오차가 아니다. 초록만으로 해당 어깨 지표의 단위·구현, 공 접촉, 선수별 힘·일 제약, 실제 개입 후 구속 변화 검증을 확정할 수 없다. 공개 코드·모델·라이선스도 이번 검색에서 확인하지 못했다. OpenSim/Moco 관련 초록이라는 이유만으로 실행 패키지가 공개되어 있다고 쓰지 않는다.

추가로 원문과 실행 가능한 모델이 확보되더라도, 우리의 공 이탈 관측과 독립 능력 측정이 필요하다. 현재는 재현 불가·손속도 대리 문제로 교체 후보에서 제외한다.

## 3. OpenSim Moco + 개인화 어깨: 기존 경로를 고칠 기반 도구

[Moco 논문, DOI 10.1371/journal.pcbi.1008493](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493)의 본문을 확인했다. 보행·근약화·squat-to-stand 검증은 야구 구속 변화 검증이 아니다. [공식 소스](https://github.com/opensim-org/opensim-core)는 Apache-2.0이며 LICENSE와 `MocoControlGoal.h`를 commit 고정해 직접 읽었다. 해당 goal은 제어값의 p승 적분이며 p≥2다. 힘이나 총 양의 기계적 일을 그대로 고정하는 제약으로 해석하면 안 된다.

어깨 보완 후보는 다음 두 가지다.

- [Maier 등 2024, DOI 10.1016/j.jbiomech.2024.112365](https://pubmed.ncbi.nlm.nih.gov/39426356/): 공식 초록, [저자 코드](https://github.com/stanfordnmbl/shoulder-personalization)의 README·개인화 코드·환경 파일을 읽었다. 10명 arm-raise에서 전체 marker 오차가 22 mm→14–17 mm로 줄었지만, 이는 공 이탈 오차가 아니다. 5개 정적 자세의 견갑골 촉진·측정이 필요하다. 코드가 공개되어 있으나 저장소 tree/API에서 LICENSE가 없어 **오픈 라이선스 재사용 권한은 미확인**이다. `environment.yaml`은 OpenSim4.5·NumPy1.23.5를 고정하므로 현재3.12 환경 그대로 실행된다고 할 수 없다. 저자 PDF는 로컬 다운로드했지만 본문을 추출해 읽지는 않았다.
- [Hammond 등 2025, DOI 10.3389/fbioe.2025.1547373](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1547373/full): 출판사 본문 확인. 3명의 CT·biplane fluoroscopy로 만든 synthetic marker 기반 폐쇄 사슬 어깨 모델이다. 평균 0.9 mm 및 leave-one-out 1.0–1.4 mm는 어깨 기구학 오차다. 개인화 근육과 predictive control은 향후 작업으로 남긴다. MATLAB 외부 최적화+OpenSim4.4를 쓴다. SimTK 모델·설정 링크는 존재하나 접근403이어서 파일·모델 라이선스·4.6 호환성을 확인하지 못했다. 논문 CC BY와 모델 파일의 라이선스는 별개다.

우리에게는 견갑골·상완 좌표와 모멘트암 검증, 정적 개인화 자료, 손목·손가락·공 위치 및 GRF가 추가로 필요하다. 어깨를 더 정확하게 만들더라도 임의 15 cm grip과 release proxy가 자동으로 교정되지는 않는다. 두 연구 모두 야구 동작을 실제 바꿔 Δ구속을 검증한 연구는 아니다.

## 4. MyoSuite/MyoArm + MuJoCo 접촉: 손가락 연구용으로는 유망

[MyoSuite 원 논문 DOI 10.48550/arXiv.2205.13600](https://arxiv.org/abs/2205.13600), [MyoChallenge2025 원문 DOI 10.48550/arXiv.2605.15650](https://arxiv.org/html/2605.15650v1)을 확인했다. 후자는 2026-05-15 공개 preprint이며 탁구·축구 제어 벤치마크다. 실제 야구선수의 grip-release 속도 변화 검증으로 부를 수 없다.

[MyoHub 공식 코드](https://github.com/MyoHub/myosuite)와 [모델 저장소](https://github.com/MyoHub/myo_sim)의 LICENSE를 직접 읽었고 둘 다 Apache-2.0이다. MyoArm README는 견갑대부터 손가락까지 38 DOF·63 근육을 명시한다. 변환 후 수동 조정, 접촉 형상 수동 설계, 접촉 행동에 맞춘 물성 조정을 명시하므로 원 OpenSim 모델과의 동역학 동등성을 가정하면 안 된다.

`relocate.py`와 대응 XML을 직접 읽었다. 물체는 손에 weld되지 않고 3 slide+3 hinge 관절을 가지며, 공개 예제 물체는 상자다. 표준 위치·회전 보상과 activation norm `act_reg`가 있고, **선수 baseline 대비 총 일 상한은 아니다**. 즉 실제 접촉을 구현할 기반은 있으나 야구공 grip/release와 개인 능력 조건을 새로 정의해야 한다. 본문·코드에서 우리의 목적에 맞는 실측 공 이탈 구속 MAE나 개입 Δ구속 오차는 확인하지 못했다.

추가 자료는 손가락별 자세·접촉 시점·공 위치/회전·구속, 손 크기·그립 기하, 접촉 마찰·압력 또는 이를 제한할 실험, 독립 근력·파워 측정이다. 공질량·반지름만 야구공으로 바꾸고 정책을 학습하는 것은 모델 검증이 아니다. OpenSim wrapping/모멘트암 오류가 변환으로 사라진다는 근거도 없다.

## Windows Python 3.12 설치성

[OpenSim 공식 Python 문서](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting%20in%20Python)는 4.6의 Python3.12와 Moco 포함을 명시한다. PyPI JSON을 Python으로 조회해 `opensim-4.6-2-cp312-cp312-win_amd64.whl`, `mujoco-3.17.0-cp312-cp312-win_amd64.whl`, MyoSuite3.0.0 universal wheel을 확인했다. **설치 패키지가 존재한다는 확인이며 후보 모델의 import·실행·수렴 성공 확인은 아니다.**

현재 공유 `.venv` 읽기 검사 결과 Python3.12.13, MuJoCo3.17.0 설치, OpenSim·MyoSuite 미설치였다. 의존성을 바꾸지 않았다. MyoSuite pyproject의 MJX extra는 Windows에 JAX/Brax 일부 의존성을 설치하지 않도록 조건을 둔다. CPU MuJoCo와 Linux GPU 학습 경로를 구분해야 한다. 구버전 어깨 repo 환경 파일을 현재 환경에 바로 적용하지 않는다.

## 교체 전에 통과해야 할 검증 설계

다음은 새 검증 제안이며 수행 결과가 아니다.

1. **평가 대상 고정:** 관측된 실제 공 분리 시점의 3D 공 중심 속도 `v_ball(t_release+)`와 baseline 대비 변화 `Δv`를 평가한다. 손 최고속도나 레이더 목표값을 최적화 constraint로 넣어 동일한 지표를 검증하지 않는다. 레이더는 완전히 보류한 비교 기준으로만 쓴다. 손·공 offset, release, Fmax를 레이더에 맞추지 않는다.
2. **관측 관문:** 동기화·카메라 보정된 고속 다중시점 공/손가락 관측과 GRF를 확보한다. 현재 자료에서 독립적으로 식별되지 않는 grip·이탈 변수는 불확실성 구간으로 남긴다. 정적 견갑 측정 없는 개인화는 일반 모델 가정으로 표시한다.
3. **모델 관문:** ROM 전역에서 근육 길이·모멘트암 불연속, 부착 좌표, 관성, 외력, 관절축을 검사한다. 같은 자세/입력에 대한 OpenSim↔MuJoCo 길이·모멘트암·토크·가속도 차이를 수치로 비교한다. 손·공 접촉에서는 침투량·접촉 충격량·운동량/에너지 수지와 이탈 이후 자유비행을 검증한다.
4. **능력 관문:** Fmax·힘-길이/속도·activation dynamics·최대 수축속도·수동성·reserve 허용량을 baseline과 변경 동작에서 동일하게 고정한다. 초기 속도·근활성·건의 저장 에너지도 같게 둔다. 추가로 `Σ∫max(P_muscle,0)dt ≤ W_baseline+` 및 `max_t Σmax(P_muscle(t),0)` 불증가를 강제한다. 일은 근섬유와 근건복합체 중 무엇을 측정하는지 고정하고 건의 에너지 수지를 따로 보고한다. 관절 수준 제약만 쓸 때는 동일 상한·양의 일·파워를 각각 보고하고 근육 일과 동일하다고 부르지 않는다. 음의 일·탄성 에너지 재사용·발 바닥 외력과 reserve의 추가 일을 따로 회계한다. Fmax 고정과 effort soft penalty만으로 통과시키지 않는다.
5. **절대 구속 재현:** release 근처의 공·손 위치, 시간, 방향 오차와 구속 MAE/bias/RMSE를 미사용 투구에 보고한다. 동일 선수·구종·조건 mean baseline, 현재 모델과 같은 split·필터·제외 기준을 사용한다. 모델 추가 전의 baseline MAE0.629742를 보존한다. 훈련·개인화 투구의 구속을 재현했다는 이유로 검증 통과시키지 않는다.
6. **작은 변경의 검증:** baseline 가까운 ±1°, ±3° 또는 관측 가능한 시간 변화부터 제한된 탐색을 하고, 힘·일 관문을 통과한 해의 Δv와 측정 불확실성을 제시한다. 실제 같은 선수의 반복·무작위 순서 개입에서 관측 Δv와 부호·크기·순위·예측구간 coverage를 비교한다. 우연한 투구 간 연관은 실제 동작 변경의 인과 효과를 대신하지 않는다. 이 절차 자체도 효과나 안전성을 보장하지 않는다.

## Python으로 확인한 단순 반례

직접 실행한 산술 결과는 `source_inventory.json`의 `executed_arithmetic_checks`에 있다. 다음 수치는 **실측 투구 각속도나 모델 성능이 아닌 가정 예시**다.

- `δv = ω × δr`에서 서로 수직인 5 mm offset 오차와 3000/7000 °/s를 가정하면 최대 속도 차이는 각각 **0.942/2.199 km/h**다. 10 mm이면 **1.885/4.398 km/h**다. 비교 대상 baseline0.629742보다 커질 수 있으므로 고정 grip을 더 정교한 엔진에 넣는 것만으로 작은 효과를 구분하기 어렵다. 실제 방향에 따른 값은 이 상한보다 작을 수 있다.
- 같은 100 N·m 토크 상한에서도 회전각이1.0→1.2 rad이면 양의 일은100→120 J이다. 최대 힘/토크 고정이 총 일 불증가를 뜻하지 않는 구체적 반례다. 반대로 총 일만 고정해도 시간 집중으로 peak power가 늘 수 있어 별도 제한이 필요하다.

## 재현 기록과 한계

`source_inventory.json`에 실제로 읽은 코드의 commit·정확한 URL·SHA256·줄 번호, 접근 실패, PyPI 설치 파일, 로컬 환경과 산술 결과를 기록했다. source snapshot은 ignored `artifacts/replacement-physics-review/`에만 두었다. 기존 모델·결과·판정 기준은 바꾸지 않았다. 문서만 추가했으므로 전체 기존 테스트는 재실행하지 않았고, JSON 파싱·링크 형식·소스 해시 및 산술을 검사했다. 미공개·접근 실패 자료의 세부 구현은 추정해 채우지 않았다.
